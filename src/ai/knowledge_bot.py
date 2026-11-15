"""Parent knowledge bot (retrieval-augmented generation over school documents).

Retrieval: TF-IDF (word + character n-grams) over individual lines of the documents, each carrying its
section heading; the best lines point to the sections that are passed on.
Generation: Claude answers only from the retrieved sections and cites them; when the documents don't
cover a question it says so and points to the office.
Offline: returns the best-matching lines with their sources.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from . import llm

DOCS = Path(__file__).resolve().parents[2] / "school_docs"
MIN_SCORE = 0.20  # below this the bot says it doesn't know
OFFICE = "Please contact the school office."

# Words that appear everywhere in a parent's question and carry no meaning for retrieval
DOMAIN_STOP = set(ENGLISH_STOP_WORDS) | {
    "school", "child", "children", "son", "daughter", "kid", "kids", "ward", "learner", "learners", "student",
    "students", "pupil", "does", "do", "did", "can", "could", "need", "want", "know", "tell", "please", "academy",
    "madina", "bright", "future", "parent", "parents", "ok", "okay", "hello", "hi"}

SYNONYMS = [  # applied to the question before retrieval
    (r"\bfirst term\b", "term 1"), (r"\bsecond term\b", "term 2"), (r"\bthird term\b", "term 3"),
    (r"\b(1st|one)\s+term\b", "term 1"), (r"\b(2nd|two)\s+term\b", "term 2"), (r"\b(3rd|three)\s+term\b", "term 3"),
    (r"\bschool fees?\b", "tuition fees"), (r"\bpick up\b", "pick-up"), (r"\bwear\b", "wear uniform"),
    (r"\bresumes?\b|\bre-?opens?\b|\bstarts? again\b", "reopens"), (r"\bvacat\w*\b|\bclose for holidays?\b", "vacates"),
    (r"\bmocks?\b", "mock examination"), (r"\bexams?\b", "examination"), (r"\bsick\b|\bill\b|\bmalaria\b|\bfever\b", "sick illness"),
    (r"\bend\b|\bclose\b", "vacates end"), (r"\b(hp|ap)\b", "proficiency grading"),
    (r"\bwhat time\b", "hours time"), (r"\bstart\b|\bbegin\b", "start arrive"),
    (r"\bmomo\b", "mobile money"), (r"\bcanteen\b|\bfood\b", "feeding lunch"), (r"\bmaths?\b", "mathematics"),
    (r"\blate\b", "late arrive"), (r"\bcane\b|\bcaning\b|\bbeat\b", "corporal punishment caning"),
    (r"\bbraids?\b|\bextensions?\b", "hair cornrows extensions"),
]

SYSTEM = (
    "You are the parent help assistant for {school}, a basic school in Ghana. "
    "Answer parents' questions using ONLY the school document excerpts provided. Be brief, warm and clear, "
    "in simple English (or in the language the parent wrote in). After each fact, cite its source in square "
    "brackets like [fees_2026_27.md > How to pay]. If the excerpts do not contain the answer, say you don't "
    "know and ask the parent to contact the school office. Never guess dates, amounts or policies."
)


def _expand_class_range(label: str) -> str:
    """'Class 1 to Class 3' -> 'Class 1 Class 2 Class 3' so a question about Class 2 matches."""
    m = re.match(r"(Class|JHS) (\d) (?:to|and) (?:Class|JHS) (\d)", label)
    if not m:
        return ""
    return " ".join(f"{m.group(1)} {i}" for i in range(int(m.group(2)), int(m.group(3)) + 1))


def normalise(q: str) -> str:
    q = q.lower()
    for pat, rep in SYNONYMS:
        q = re.sub(pat, rep, q)
    return q


def doc_to_text(path: Path) -> str:
    """Read .md/.txt/.docx into text where sections start with '## '."""
    if path.suffix.lower() == ".docx":
        import docx
        lines = []
        for para in docx.Document(path).paragraphs:
            t = para.text.strip()
            if not t:
                continue
            style = (para.style.name or "").lower()
            lines.append(f"## {t}" if style.startswith("heading") or style == "title" else
                         (f"- {t}" if "list" in style else t))
        text = "\n".join(lines)
    else:
        text = path.read_text(encoding="utf-8", errors="ignore")
    if "\n## " not in "\n" + text:  # no headings: make sections from blocks of paragraphs
        paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        text = "\n\n".join(f"## {p.splitlines()[0][:60]}\n{p}" for p in paras)
    return text


class KnowledgeBot:
    def __init__(self, docs_dir: Path = DOCS, school_name: str = "the school", office: str = ""):
        self.school_name = school_name
        self.office = office or OFFICE
        self.sections, self.lines = [], []
        files = sorted(f for f in Path(docs_dir).glob("*") if f.suffix.lower() in (".md", ".txt", ".docx"))
        for f in files:
            text = doc_to_text(f)
            for part in re.split(r"\n(?=## )", text):
                rows = part.strip().splitlines()
                if not rows or not rows[0].startswith("## "):
                    continue
                heading = rows[0][3:].strip()
                body = "\n".join(rows[1:]).strip()
                sec_id = len(self.sections)
                self.sections.append(dict(source=f.name, section=heading, text=body))
                header = None
                for ln in body.splitlines():
                    ln = ln.strip()
                    if not ln or set(ln) <= set("|-: "):
                        continue
                    if ln.startswith("|"):  # table row: keep the header with each row for context
                        cells = [c.strip() for c in ln.strip("|").split("|")]
                        if header is None:
                            header = cells
                            continue
                        ln = "; ".join(f"{h}: {c}" for h, c in zip(header, cells))
                        search = ln + " " + _expand_class_range(cells[0])
                    else:
                        header = None
                        search = ln
                    self.lines.append(dict(section_id=sec_id, text=ln.lstrip("- ").strip(), search=search))
        if not self.lines:
            raise ValueError("No readable documents found.")
        corpus = [f"{self.sections[l['section_id']]['section']} {l['search']}".lower() for l in self.lines]
        self.word = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words=list(DOMAIN_STOP),
                                    token_pattern=r"(?u)\b\w+\b")
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self.matrix = self._stack(self.word.fit_transform(corpus), self.char.fit_transform(corpus))

    @staticmethod
    def _stack(w, c):
        from sklearn.preprocessing import normalize
        return normalize(hstack([w * 0.65, c * 0.35]).tocsr())

    def _vec(self, q):
        q = normalise(q)
        w = self.word.transform([q])
        # questions with no meaningful words cannot be answered from character overlap alone
        return self._stack(w, self.char.transform([q])), w.nnz

    def retrieve(self, question: str, k: int = 4) -> list[dict]:
        """Top-k sections, scored by their best-matching line."""
        v, n_words = self._vec(question)
        sims = (self.matrix @ v.T).toarray().ravel() * (1.0 if n_words else 0.3)
        best = {}
        for i in np.argsort(-sims):
            sid = self.lines[i]["section_id"]
            if sid not in best:
                best[sid] = (float(sims[i]), self.lines[i]["text"])
            if len(best) >= k:
                break
        return [dict(self.sections[sid], score=s, best_line=ln) for sid, (s, ln) in best.items()]

    def answer(self, question: str, use_llm: bool = True) -> dict:
        hits = self.retrieve(question)
        if use_llm and llm.credentials_available():
            # Claude decides from the excerpts whether the answer is there
            context = "\n\n".join(f"<excerpt source=\"{h['source']} > {h['section']}\">\n{h['text']}\n</excerpt>"
                                  for h in hits)
            text = llm.ask(SYSTEM.replace("{school}", self.school_name), f"School document excerpts:\n{context}\n\nParent's question: {question}",
                           effort="low", max_tokens=4000)
            return dict(answer=text, sources=hits, mode="claude")
        if hits[0]["score"] < MIN_SCORE:
            return dict(answer=f"Sorry, I couldn't find that in the school documents. {self.office}",
                        sources=[], mode="abstain")
        top = [h for h in hits if h["score"] >= max(MIN_SCORE, hits[0]["score"] * 0.85)][:2]
        body = "\n".join(f"- {h['best_line']}  \n  _Source: {h['source']} > {h['section']}_" for h in top)
        return dict(answer=body, sources=top, mode="offline")
