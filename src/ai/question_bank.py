"""Question bank: built-in BECE-style/textbook-style items plus the school's own imported questions
(e.g. real BECE past questions or textbook items the school has the right to use)."""
from __future__ import annotations

import random

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from curriculum import LABEL_TO_LEVEL

from .bank_data import ITEMS


def builtin() -> pd.DataFrame:
    rows = []
    for subj, lv, topic, q, opts, ans, expl, diff, cog, style in ITEMS:
        lo, hi = map(int, lv.split("-"))
        letters = "ABCD"
        options = [f"{letters[i]}. {o}" for i, o in enumerate(opts)]
        answer = f"{ans}. {opts[letters.index(ans)]}" if opts and ans in letters else ans
        rows.append(dict(subject=subj, level_lo=lo, level_hi=hi, topic=topic, question=q, options=options,
                         answer=answer, explanation=expl, difficulty=diff, cognitive_level=cog,
                         type="multiple_choice" if opts else ("structured" if len(q) > 140 else "short_answer"),
                         source=f"{style} (written for this pilot)"))
    return pd.DataFrame(rows)


def from_school(df: pd.DataFrame) -> pd.DataFrame:
    """Convert imported question_bank rows to the bank format."""
    if df is None or df.empty:
        return pd.DataFrame()
    rows = []
    for _, r in df.iterrows():
        opts = [r.get(f"option_{x}", "") for x in "abcd"]
        opts = [o for o in opts if isinstance(o, str) and o.strip()]
        options = [f"{'ABCD'[i]}. {o}" for i, o in enumerate(opts)]
        ans = str(r.answer).strip()
        if options and ans.upper() in "ABCD" and len(ans) == 1 and "ABCD".index(ans.upper()) < len(opts):
            ans = options["ABCD".index(ans.upper())]
        lvl = LABEL_TO_LEVEL.get(r.class_label, 9)
        rows.append(dict(subject=r.subject, level_lo=lvl, level_hi=lvl, topic=r.topic, question=r.question,
                         options=options, answer=ans, explanation=r.get("explanation", "") or "",
                         difficulty=r.get("difficulty", "") or "Medium", cognitive_level="",
                         type="multiple_choice" if options else "short_answer", source=f"School bank: {r.source}"))
    return pd.DataFrame(rows)


class Bank:
    def __init__(self, school_df: pd.DataFrame | None = None):
        self.df = pd.concat([from_school(school_df), builtin()], ignore_index=True)
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.m = self.vec.fit_transform((self.df.topic + " " + self.df.topic + " " + self.df.question).str.lower())

    def search(self, subject: str, level: int, topic: str, k: int = 5, difficulty: str | None = None,
               exclude: set | None = None, rng: random.Random | None = None) -> pd.DataFrame:
        """Items for this subject and class, most relevant to the topic first, with some randomness for variety."""
        rng = rng or random.Random()
        ok = (self.df.subject == subject) & (self.df.level_lo <= level + 1) & (self.df.level_hi >= level - 1)
        if exclude:
            ok &= ~self.df.question.isin(exclude)
        cand = self.df[ok].copy()
        if cand.empty:
            return cand
        sims = cosine_similarity(self.vec.transform([topic.lower()]), self.m[cand.index])[0]
        cand["score"] = sims + [rng.uniform(0, 0.15) for _ in range(len(cand))]
        cand["school"] = cand.source.str.startswith("School bank")
        if difficulty:
            cand["score"] += (cand.difficulty == difficulty) * 0.1
        cand["score"] += cand.school * 0.2  # prefer the school's own questions
        return cand.sort_values("score", ascending=False).head(k)
