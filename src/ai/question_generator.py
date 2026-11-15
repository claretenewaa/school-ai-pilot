"""Curriculum-aligned question generator.

With Claude: questions for any subject and topic, guided by examples from the question bank (the school's own
BECE past questions and textbook items first) and told to vary format, context and numbers.
Offline: a mix of question-bank items and randomised Mathematics templates (several different templates per
topic, never the same question twice in one set).
"""
from __future__ import annotations

import math
import random
import re

from curriculum import LEVEL_LABEL, band

from . import llm
from .question_bank import Bank

NAMES = ["Kofi", "Ama", "Kwame", "Akosua", "Nii", "Naa", "Selorm", "Fatima", "Yaw", "Esi", "Kekeli", "Abdul",
         "Adwoa", "Kojo", "Dzifa", "Mansa", "Ibrahim", "Efua", "Elikem", "Afia"]
PLACES = ["Makola Market", "Madina Market", "Kaneshie Market", "the lorry station", "Accra Mall", "Osu", "Agbogbloshie",
          "Kantamanto", "Tema Station"]
TYPES = ["multiple_choice", "short_answer", "structured", "fill_in_blank"]

SCHEMA = {
    "type": "object",
    "properties": {"questions": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "question": {"type": "string"},
            "type": {"type": "string", "enum": TYPES},
            "options": {"type": "array", "items": {"type": "string"}},
            "answer": {"type": "string"},
            "explanation": {"type": "string"},
            "cognitive_level": {"type": "string", "enum": ["Recall", "Understanding", "Application", "Reasoning"]},
            "difficulty": {"type": "string", "enum": ["Easy", "Medium", "Hard"]},
        },
        "required": ["question", "type", "options", "answer", "explanation", "cognitive_level", "difficulty"],
        "additionalProperties": False,
    }}},
    "required": ["questions"],
    "additionalProperties": False,
}

SYSTEM = (
    "You write assessment questions for basic schools in Ghana (Class 1 to JHS 3), aligned with the GES/NaCCA "
    "Standards-Based Curriculum and the Common Core Programme, in the style of the BECE and of good Ghanaian "
    "textbooks. Use Ghanaian contexts: Ghanaian names, Ghana cedis and pesewas, local places, foods, festivals and "
    "everyday situations. Keep the language simple enough for the class level. Every question must have one clearly "
    "correct answer and a short explanation or marking guide. A teacher will review every question before use.\n\n"
    "Variety matters: in one set, no two questions may share the same structure, context or numbers. Mix the forms "
    "(multiple choice, short answer, fill in the blank, multi-part structured questions like BECE Paper 2, "
    "'spot the mistake' questions, questions read from a table or a short scenario) and the contexts (market, farm, "
    "school, lorry station, mobile money, sports, festivals, the home). Do not copy the example questions; use them "
    "only to match the style and level."
)

_BANKS = {}


def _bank(school_df=None) -> Bank:
    key = 0 if school_df is None or school_df.empty else len(school_df)
    if key not in _BANKS:
        _BANKS[key] = Bank(school_df)
    return _BANKS[key]


def generate(subject: str, level: int, topic: str, difficulty: str = "Medium", n: int = 5,
             focus: str = "Mostly reasoning and application", error_focus: str | None = None,
             use_llm: bool = True, seed: int | None = None, avoid: list | None = None,
             school_bank=None) -> tuple[list[dict], str]:
    """Returns (questions, source) where source is 'claude' or 'offline'."""
    rng = random.Random(seed)
    bank = _bank(school_bank)
    if use_llm and llm.credentials_available():
        ex = bank.search(subject, level, topic, k=4, difficulty=difficulty, rng=rng)
        examples = "\n".join(f"- [{r.source}] {r.question} " + (" / ".join(r.options) if r.options else "")
                             for r in ex.itertuples())
        prompt = (
            f"Write {n} questions.\nSubject: {subject}\nClass: {LEVEL_LABEL[level]}\nTopic: {topic}\n"
            f"Difficulty: {difficulty}\nCognitive mix: {focus}. Fewer than a quarter should be pure recall.\n"
            "Write options for multiple choice as 'A. ...', 'B. ...' and so on; leave options empty for other types. "
            f"Use at least {min(n, 3)} different question forms and {min(n, 4)} different everyday contexts.")
        if examples:
            prompt += f"\n\nExamples of the expected style (do not copy):\n{examples}"
        if error_focus:
            prompt += (f"\n\nLearners often make this kind of mistake on this topic: {error_focus}. "
                       "Include questions that expose and help fix that mistake.")
        if avoid:
            prompt += "\n\nThese questions were already generated; write different ones:\n" + "\n".join(
                f"- {a[:160]}" for a in list(avoid)[-25:])
        qs = llm.ask(SYSTEM, prompt, schema=SCHEMA, effort="medium")["questions"]
        for q in qs:
            q["topic"], q["source"] = topic, "Claude (new question)"
        return qs, "claude"
    return offline(subject, level, topic, difficulty, n, rng, bank, set(avoid or [])), "offline"


# --------------------------------------------------------------------------- offline
def offline(subject, level, topic, difficulty, n, rng, bank, avoid) -> list[dict]:
    out, seen = [], set(avoid)
    from_bank = bank.search(subject, level, topic, k=n * 3, difficulty=difficulty, exclude=seen, rng=rng)
    on_topic = from_bank[from_bank.topic.str.lower() == topic.lower()]
    related = from_bank.drop(on_topic.index)

    def add_bank(rows):
        for r in rows:
            if len(out) >= n:
                return
            if r.question not in seen:
                seen.add(r.question)
                out.append(_bank_item(r, topic))

    if subject == "Mathematics":
        add_bank(list(on_topic.head(max(1, n // 3)).itertuples()))
        variants = _variants_for(topic, level)
        rng.shuffle(variants)
        cap = len(out) + len(variants) + n // 3  # limit repeats of the same template
        for i in range(len(variants) * 4):
            if len(out) >= min(n, cap):
                break
            q = variants[i % len(variants)](rng, level, difficulty)
            if q["question"] not in seen:
                seen.add(q["question"])
                out.append(dict(q, topic=topic, difficulty=difficulty, source="Template (randomised)"))
    add_bank(list(on_topic.itertuples()) + list(related.itertuples()))
    if not out:
        out.append(dict(question=f"The offline bank has no {subject} questions for {LEVEL_LABEL[level]} yet. Add a "
                                 "Claude API key, or import your school's past questions on the Import data page.",
                        type="short_answer", options=[], answer="", explanation="", cognitive_level="",
                        difficulty=difficulty, topic=topic, source=""))
    rng.shuffle(out)
    return out


def _bank_item(r, topic):
    t = r.topic if r.topic.lower() == topic.lower() else f"{r.topic} (related topic)"
    return dict(question=r.question, type=r.type, options=list(r.options), answer=r.answer, explanation=r.explanation,
                cognitive_level=r.cognitive_level, difficulty=r.difficulty, topic=t, source=r.source)


def _mcq(rng, correct, wrong, question, explanation, cog):
    opts = []
    for o in [correct] + list(wrong):
        if str(o) not in opts:
            opts.append(str(o))
    k = 1
    while len(opts) < 4:
        cand = _bump(correct, k)
        k += 1
        if cand not in opts:
            opts.append(cand)
    opts = opts[:4]
    rng.shuffle(opts)
    L = "ABCD"
    return dict(question=question, type="multiple_choice", options=[f"{L[i]}. {o}" for i, o in enumerate(opts)],
                answer=f"{L[opts.index(str(correct))]}. {correct}", explanation=explanation, cognitive_level=cog)


def _bump(v, k):
    m = re.search(r"-?\d+(\.\d+)?", str(v))
    if not m:
        return f"{v} ({k})"
    num = float(m.group())
    new = num + k * (1 if abs(num) < 10 else max(1, round(abs(num) * 0.1)))
    txt = f"{new:.2f}" if m.group(1) else str(int(new))
    return str(v)[:m.start()] + txt + str(v)[m.end():]


def _short(question, answer, explanation, cog, kind="short_answer"):
    return dict(question=question, type=kind, options=[], answer=str(answer), explanation=explanation, cognitive_level=cog)


def _sc(diff):
    return {"Easy": 1, "Medium": 2, "Hard": 3}[diff]


def _nm(rng, k=2):
    return rng.sample(NAMES, k)


# ---- each function is one distinct question template: (rng, level, difficulty) -> question
def add_story(rng, lv, d):
    a, b = rng.randint(2, 9 * _sc(d) + 3), rng.randint(2, 9 * _sc(d) + 3)
    x, y = _nm(rng)
    item = rng.choice(["oranges", "mangoes", "bottle tops", "marbles", "pencils"])
    return _mcq(rng, a + b, [a + b - 1, abs(a - b), a + b + 2],
                f"{x} has {a} {item}. {y} gives {x} {b} more. How many {item} does {x} have now?",
                f"{a} + {b} = {a + b}.", "Application")


def add_missing(rng, lv, d):
    b, s = rng.randint(2, 9), rng.randint(10, 10 * _sc(d) + 10)
    return _short(f"Fill in the missing number: ___ + {b} = {s}", s - b, f"{s} − {b} = {s - b}.", "Reasoning",
                  "fill_in_blank")


def add_market(rng, lv, d):
    a, b = rng.randint(5, 15 * _sc(d)), rng.randint(5, 15 * _sc(d))
    x = rng.choice(["Ama", "Akosua", "Naa", "Esi", "Fatima", "Adwoa"])
    return _short(f"{x} sold {a} sachets of water in the morning and {b} in the afternoon at {rng.choice(PLACES)}. "
                  "How many sachets did she sell in all?", a + b, f"{a} + {b} = {a + b}.", "Application")


def sub_story(rng, lv, d):
    a = rng.randint(12, 20 * _sc(d) + 10)
    b = rng.randint(2, a - 2)
    return _mcq(rng, a - b, [a + b, a - b + 1, b],
                f"There were {a} pupils at assembly. {b} went back to class. How many are still at assembly?",
                f"{a} − {b} = {a - b}.", "Application")


def sub_compare(rng, lv, d):
    x, y = _nm(rng)
    a, b = rng.randint(10, 30 * _sc(d)), rng.randint(3, 9)
    return _short(f"{x} has {a + b} stickers and {y} has {a}. How many more stickers does {x} have than {y}?", b,
                  f"{a + b} − {a} = {b}.", "Reasoning")


def sub_missing(rng, lv, d):
    a = rng.randint(15, 20 * _sc(d) + 10)
    b = rng.randint(3, a - 3)
    return _short(f"Fill in the missing number: {a} − ___ = {a - b}", b, f"{a} − {a - b} = {b}.", "Reasoning",
                  "fill_in_blank")


def money_change(rng, lv, d):
    price, qty = rng.randint(2, 9 * _sc(d)), rng.randint(2, 4)
    note = 50 if price * qty < 50 else 100
    x = _nm(rng, 1)[0]
    item = rng.choice(["pencils", "erasers", "pens", "rulers"])
    return _mcq(rng, f"GH₵ {note - price * qty}", [f"GH₵ {price * qty}", f"GH₵ {note - price}"],
                f"{x} buys {qty} {item} at GH₵ {price} each and pays with a GH₵ {note} note. How much change should {x} get?",
                f"Cost = {qty} × {price} = GH₵ {price * qty}. Change = {note} − {price * qty} = GH₵ {note - price * qty}.",
                "Application")


def money_enough(rng, lv, d):
    price, qty = rng.randint(3, 8), rng.randint(3, 5)
    have = price * qty + rng.choice([-3, -2, 2, 4])
    ok = have >= price * qty
    x = _nm(rng, 1)[0]
    return _short(f"{x} has GH₵ {have}. A loaf of bread costs GH₵ {price}. Can {x} buy {qty} loaves? Explain.",
                  "Yes" if ok else "No", f"{qty} × {price} = GH₵ {price * qty}, which is "
                  f"{'not more than' if ok else 'more than'} GH₵ {have}.", "Reasoning")


def money_total(rng, lv, d):
    a, b = rng.randint(2, 15), rng.randint(2, 15)
    p1, p2 = rng.choice(["kenkey", "fish", "gari", "bread"]), rng.choice(["groundnuts", "bananas", "milk", "sugar"])
    return _mcq(rng, f"GH₵ {a + b}", [f"GH₵ {abs(a - b)}", f"GH₵ {a * b}"],
                f"Mother bought {p1} for GH₵ {a} and {p2} for GH₵ {b}. How much did she spend altogether?",
                f"{a} + {b} = GH₵ {a + b}.", "Application")


def money_coins(rng, lv, d):
    coin, total = rng.choice([(2, 10), (2, 20), (5, 20), (5, 50), (10, 50), (20, 100)])
    return _short(f"How many GH₵ {coin} notes or coins make GH₵ {total}?", total // coin, f"{total} ÷ {coin} = {total // coin}.",
                  "Understanding", "fill_in_blank")


def money_pesewas(rng, lv, d):
    ced, pes = rng.randint(1, 9), rng.choice([20, 50, 10])
    return _mcq(rng, f"{ced * 100 + pes} pesewas", [f"{ced + pes} pesewas", f"{ced * 10 + pes} pesewas"],
                f"Change GH₵ {ced}.{pes:02d} to pesewas.", f"GH₵ 1 = 100 pesewas, so {ced} × 100 + {pes} = {ced * 100 + pes}.",
                "Understanding")


def pv_digit(rng, lv, d):
    n = rng.randint(10 ** (1 + _sc(d)), 10 ** (2 + _sc(d)) - 1)
    s = str(n)
    i = rng.randint(0, len(s) - 2)
    while s[i] == "0":
        i = rng.randint(0, len(s) - 2)
    v = int(s[i]) * 10 ** (len(s) - i - 1)
    return _mcq(rng, f"{v:,}", [s[i], f"{v * 10:,}", f"{max(v // 10, 1):,}"],
                f"What is the value of the digit {s[i]} in {n:,}?", f"{s[i]} is in the {10 ** (len(s) - i - 1):,}s place.",
                "Understanding")


def pv_expanded(rng, lv, d):
    n = rng.randint(1000, 9999)
    parts = [f"{int(c) * 10 ** (3 - i):,}" for i, c in enumerate(str(n)) if c != "0"]
    return _short(f"Write {n:,} in expanded form.", " + ".join(parts), "Split the number by place value.", "Understanding")


def pv_largest(rng, lv, d):
    digs = rng.sample(range(1, 10), 3 + (1 if _sc(d) > 1 else 0))
    big = int("".join(map(str, sorted(digs, reverse=True))))
    return _short(f"Use the digits {', '.join(map(str, digs))} once each to make the largest possible number.", f"{big:,}",
                  "Put the biggest digit in the highest place value.", "Reasoning")


def mul_crates(rng, lv, d):
    c = rng.randint(3, 6 * _sc(d))
    return _mcq(rng, c * 30, [c + 30, c * 30 - 30, c * 3],
                f"A crate holds 30 eggs. A trader at {rng.choice(PLACES)} sold {c} full crates. How many eggs did she sell?",
                f"{c} × 30 = {c * 30}.", "Application")


def div_benches(rng, lv, d):
    per = rng.choice([3, 4, 5, 6])
    pupils = per * rng.randint(6, 12) + rng.randint(1, per - 1)
    return _short(f"There are {pupils} pupils. Each bench seats {per} pupils. What is the least number of benches needed "
                  "so that every pupil sits?", math.ceil(pupils / per),
                  f"{pupils} ÷ {per} = {pupils // per} remainder {pupils % per}; the remaining pupils need one more bench.",
                  "Reasoning")


def mul_missing(rng, lv, d):
    a, b = rng.randint(3, 12), rng.randint(3, 12)
    return _short(f"Fill in the gap: {a} × ___ = {a * b}", b, f"{a * b} ÷ {a} = {b}.", "Understanding", "fill_in_blank")


def frac_of(rng, lv, d):
    den = rng.choice([3, 4, 5, 6, 8])
    num = rng.choice([k for k in range(1, den) if math.gcd(k, den) == 1])
    tot = den * rng.randint(2, 6 * _sc(d))
    x = _nm(rng, 1)[0]
    ans = tot * num // den
    return _mcq(rng, f"GH₵ {ans}", [f"GH₵ {tot // den}", f"GH₵ {tot - ans}"],
                f"{x} had GH₵ {tot} and spent {num}/{den} of it on books. How much was spent on books?",
                f"{num}/{den} × {tot} = GH₵ {ans}.", "Application")


def frac_compare(rng, lv, d):
    da, db = rng.choice([5, 6, 7, 8]), rng.choice([3, 4, 9, 10])
    a = (rng.choice([k for k in range(1, 5) if math.gcd(k, da) == 1]), da)
    b = (rng.choice([k for k in range(1, db) if math.gcd(k, db) == 1]), db)
    if a[0] * b[1] == b[0] * a[1]:
        b = (b[0] + 1, b[1] + 1)
    big, small = (a, b) if a[0] * b[1] > b[0] * a[1] else (b, a)
    return _mcq(rng, f"{big[0]}/{big[1]}", [f"{small[0]}/{small[1]}", "They are equal"],
                f"Which is bigger: {a[0]}/{a[1]} or {b[0]}/{b[1]}?",
                f"Common denominator {a[1] * b[1]}: {a[0] * b[1]}/{a[1] * b[1]} and {b[0] * a[1]}/{a[1] * b[1]}.",
                "Understanding")


def frac_error(rng, lv, d):
    a, b = rng.sample([2, 3, 4, 5, 6], 2)
    x = _nm(rng, 1)[0]
    s, p = a + b, a * b
    g = math.gcd(s, p)
    return _short(f"{x} says 1/{a} + 1/{b} = 2/{a + b}. Is {x} correct? Explain.",
                  f"No. 1/{a} + 1/{b} = {s // g}/{p // g}",
                  f"Use a common denominator {p}: {b}/{p} + {a}/{p} = {s}/{p}. You cannot add the denominators.",
                  "Reasoning")


def frac_add(rng, lv, d):
    den = rng.choice([5, 7, 8, 9, 10])
    a, b = rng.randint(1, den // 2), rng.randint(1, den // 2)
    g = math.gcd(a + b, den)
    return _short(f"Simplify {a}/{den} + {b}/{den}.", f"{(a + b) // g}/{den // g}",
                  f"Same denominator: add the numerators, {a + b}/{den}" + (", then simplify." if g > 1 else "."),
                  "Application")


def dec_add(rng, lv, d):
    a, b = round(rng.uniform(1, 20 * _sc(d)), 2), round(rng.uniform(1, 20 * _sc(d)), 2)
    x = _nm(rng, 1)[0]
    return _mcq(rng, f"GH₵ {a + b:.2f}", [f"GH₵ {abs(a - b):.2f}", f"GH₵ {a + b + 1:.2f}"],
                f"{x} bought kenkey for GH₵ {a:.2f} and fish for GH₵ {b:.2f}. How much did {x} spend?",
                f"Line up the decimal points: {a:.2f} + {b:.2f} = {a + b:.2f}.", "Application")


def dec_order(rng, lv, d):
    base = rng.randint(1, 9)
    vals = [base / 10, base / 100, (base * 10 + rng.randint(1, 9)) / 100]
    rng.shuffle(vals)
    srt = ", ".join(f"{v:g}" for v in sorted(vals))
    return _short(f"Arrange in ascending order: {', '.join(f'{v:g}' for v in vals)}", srt,
                  "Compare the tenths first, then the hundredths.", "Understanding")


def dec_times(rng, lv, d):
    v = round(rng.uniform(0.1, 9.9), 2)
    m = rng.choice([10, 100])
    return _mcq(rng, f"{round(v * m, 2):g}", [f"{v / m:g}", f"{round(v + m, 2):g}"], f"Work out {v:g} × {m}.",
                f"Multiplying by {m} moves each digit {len(str(m)) - 1} place(s) to the left.", "Understanding")


def pct_discount(rng, lv, d):
    price = rng.choice([40, 60, 80, 120, 160, 200]) * _sc(d)
    p = rng.choice([10, 20, 25, 50])
    new = price - price * p // 100
    return _mcq(rng, f"GH₵ {new}", [f"GH₵ {price * p // 100}", f"GH₵ {price + price * p // 100}"],
                f"A school bag at {rng.choice(PLACES)} costs GH₵ {price}. The seller gives a {p}% discount. What is the new price?",
                f"Discount = {p}% of {price} = GH₵ {price * p // 100}. New price = GH₵ {new}.", "Application")


def ratio_share(rng, lv, d):
    r1, r2 = rng.randint(1, 4), rng.randint(2, 5)
    amt = (r1 + r2) * rng.randint(5, 20 * _sc(d))
    x, y = _nm(rng)
    return _mcq(rng, f"GH₵ {amt * r1 // (r1 + r2)}", [f"GH₵ {amt // 2}", f"GH₵ {amt * r2 // (r1 + r2)}"],
                f"{x} and {y} share GH₵ {amt} in the ratio {r1} : {r2}. How much does {x} get?",
                f"Total parts = {r1 + r2}. {x} gets {r1}/{r1 + r2} × {amt} = GH₵ {amt * r1 // (r1 + r2)}.", "Reasoning")


def pct_class(rng, lv, d):
    tot = rng.choice([20, 25, 40, 50])
    passed = rng.randint(tot // 2, tot - 1)
    return _short(f"In a class of {tot} pupils, {passed} passed the Mathematics test. What percentage passed?",
                  f"{passed / tot * 100:g}%", f"{passed}/{tot} × 100 = {passed / tot * 100:g}%.", "Application")


def pct_profit(rng, lv, d):
    cost = rng.choice([100, 200, 250, 400, 500])
    p = rng.choice([10, 20, 25, 40])
    sell = cost + cost * p // 100
    return _short(f"A trader bought a phone for GH₵ {cost} and sold it for GH₵ {sell}. Find the percentage profit.",
                  f"{p}%", f"Profit = GH₵ {sell - cost}. {sell - cost}/{cost} × 100 = {p}%.", "Application")


def simple_interest(rng, lv, d):
    P, R, Tm = rng.choice([500, 800, 1000, 1200, 2000]), rng.choice([5, 8, 10, 12]), rng.randint(2, 4)
    return _mcq(rng, f"GH₵ {P * R * Tm // 100}", [f"GH₵ {P * R // 100}", f"GH₵ {P + P * R * Tm // 100}"],
                f"Find the simple interest on GH₵ {P:,} at {R}% per annum for {Tm} years.",
                f"I = PRT/100 = {P} × {R} × {Tm} / 100 = GH₵ {P * R * Tm // 100}.", "Application")


def alg_sub(rng, lv, d):
    a, b, x = rng.randint(2, 6), rng.randint(1, 9), rng.randint(2, 5 * _sc(d))
    return _mcq(rng, a * x + b, [a + x + b, a * (x + b), a * x - b], f"Find the value of {a}x + {b} when x = {x}.",
                f"{a} × {x} + {b} = {a * x + b}.", "Understanding")


def alg_like(rng, lv, d):
    a, b, c, e = rng.randint(3, 8), rng.randint(2, 7), rng.randint(1, 2), rng.randint(1, 5)
    return _short(f"Simplify {a}m + {b}n − {c}m + {e}n.", f"{a - c}m + {b + e}n",
                  "Collect like terms: m terms together and n terms together.", "Application")


def alg_expand(rng, lv, d):
    a, b, c = rng.randint(2, 5), rng.randint(2, 6), rng.randint(1, 3)
    return _short(f"Expand and simplify {a}(x + {b}) − {c}.", f"{a}x + {a * b - c}",
                  f"{a}x + {a * b} − {c} = {a}x + {a * b - c}.", "Application")


def seq_next(rng, lv, d):
    start, step = rng.randint(1, 10), rng.randint(2, 5 * _sc(d))
    seq = [start + step * k for k in range(4)]
    return _mcq(rng, seq[-1] + step, [seq[-1] + step + 1, seq[-1] + 2 * step],
                f"What number comes next? {', '.join(map(str, seq))}, ...", f"The numbers increase by {step}.", "Reasoning")


def eq_solve(rng, lv, d):
    a, x, b = rng.randint(2, 7), rng.randint(1, 6 * _sc(d)), rng.randint(1, 15)
    return _mcq(rng, f"x = {x}", [f"x = {x + 1}", f"x = {a * x}", f"x = {a * x + 2 * b}"], f"Solve: {a}x + {b} = {a * x + b}",
                f"{a}x = {a * x}, so x = {x}.", "Application")


def eq_think(rng, lv, d):
    k, a, b = rng.randint(3, 15), rng.randint(2, 5), rng.randint(2, 12)
    x = _nm(rng, 1)[0]
    return _short(f"{x} thinks of a number, multiplies it by {a} and adds {b}. The result is {a * k + b}. "
                  "(a) Write an equation for this. (b) Find the number.", f"(a) {a}n + {b} = {a * k + b}  (b) n = {k}",
                  f"{a}n = {a * k}, so n = {k}.", "Reasoning", "structured")


def ineq(rng, lv, d):
    a, b, c = rng.randint(2, 5), rng.randint(1, 9), rng.randint(2, 6)
    return _mcq(rng, f"x < {c}", [f"x > {c}", f"x < {c + b}", f"x ≤ {c - 1}"], f"Solve the inequality {a}x + {b} < {a * c + b}.",
                f"{a}x < {a * c}, so x < {c}.", "Application")


def eq_error(rng, lv, d):
    a, x, c = rng.randint(2, 5), rng.randint(2, 6), rng.randint(1, 9)
    rhs = c - a * x
    nm = _nm(rng, 1)[0]
    return _short(f"{nm} solved {c} − {a}x = {rhs} and got x = −{x}. Is {nm} correct? Show the correct working.",
                  f"No, x = {x}", f"−{a}x = {rhs} − {c} = {rhs - c}, so x = {x}. {nm} made a sign error.", "Reasoning")


def ind_product(rng, lv, d):
    b, p, q = rng.choice([2, 3, 5]), rng.randint(2, 4), rng.randint(1, 3)
    return _mcq(rng, f"{b}^{p + q} = {b ** (p + q)}", [f"{b}^{p * q} = {b ** (p * q)}", f"{b * 2}^{p + q} = {(b * 2) ** (p + q)}"],
                f"Simplify {b}^{p} × {b}^{q}.", f"Add the indices: {b}^{p + q} = {b ** (p + q)}.", "Understanding")


def ind_stdform(rng, lv, d):
    m = rng.randint(11, 99) / 10
    e = rng.randint(3, 7) * rng.choice([1, -1])
    shown = f"{m * 10 ** e:,.0f}" if e > 0 else f"{m * 10 ** e:.{-e + 1}f}"
    return _short(f"Write {shown} in standard form.", f"{m:g} × 10^{e}", "Move the decimal point until one non-zero digit "
                  "is in front of it, and count the places moved.", "Understanding")


def ind_solve(rng, lv, d):
    b, x = rng.choice([2, 3, 4, 5]), rng.randint(2, 5)
    return _short(f"Find x if {b}^x = {b ** x}.", x, f"{b ** x} = {b}^{x}.", "Understanding", "fill_in_blank")


def ind_divide(rng, lv, d):
    b, p, q = rng.choice(["a", "x", "y"]), rng.randint(5, 9), rng.randint(1, 4)
    return _short(f"Simplify {b}^{p} ÷ {b}^{q}.", f"{b}^{p - q}", "Subtract the indices when dividing.", "Understanding")


def meas_perim(rng, lv, d):
    l, w = rng.randint(4, 12 * _sc(d)), rng.randint(3, 9 * _sc(d))
    return _mcq(rng, f"{2 * (l + w)} m", [f"{l * w} m", f"{l + w} m"],
                f"A school garden is {l} m long and {w} m wide. How much fencing goes round it?",
                f"Perimeter = 2 × ({l} + {w}) = {2 * (l + w)} m.", "Application")


def meas_area(rng, lv, d):
    if band(lv) == "jhs" and rng.random() < 0.5:
        b, h = 2 * rng.randint(2, 8), rng.randint(3, 12)
        return _short(f"Find the area of a triangle with base {b} cm and height {h} cm.", f"{b * h // 2} cm²",
                      f"½ × {b} × {h} = {b * h // 2} cm².", "Application")
    l, w = rng.randint(3, 15), rng.randint(2, 10)
    return _short(f"Tiles are laid on a floor {l} m by {w} m. What area is covered?", f"{l * w} m²", f"{l} × {w} = {l * w} m².",
                  "Application")


def meas_volume(rng, lv, d):
    l, w, h = rng.randint(2, 5), rng.randint(1, 4), rng.randint(1, 3)
    return _mcq(rng, f"{l * w * h * 1000:,} litres", [f"{l * w * h:,} litres", f"{(l + w + h) * 1000:,} litres"],
                f"A water tank is {l} m long, {w} m wide and {h} m high. How many litres can it hold? (1 m³ = 1000 litres)",
                f"Volume = {l * w * h} m³ = {l * w * h * 1000:,} litres.", "Application")


def meas_circle(rng, lv, d):
    r = 7 * rng.randint(1, 3)
    if rng.random() < 0.5:
        return _short(f"Find the circumference of a circle of radius {r} cm. (Take π = 22/7)", f"{2 * 22 * r // 7} cm",
                      f"C = 2πr = 2 × 22/7 × {r} = {2 * 22 * r // 7} cm.", "Application")
    return _short(f"Find the area of a circular table top of radius {r} cm. (Take π = 22/7)", f"{22 * r * r // 7} cm²",
                  f"A = πr² = 22/7 × {r}² = {22 * r * r // 7} cm².", "Application")


def meas_convert(rng, lv, d):
    cm = rng.randint(120, 950)
    return _short(f"A piece of cloth is {cm} cm long. What is its length in metres?", f"{cm / 100:g} m", "100 cm = 1 m.",
                  "Understanding")


def meas_capacity(rng, lv, d):
    cups, ml = rng.randint(3, 8), rng.choice([250, 500])
    return _short(f"A jug fills {cups} cups of {ml} ml each. How many litres does the jug hold?", f"{cups * ml / 1000:g} litres",
                  f"{cups} × {ml} ml = {cups * ml} ml = {cups * ml / 1000:g} litres.", "Application")


def ang_line(rng, lv, d):
    a = rng.randint(20, 160)
    return _mcq(rng, f"{180 - a}°", [f"{360 - a}°", f"{a}°"], f"Two angles lie on a straight line. One is {a}°. Find the other.",
                f"Angles on a straight line add up to 180°: 180 − {a} = {180 - a}°.", "Reasoning")


def ang_triangle(rng, lv, d):
    a, b = rng.randint(30, 80), rng.randint(30, 80)
    return _short(f"Two angles of a triangle are {a}° and {b}°. Find the third angle.", f"{180 - a - b}°",
                  "Angles in a triangle add up to 180°.", "Application")


def ang_polygon(rng, lv, d):
    n, name = rng.choice([(5, "pentagon"), (6, "hexagon"), (8, "octagon"), (10, "decagon")])
    return _mcq(rng, f"{(n - 2) * 180 // n}°", [f"{360 // n}°", f"{(n - 2) * 180}°"],
                f"Find the size of each interior angle of a regular {name}.",
                f"Sum = ({n} − 2) × 180 = {(n - 2) * 180}°. Each = {(n - 2) * 180} ÷ {n} = {(n - 2) * 180 // n}°.", "Application")


def ang_point(rng, lv, d):
    a, b = rng.randint(60, 150), rng.randint(40, 120)
    return _short(f"Three angles meet at a point. Two of them are {a}° and {b}°. Find the third.", f"{360 - a - b}°",
                  "Angles at a point add up to 360°.", "Application")


def shape_sides(rng, lv, d):
    shape, s = rng.choice([("triangle", 3), ("square", 4), ("pentagon", 5), ("hexagon", 6)])
    return _mcq(rng, s, [s + 1, s - 1], f"How many sides does a {shape} have?", f"A {shape} has {s} sides.", "Recall")


def shape_3d(rng, lv, d):
    obj, faces = rng.choice([("a cube (like a dice)", 6), ("a cuboid (like a matchbox)", 6), ("a triangular prism", 5),
                             ("a square-based pyramid", 5)])
    return _short(f"How many faces does {obj} have?", faces, "Count every flat surface.", "Understanding")


def shape_corners(rng, lv, d):
    shape, c = rng.choice([("rectangle", 4), ("triangle", 3), ("hexagon", 6)])
    obj = {"rectangle": "a door", "triangle": "a road sign", "hexagon": "a honeycomb cell"}[shape]
    return _short(f"{obj.capitalize()} has the shape of a {shape}. How many corners does it have?", c,
                  f"A {shape} has {c} corners.", "Understanding")


def sets_union(rng, lv, d):
    a, b, both = rng.randint(10, 25), rng.randint(10, 25), rng.randint(3, 9)
    return _mcq(rng, a + b - both, [a + b, a + b + both, a - both],
                f"In a JHS class, {a} learners like football, {b} like ampe and {both} like both. Each learner likes at least "
                "one. How many learners are in the class?", f"n(F ∪ A) = {a} + {b} − {both} = {a + b - both}.", "Reasoning")


def sets_list(rng, lv, d):
    A = sorted(rng.sample(range(1, 13), 5))
    B = sorted(rng.sample(range(1, 13), 5))
    inter = sorted(set(A) & set(B))
    return _short(f"Given A = {{{', '.join(map(str, A))}}} and B = {{{', '.join(map(str, B))}}}, list A ∩ B.",
                  "{" + ", ".join(map(str, inter)) + "}" if inter else "{ } (empty set)",
                  "A ∩ B contains the elements found in both sets.", "Understanding")


def lcm_hcf(rng, lv, d):
    a, b = rng.choice([(12, 18), (8, 12), (15, 20), (16, 24), (9, 12), (14, 21)])
    if rng.random() < 0.5:
        return _short(f"Find the LCM of {a} and {b}.", a * b // math.gcd(a, b), "Smallest number both divide into.", "Application")
    return _short(f"Find the HCF of {a} and {b}.", math.gcd(a, b), "Largest number that divides both.", "Application")


def stat_mean(rng, lv, d):
    v = [rng.randint(5, 20 * _sc(d)) for _ in range(5)]
    while sum(v) % 5:
        v[0] += 1
    x = _nm(rng, 1)[0]
    return _mcq(rng, sum(v) // 5, [sorted(v)[2] + 1, max(v) - min(v), sum(v)],
                f"{x} sold these numbers of sachets of water on five days: {', '.join(map(str, v))}. Find the mean.",
                f"{sum(v)} ÷ 5 = {sum(v) // 5}.", "Application")


def stat_median(rng, lv, d):
    v = rng.sample(range(30, 95), 7)
    return _short(f"These are test marks: {', '.join(map(str, v))}. Find the median mark.", sorted(v)[3],
                  f"Arrange in order: {', '.join(map(str, sorted(v)))}. The middle value is {sorted(v)[3]}.", "Application")


def stat_mode(rng, lv, d):
    m = rng.randint(2, 9)
    v = [m, m, m] + rng.sample([k for k in range(1, 12) if k != m], 4)
    rng.shuffle(v)
    return _short(f"The shoe sizes of some pupils are {', '.join(map(str, v))}. What is the mode?", m,
                  f"{m} appears most often.", "Recall")


def stat_prob(rng, lv, d):
    r, b, g = rng.randint(2, 6), rng.randint(2, 6), rng.randint(1, 5)
    col, k = rng.choice([("red", r), ("blue", b), ("green", g)])
    t = r + b + g
    gg = math.gcd(k, t)
    return _mcq(rng, f"{k // gg}/{t // gg}", [f"{k}/{t - k}", f"1/{t}", f"{t - k}/{t}"],
                f"A bag has {r} red, {b} blue and {g} green balls. One is picked at random. What is the probability it is {col}?",
                f"{k} {col} out of {t} balls = {k}/{t}" + (f" = {k // gg}/{t // gg}." if gg > 1 else "."), "Application")


def stat_pictograph(rng, lv, d):
    each, pics = rng.choice([2, 5, 10]), rng.randint(3, 8)
    return _short(f"In a pictograph, each picture of a mango stands for {each} mangoes. Class 3 has {pics} pictures. "
                  "How many mangoes did Class 3 collect?", pics * each, f"{pics} × {each} = {pics * each}.", "Understanding")


def stat_table(rng, lv, d):
    days = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    v = [rng.randint(20, 45) for _ in days]
    hi = days[v.index(max(v))]
    return _short("Number of learners who bought lunch at the canteen: " + ", ".join(f"{a} {b}" for a, b in zip(days, v))
                  + ". (a) On which day did most learners buy lunch? (b) How many bought lunch in the whole week?",
                  f"(a) {hi}  (b) {sum(v)}", "Read the largest value; add all five values.", "Application", "structured")


VARIANTS = [
    (r"addition|counting", [add_story, add_missing, add_market]),
    (r"subtraction", [sub_story, sub_compare, sub_missing]),
    (r"money", [money_change, money_enough, money_total, money_coins, money_pesewas]),
    (r"place value|whole numbers", [pv_digit, pv_expanded, pv_largest]),
    (r"multiplication|division", [mul_crates, div_benches, mul_missing]),
    (r"fraction|rational", [frac_of, frac_compare, frac_error, frac_add]),
    (r"decimal", [dec_add, dec_order, dec_times]),
    (r"percent|\bratio\b|proportion", [pct_discount, ratio_share, pct_class, pct_profit, simple_interest]),
    (r"pattern|algebraic expression", [alg_sub, alg_like, alg_expand, seq_next]),
    (r"linear equation|inequalit", [eq_solve, eq_think, ineq, eq_error]),
    (r"indices|standard form", [ind_product, ind_stdform, ind_solve, ind_divide]),
    (r"perimeter|area|mensuration|volume", [meas_perim, meas_area, meas_volume, meas_circle]),
    (r"length|capacity", [meas_convert, meas_capacity, meas_perim]),
    (r"angle|construction", [ang_line, ang_triangle, ang_polygon, ang_point]),
    (r"shape", [shape_sides, shape_3d, shape_corners]),
    (r"number sets|sets", [sets_union, sets_list, lcm_hcf]),
    (r"statistic|probability|data|pictograph|graph|mean",
     [stat_mean, stat_median, stat_mode, stat_prob, stat_pictograph, stat_table]),
]


def _variants_for(topic: str, level: int) -> list:
    t = topic.lower()
    out = []
    for pat, fns in VARIANTS:
        if re.search(pat, t):
            out.extend(f for f in fns if f not in out)
    if band(level) == "lower":
        out = [f for f in out if f not in (simple_interest, eq_error, ind_stdform, sets_union, stat_prob, meas_circle,
                                           stat_median)] or out
    return out or [stat_mean, frac_of, pct_discount, add_story]
