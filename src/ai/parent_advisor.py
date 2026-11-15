"""Parent advisor: turns a learner's profile into a plain-language progress summary for parents.

Output is always a DRAFT that a teacher must approve before it is sent.
"""
from __future__ import annotations

import json

from . import llm

LANGUAGES = ["English", "Twi", "Ga", "Ewe", "Hausa", "French"]

SYSTEM = (
    "You help teachers at a basic school in Accra, Ghana write short progress updates for parents. "
    "Write warmly and plainly, as a caring teacher would speak to a parent, with no jargon, no percentages "
    "beyond one or two key numbers, and no labels like 'at risk'. Structure: (1) one or two sentences on how "
    "the child is doing overall, (2) what they are good at, (3) the two most important things to work on, "
    "explained simply, (4) three practical, low-cost things the parent can do at home this week, suited to "
    "Ghanaian family life, (5) attendance, only if it is a concern. Keep it under 220 words. Use only facts "
    "from the profile; do not invent anything. Do not mention AI."
)


def summarise(profile: dict, language: str = "English", use_llm: bool = True) -> tuple[str, str]:
    """Returns (summary, source)."""
    if use_llm and llm.credentials_available():
        prompt = (f"Write the update in {language}." + (
            " Use natural, everyday language that parents in Accra speak; keep subject names in English if "
            "that is more natural." if language != "English" else "") +
            f"\n\nLearner profile (JSON):\n{json.dumps(profile, indent=1, default=str)}")
        return llm.ask(SYSTEM, prompt, effort="medium", max_tokens=6000), "claude"
    return template(profile), "offline"


def template(p: dict) -> str:
    subj = sorted(p["subjects"], key=lambda s: -s["latest_total"])
    if not subj:
        return f"No exam results are recorded yet for {p['first_name']}. Import end-of-term results first."
    best = ", ".join(s["subject"] for s in subj[:2])
    avg = sum(s["latest_total"] for s in subj) / max(1, len(subj))
    overall = ("doing very well" if avg >= 70 else "making steady progress" if avg >= 55
               else "finding some subjects difficult and needs extra support")
    lines = [f"Dear Parent,", "",
             f"{p['first_name']} ({p['class_label']}) is {overall} this term (Term {p['term']}). "
             f"{p['first_name']}'s strongest subjects are {best}."]
    if p["strong_topics"]:
        lines.append("Topics done well: " + ", ".join(t["topic"] for t in p["strong_topics"][:3]) + ".")
    if p["gaps"]:
        lines += ["", "Things to work on:"]
        for g in p["gaps"][:2]:
            lines.append(f"- {g['subject']}: {g['topic']}. Most marks are lost through "
                         f"{g['main_error'].lower()}.")
    lines += ["", "How you can help at home this week:",
              "- Check and sign the homework book every evening.",
              "- Spend 15 minutes a day on the topics above; ask your child to explain one thing they learnt.",
              "- Use everyday moments (counting change, reading labels, telling stories) to practise."]
    if p["attendance_rate"] < 0.9:
        lines += ["", f"Attendance: {p['first_name']} missed {p['days_absent']} school days this term. "
                      "Regular attendance will make a big difference."]
    lines += ["", "Thank you for your support.", "Class Teacher"]
    return "\n".join(lines)
