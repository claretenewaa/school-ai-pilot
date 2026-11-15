"""Learning profile and gap analysis.

Topic mastery comes from per-question marks, smoothed towards the class average
(a beta-binomial estimate) and weighted towards recent terms. Wrong answers are traced
to topics and error types, and each weak topic gets a practice suggestion.
"""
from __future__ import annotations

import pandas as pd

TERM_WEIGHT = {0: 1.0, 1: 0.7, 2: 0.5}  # weight by how many terms ago
PRIOR_STRENGTH = 6.0                     # marks' worth of pull towards the class average

STRONG, WEAK = 0.75, 0.50

ERROR_ACTIONS = {
    "Conceptual misunderstanding": "Re-teach the idea with concrete, local examples (counters, market prices, "
                                   "everyday objects), then give 5 short practice questions.",
    "Procedural/calculation error": "Work through 2-3 examples step by step, then practise the steps with the "
                                    "learner checking each line.",
    "Careless slip": "Teach a checking routine: re-read the question, estimate first, check the answer. "
                     "Practise without rushing.",
    "Misread question": "Practise underlining key words in the question and reading questions aloud before answering.",
    "Incomplete answer": "Show what a full answer looks like and use an answer checklist (all parts answered, "
                         "working shown).",
    "No attempt": "Build confidence with easier starter questions on the topic and encourage attempting every question.",
    "Language/vocabulary gap": "Pre-teach the key words for the topic (word wall, flash cards) and do short "
                               "reading practice with them.",
}


def topic_mastery(qr: pd.DataFrame, student_id: str) -> pd.DataFrame:
    """Mastery per subject and topic for one student, compared with classmates."""
    mine = qr[qr.student_id == student_id]
    if mine.empty:
        return pd.DataFrame()
    latest = mine.term.max()
    # class average per question paper (same papers share question_id prefixes)
    class_qr = qr[qr.question_id.isin(mine.question_id.unique())]
    class_avg = (class_qr.groupby(["subject", "topic"]).agg(m=("marks_awarded", "sum"), x=("max_marks", "sum")))
    class_avg = (class_avg.m / class_avg.x).rename("class_average")

    mine = mine.assign(w=(latest - mine.term).map(TERM_WEIGHT).fillna(0.4))
    g = mine.assign(wm=mine.marks_awarded * mine.w, wx=mine.max_marks * mine.w).groupby(["subject", "topic"]).agg(
        strand=("strand", "first"), term_taught=("term", "max"), wm=("wm", "sum"), wx=("wx", "sum"),
        questions=("question_id", "count"), raw_correct=("fully_correct", "mean"))
    g = g.join(class_avg)
    g["mastery"] = (g.wm + PRIOR_STRENGTH * g.class_average) / (g.wx + PRIOR_STRENGTH)
    g["status"] = pd.cut(g.mastery, [-1, WEAK, STRONG, 2], labels=["Weak", "Developing", "Strong"])
    g["vs_class"] = g.mastery - g.class_average
    errs = mine[mine.error_type.fillna("") != ""].groupby(["subject", "topic"]).error_type.agg(
        lambda s: s.value_counts().index[0])
    g["main_error"] = errs
    return g.drop(columns=["wm", "wx"]).reset_index().sort_values(["subject", "mastery"])


def error_profile(qr: pd.DataFrame, student_id: str) -> pd.DataFrame:
    mine = qr[(qr.student_id == student_id) & (qr.error_type.fillna("") != "")]
    return mine.groupby(["subject", "error_type"]).size().rename("count").reset_index().sort_values(
        ["subject", "count"], ascending=[True, False])


def gap_analysis(mastery: pd.DataFrame) -> list[dict]:
    """Weak topics, why the learner is losing marks, and what to practise."""
    if mastery.empty:
        return []
    gaps = mastery[mastery.status == "Weak"].copy()
    if gaps.empty:  # nothing weak: point to the lowest developing topics
        gaps = mastery[mastery.status == "Developing"].nsmallest(2, "mastery")
    out = []
    for _, r in gaps.sort_values("mastery").iterrows():
        err = r.main_error if isinstance(r.main_error, str) else "Conceptual misunderstanding"
        out.append(dict(subject=r.subject, topic=r.topic, mastery=round(float(r.mastery), 2),
                        class_average=round(float(r.class_average), 2), main_error=err,
                        action=ERROR_ACTIONS[err]))
    return out


def subject_summary(sst: pd.DataFrame, student_id: str) -> pd.DataFrame:
    """Per-subject results across terms with trend, from the main table."""
    mine = sst[sst.student_id == student_id]
    if mine.empty:
        return pd.DataFrame()
    piv = mine.pivot_table(index="subject", columns="term", values="total_score_100")
    if piv.empty:
        return pd.DataFrame()
    piv.columns = [f"Term {c}" for c in piv.columns]
    done = mine[mine.total_score_100.notna()]
    latest = done[done.term == done.term.max()].set_index("subject")
    piv["Latest grade"] = latest.grade
    piv["Trend"] = (piv.iloc[:, -2] - piv.iloc[:, -3]).round(1) if piv.shape[1] >= 3 else float("nan")
    piv["Lesson attendance"] = mine[mine.term == mine.term.max()].set_index("subject").lesson_attendance_rate
    piv["Teacher comment"] = latest.teacher_comment
    return piv.reset_index()


def build_profile(students, sst, qr, student_id: str) -> dict:
    """Everything the parent advisor needs, as a plain dictionary."""
    s = students.set_index("student_id").loc[student_id]
    mine = sst[sst.student_id == student_id]
    lt = mine[mine.term == mine.term.max()]  # current term: attendance and engagement
    done = mine[mine.total_score_100.notna()]
    latest_t = int(done.term.max()) if len(done) else int(mine.term.max())  # latest term with results
    mastery = topic_mastery(qr, student_id)
    subj = subject_summary(sst, student_id)
    return dict(
        name=f"{s.first_name} {s.surname}", first_name=s.first_name, class_label=s.class_label,
        term=latest_t, academic_year=str(mine.academic_year.iat[0]),
        attendance_term=int(lt.term.iat[0]),
        attendance_rate=float(lt.attendance_rate.iat[0]), days_absent=int(lt.days_absent.fillna(0).iat[0]),
        late_arrivals=int(lt.late_arrivals.fillna(0).iat[0]), main_absence_reason=str(lt.main_absence_reason.iat[0]),
        subjects=[dict(subject=r.subject, latest_total=round(float(r[f"Term {latest_t}"]), 1),
                       grade=str(r["Latest grade"]), trend=None if pd.isna(r.Trend) else float(r.Trend),
                       teacher_comment=r["Teacher comment"]) for _, r in subj.iterrows()
                  if f"Term {latest_t}" in subj.columns and not pd.isna(r[f"Term {latest_t}"])],
        strong_topics=mastery[mastery.status == "Strong"].nlargest(4, "mastery")[["subject", "topic"]].to_dict("records")
        if not mastery.empty else [],
        gaps=gap_analysis(mastery)[:4],
        engagement=dict(participation=float(lt.participation_rating.mean()),
                        submission_rate=float(lt.assignment_submission_rate.mean()),
                        time_on_task=float(lt.time_on_task_pct.mean())),
    )
