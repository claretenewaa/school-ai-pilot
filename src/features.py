"""Turn a school's raw imported records into the tables the AI uses.

Produces the same shapes as the synthetic dataset:
  students, sst (one row per student-subject-term, with model features) and qr (per-question marks).
Rows for the current term have no exam yet; those are exactly the rows the early-warning model scores.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import school_store as store
from curriculum import LABEL_TO_LEVEL, TOPICS, band, jhs_grade, primary_proficiency


def _term_of(dates: pd.Series, term_dates: dict) -> pd.Series:
    d = pd.to_datetime(dates, errors="coerce")
    t = pd.Series(np.nan, index=dates.index)
    for k, (s, e) in term_dates.items():
        t[(d >= pd.Timestamp(s)) & (d <= pd.Timestamp(e))] = int(k)
    return t


def _max_run(flags) -> int:
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


def attendance_summary(att: pd.DataFrame, assess: pd.DataFrame, term_dates: dict) -> pd.DataFrame:
    """Per student and term: day-level attendance measures."""
    if att.empty:
        return pd.DataFrame()
    att = att.copy()
    att["term"] = _term_of(att.date, term_dates)
    att = att.dropna(subset=["term"])
    att["term"] = att.term.astype(int)
    day = att[att.subject.fillna("") == ""]
    lesson = att[att.subject.fillna("") != ""]
    if day.empty and not lesson.empty:  # derive whole-day status from lesson records
        g = lesson.groupby(["date", "student_id", "term"]).status
        day = pd.DataFrame({"status": g.agg(lambda s: "A" if (s == "A").all() else
                                            ("L" if (s == "L").any() else ("E" if (s == "E").any() else "P")))}).reset_index()
        day["reason"] = ""
    test_dates = pd.to_datetime(assess.loc[assess.assessment_type.isin(["class_test", "mid_term", "exam"]), "date"],
                                errors="coerce").dropna().unique() if not assess.empty else []
    rows = []
    for (sid, term), g in day.groupby(["student_id", "term"]):
        g = g.sort_values("date")
        absent = (g.status == "A").to_numpy()
        dates = pd.to_datetime(g.date)
        before = 0
        for td in test_dates:
            prior = dates[dates < td].tail(2)
            before += int(absent[dates.isin(prior).to_numpy()].sum())
        reasons = g.loc[g.status == "A", "reason"].replace("", np.nan).dropna()
        rows.append(dict(student_id=sid, term=term, school_days_in_term=len(g), days_absent=int(absent.sum()),
                         days_present=int((~absent).sum()), attendance_rate=round(1 - absent.mean(), 3),
                         late_arrivals=int((g.status == "L").sum()), times_left_early=int((g.status == "E").sum()),
                         max_consecutive_days_absent=_max_run(absent), days_absent_before_tests=before,
                         main_absence_reason=reasons.mode().iat[0] if len(reasons) else
                         ("No reason given" if absent.any() else "None")))
    out = pd.DataFrame(rows)
    if not lesson.empty:
        les = lesson.assign(missed=lesson.status.eq("A")).groupby(["student_id", "term", "subject"]).agg(
            lessons_scheduled=("missed", "size"), lessons_missed=("missed", "sum")).reset_index()
        les["lesson_attendance_rate"] = 1 - les.lessons_missed / les.lessons_scheduled
        out = out.merge(les.drop(columns="lessons_missed"), on=["student_id", "term"], how="left")
    return out


def build(profile: dict | None = None) -> dict:
    profile = profile or store.get_profile()
    students = store.read("students")
    if students.empty:
        return {}
    students = students.assign(class_level=students.class_label.map(LABEL_TO_LEVEL))
    assess = store.read("assessments")
    att = store.read("attendance")
    eng = store.read("engagement")
    qm = store.read("question_marks")
    disc = store.read("discipline")
    intv = store.read("interventions")
    td = profile["term_dates"]
    ca_w = float(profile.get("ca_weight", 30))

    if assess.empty:
        return dict(students=students, sst=pd.DataFrame(), qr=pd.DataFrame())
    a = assess.copy()
    for c in ("score", "max_score", "time_taken_min", "time_allowed_min"):
        a[c] = pd.to_numeric(a[c], errors="coerce")
    a["term"] = a.term.astype(int)
    a["pct"] = 100 * a.score / a.max_score
    a["submitted_b"] = a.submitted.fillna("Yes").eq("Yes")

    def comp(types, col="pct"):
        sub = a[a.assessment_type.isin(types) & a.submitted_b]
        return sub.groupby(["student_id", "term", "subject"])[col].mean()

    key = ["student_id", "term", "subject"]
    base = a[a.assessment_type != "previous_exam"][key].drop_duplicates().set_index(key)
    hw = a[a.assessment_type == "homework"].groupby(key).agg(homework_given=("submitted_b", "size"),
                                                             homework_submitted=("submitted_b", "sum"))
    cw = a[a.assessment_type == "classwork"].groupby(key).submitted_b.mean().rename("classwork_rate")
    tests = a[a.assessment_type.isin(["class_test", "mid_term"])].groupby(key).agg(
        avg_test_time_min=("time_taken_min", "mean"), test_time_allowed_min=("time_allowed_min", "mean"))
    exam = a[a.assessment_type == "exam"].groupby(key).agg(exam_score_100=("pct", "mean"),
                                                           teacher_comment=("teacher_comment", "last"))
    df = base.join([comp(["class_test", "mid_term"]).rename("class_test_avg"), comp(["quiz"]).rename("quiz_avg"),
                    comp(["homework"]).rename("homework_avg"), hw, cw, tests, exam]).reset_index()
    df["assignment_submission_rate"] = (df.homework_submitted / df.homework_given).round(3)

    # previous exam: same subject, previous term; or an imported 'previous_exam' for term 1
    prev = exam.exam_score_100.reset_index()
    prev["term"] += 1
    df = df.merge(prev.rename(columns={"exam_score_100": "prev_exam_score"}), on=key, how="left")
    py = a[a.assessment_type == "previous_exam"].groupby(key).pct.mean().rename("prev_imported").reset_index()
    df = df.merge(py, on=key, how="left")
    df["prev_exam_score"] = df.prev_exam_score.fillna(df.prev_imported)

    if not eng.empty:
        e = eng.copy()
        e["term"] = e.term.astype(int)
        for c in e.columns.difference(["student_id", "subject", "term"]):
            e[c] = pd.to_numeric(e[c], errors="coerce")
        df = df.merge(e, on=key, how="left")
    for c in ["participation_rating", "answers_per_week", "group_work_rating", "help_seeking_rating",
              "time_on_task_pct", "observation_rating", "peer_learning_rating", "improvement_after_feedback",
              "classwork_completion_pct"]:
        if c not in df:
            df[c] = np.nan
    df["classwork_completion_rate"] = (df.classwork_completion_pct / 100).fillna(df.classwork_rate)

    # class score out of 30 (or the school's CA weight)
    parts = (0.40 * df.class_test_avg.fillna(df.quiz_avg) + 0.20 * df.quiz_avg.fillna(df.class_test_avg)
             + 0.25 * df.homework_avg.fillna(0) * df.assignment_submission_rate.fillna(1)
             + 0.15 * 100 * df.classwork_completion_rate.fillna(0.8))
    df["ca_score_30"] = (30 * parts / 100).clip(0, 30).round(1)
    df["total_score_100"] = (df.ca_score_30 * ca_w / 30 + df.exam_score_100 * (100 - ca_w) / 100).round(1)

    att_s = attendance_summary(att, a, td)
    if not att_s.empty:
        if "subject" in att_s:
            day_cols = [c for c in att_s.columns if c not in ("subject", "lessons_scheduled", "lesson_attendance_rate")]
            df = df.merge(att_s[day_cols].drop_duplicates(["student_id", "term"]), on=["student_id", "term"], how="left")
            df = df.merge(att_s.dropna(subset=["subject"])[["student_id", "term", "subject", "lessons_scheduled",
                                                            "lesson_attendance_rate"]], on=key, how="left")
        else:
            df = df.merge(att_s, on=["student_id", "term"], how="left")
    for c in ["attendance_rate", "days_absent", "late_arrivals", "times_left_early", "max_consecutive_days_absent",
              "days_absent_before_tests", "main_absence_reason", "lesson_attendance_rate", "school_days_in_term"]:
        if c not in df:
            df[c] = np.nan
    df["lesson_attendance_rate"] = df.lesson_attendance_rate.fillna(df.attendance_rate)

    if not disc.empty:
        disc = disc.assign(term=_term_of(disc.date, td))
        df = df.merge(disc.groupby(["student_id", "term"]).size().rename("discipline_incidents").reset_index(),
                      on=["student_id", "term"], how="left")
    df["discipline_incidents"] = df.get("discipline_incidents", pd.Series(0, index=df.index)).fillna(0)
    if not intv.empty:
        intv = intv.assign(term=_term_of(intv.date, td))
        flag = intv.dropna(subset=["term"]).groupby(["student_id", "term"]).size().rename("n_int").reset_index()
        df = df.merge(flag, on=["student_id", "term"], how="left")
        df["intervention_this_term"] = df.n_int.fillna(0) > 0
    else:
        df["intervention_this_term"] = False

    df = df.merge(students[["student_id", "class_level", "class_label"]], on="student_id", how="left")
    df = df.dropna(subset=["class_level"])
    df["class_level"] = df.class_level.astype(int)
    df["academic_year"] = profile.get("academic_year", "")
    df["school_id"] = "SCHOOL"

    # question marks -> per-question table, and the main error type per student-subject-term
    qr = pd.DataFrame()
    if not qm.empty:
        q = qm.copy()
        q["term"] = q.term.astype(int)
        for c in ("max_marks", "marks_awarded", "question_no"):
            q[c] = pd.to_numeric(q[c], errors="coerce")
        q = q.merge(students[["student_id", "class_level", "class_label"]], on="student_id", how="left")
        strand_of = {t[1].lower(): t[0] for subj in TOPICS.values() for b in subj.values() for t in b}
        qr = pd.DataFrame(dict(
            student_id=q.student_id, term=q.term, subject=q.subject, class_level=q.class_level,
            question_id=q.class_label + "|" + q.subject + "|T" + q.term.astype(str) + "|" + q.assessment_name + "|Q"
            + q.question_no.astype(int).astype(str),
            question_no=q.question_no, strand=q.topic.str.lower().map(strand_of).fillna(""), topic=q.topic,
            difficulty=q.difficulty.replace("", "Medium").fillna("Medium"), cognitive_level="", max_marks=q.max_marks,
            marks_awarded=q.marks_awarded, fully_correct=(q.marks_awarded >= q.max_marks).astype(int),
            error_type=q.error_type.fillna(""), time_spent_sec=np.nan))
        main_err = qr[qr.error_type != ""].groupby(key).error_type.agg(lambda s: s.value_counts().index[0])
        df = df.merge(main_err.rename("common_error_type").reset_index(), on=key, how="left")
        ex = qr.groupby(key).agg(exam_questions_attempted=("error_type", lambda s: int((s != "No attempt").sum())),
                                 exam_questions_correct=("fully_correct", "sum"),
                                 exam_questions_total=("question_no", "count")).reset_index()
        df = df.merge(ex, on=key, how="left")
    if "common_error_type" not in df:
        df["common_error_type"] = np.nan
    df["common_error_type"] = df.common_error_type.fillna("Unknown")

    # outcomes (only where the exam is in)
    has_exam = df.exam_score_100.notna()
    df["at_risk"] = np.where(has_exam, ((df.exam_score_100 < 50) |
                                        ((df.prev_exam_score - df.exam_score_100) >= 15)).astype(float), np.nan)

    def grade(r):
        if pd.isna(r.total_score_100):
            return pd.Series({"grade": "", "grade_description": ""})
        if band(int(r.class_level)) == "jhs":
            g, dsc = jhs_grade(r.total_score_100)
            return pd.Series({"grade": str(g), "grade_description": dsc})
        g, dsc = primary_proficiency(r.total_score_100)
        return pd.Series({"grade": g, "grade_description": dsc})

    df[["grade", "grade_description"]] = df.apply(grade, axis=1)
    df["teacher_comment"] = df.teacher_comment.fillna("")
    df["fees_paid_pct"] = np.nan
    return dict(students=students, sst=df, qr=qr)


def health(profile: dict | None = None) -> pd.DataFrame:
    """The data champion's weekly check: what is in, what is missing."""
    profile = profile or store.get_profile()
    st = store.read("students")
    rows = []
    n = len(st)
    rows.append(("Student register", f"{n} learners" if n else "Empty: import the register first", bool(n)))
    att = store.read("attendance")
    if att.empty:
        rows.append(("Attendance", "No attendance yet", False))
    else:
        last = pd.to_datetime(att.date).max()
        cover = att[pd.to_datetime(att.date) == last].student_id.nunique()
        rows.append(("Attendance", f"{len(att):,} marks; latest day {last:%a %d %b %Y} covers {cover} of {n} learners",
                     cover >= 0.9 * n if n else False))
    a = store.read("assessments")
    if a.empty:
        rows.append(("Scores", "No scores yet", False))
    else:
        per = a.groupby("assessment_type").size().to_dict()
        rows.append(("Scores", ", ".join(f"{k}: {v:,}" for k, v in per.items()), True))
        orphans = set(a.student_id) - set(st.student_id)
        if orphans:
            rows.append(("Student IDs", f"{len(orphans)} IDs in scores are not in the register", False))
    q = store.read("question_marks")
    rows.append(("Marks per question", f"{len(q):,} question marks" if len(q) else
                 "None yet: needed for learning profiles and gap analysis", bool(len(q))))
    e = store.read("engagement")
    rows.append(("Engagement ratings", f"{len(e):,} ratings" if len(e) else "None yet (optional, improves warnings)",
                 bool(len(e))))
    qb = store.read("question_bank")
    rows.append(("Question bank", f"{len(qb):,} questions" if len(qb) else "No school questions yet (optional)",
                 bool(len(qb))))
    return pd.DataFrame(rows, columns=["Data", "Status", "OK"])
