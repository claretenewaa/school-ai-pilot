"""Write sample import files for ONE school (Madina Bright Future Academy, SCH05) in the formats a school
keeps: student register, daily attendance, the class register layout, scores, a broadsheet, per-question marks,
engagement ratings, discipline and interventions.

Terms 1 and 2 are complete. Term 3 is "in progress": attendance and class scores, but no exam yet,
so the early-warning model has something real to predict.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from generate_data import DAYS, EXAM_START, TERMS  # school calendar 2025/2026

ROOT = Path(__file__).resolve().parent.parent
SRC, OUT = ROOT / "data", ROOT / "sample_imports"
SCHOOL = "SCH05"
rng = np.random.default_rng(5)


def new_id(s: str) -> str:
    return s.replace(f"{SCHOOL}-", "BFA-")


def spread(mean, n, lo, hi, sd):
    """n individual marks whose average is close to `mean`."""
    x = np.clip(rng.normal(mean, sd, n), lo, hi)
    return np.round(x + (mean - x.mean()), 0).clip(lo, hi)


def main():
    OUT.mkdir(exist_ok=True)
    st = pd.read_csv(SRC / "students.csv")
    st = st[st.school_id == SCHOOL].copy()
    sst = pd.read_csv(SRC / "student_subject_term.csv", dtype={"grade": str})
    sst = sst[sst.school_id == SCHOOL].copy()
    qr = pd.read_csv(SRC / "question_responses.csv")
    qr = qr[qr.student_id.isin(st.student_id)]
    ab = pd.read_csv(SRC / "absence_events.csv")
    ab = ab[ab.student_id.isin(st.student_id)]
    iv = pd.read_csv(SRC / "interventions.csv")
    iv = iv[iv.student_id.isin(st.student_id)]

    # 1. student register (as a school would type it: DOB day-first, M/F)
    reg = pd.DataFrame({
        "Student ID": st.student_id.map(new_id), "First Name": st.first_name, "Surname": st.surname,
        "Sex": st.gender.str[0], "Class": st.class_label,
        "DOB": pd.to_datetime(st.date_of_birth).dt.strftime("%d/%m/%Y"), "Home language": st.home_language,
        "Parent occupation": st.guardian_occupation, "Special needs": st.special_needs.replace("None", ""),
        "Status": st.status.str.replace(r" \(.*\)", "", regex=True)})
    reg.to_excel(OUT / "1_student_register.xlsx", index=False)

    # 2. daily attendance (all classes, all terms) rebuilt from absence spells + lateness counts
    term_rows = sst.drop_duplicates(["student_id", "term"])[["student_id", "term", "late_arrivals", "times_left_early"]]
    absent = {}
    for _, r in ab.iterrows():
        for d in pd.bdate_range(r.start_date, r.end_date):
            absent[(r.student_id, d.date())] = r.reason
    rows = []
    for _, r in term_rows.iterrows():
        days = DAYS[int(r.term)]
        present_days = [d for d in days if (r.student_id, d) not in absent]
        late = set(rng.choice(len(present_days), min(int(r.late_arrivals), len(present_days)), replace=False))
        early = set(rng.choice(len(present_days), min(int(r.times_left_early), len(present_days)), replace=False)) - late
        pi = 0
        for d in days:
            if (r.student_id, d) in absent:
                rows.append((d.strftime("%d/%m/%Y"), new_id(r.student_id), "A", absent[(r.student_id, d)]))
            else:
                mark = "L" if pi in late else ("E" if pi in early else "P")
                rows.append((d.strftime("%d/%m/%Y"), new_id(r.student_id), mark, ""))
                pi += 1
    att = pd.DataFrame(rows, columns=["Date", "Student ID", "Status", "Reason"])
    jhs2_t3 = att[att["Student ID"].isin(reg.loc[reg.Class == "JHS 2", "Student ID"]) &
                  pd.to_datetime(att.Date, dayfirst=True).between(*[pd.Timestamp(x) for x in TERMS[3]])]
    att.drop(jhs2_t3.index).to_csv(OUT / "2_attendance_daily_2025_26.csv", index=False)

    # 3. the same JHS 2 Term 3 attendance in the class register layout (one column per day)
    names = reg.set_index("Student ID")
    wide = jhs2_t3.pivot(index="Student ID", columns="Date", values="Status")
    wide = wide[sorted(wide.columns, key=lambda s: pd.to_datetime(s, dayfirst=True))]
    wide.insert(0, "Name", names.loc[wide.index, "First Name"] + " " + names.loc[wide.index, "Surname"])
    wide.reset_index().to_excel(OUT / "3_attendance_register_JHS2_term3.xlsx", index=False)

    # 4. scores: class tests, quizzes, homework, previous exam; exams for terms 1-2 (not JHS 2 term 2 - see 5)
    srows = []
    for _, r in sst.iterrows():
        sid, t, subj = new_id(r.student_id), int(r.term), r.subject
        start = DAYS[t][0]
        for k, m in enumerate(spread(r.class_test_avg * 0.2, 3, 0, 20, 2.2), 1):
            srows.append((sid, t, subj, "class_test", f"Test {k}", DAYS[t][min(12 + 15 * (k - 1), len(DAYS[t]) - 1)],
                          m, 20, "Yes", round(r.avg_test_time_min + rng.normal(0, 2), 0), r.test_time_allowed_min, ""))
        if not np.isnan(r.quiz_avg):
            for k, m in enumerate(spread(r.quiz_avg * 0.1, 4, 0, 10, 1.4), 1):
                srows.append((sid, t, subj, "quiz", f"Quiz {k}", DAYS[t][8 + 10 * (k - 1)], m, 10, "Yes", "", "", ""))
        done = rng.permutation([True] * int(r.homework_submitted) + [False] * int(r.homework_given - r.homework_submitted))
        hw_marks = spread(r.homework_avg * 0.1 if not np.isnan(r.homework_avg) else 6, int(r.homework_submitted) or 1, 0, 10, 1.5)
        j = 0
        for k, ok in enumerate(done, 1):
            day = DAYS[t][min(5 * k, len(DAYS[t]) - 1)]
            if ok:
                srows.append((sid, t, subj, "homework", f"Homework {k}", day, hw_marks[j], 10, "Yes", "", "", ""))
                j += 1
            else:
                srows.append((sid, t, subj, "homework", f"Homework {k}", day, "", 10, "No", "", "", ""))
        if t == 1 and not np.isnan(r.prev_exam_score):
            srows.append((sid, 1, subj, "previous_exam", "Last year Term 3 exam", "", r.prev_exam_score, 100, "Yes",
                          "", "", ""))
        if t in (1, 2) and not (t == 2 and r.class_label == "JHS 2"):
            srows.append((sid, t, subj, "exam", "End of term", EXAM_START[t], r.exam_score_100, 100, "Yes", "", "",
                          r.teacher_comment))
    scores = pd.DataFrame(srows, columns=["student_id", "term", "subject", "assessment_type", "assessment_name",
                                          "date", "score", "max_score", "submitted", "time_taken_min",
                                          "time_allowed_min", "teacher_comment"])
    scores["date"] = pd.to_datetime(scores.date).dt.strftime("%d/%m/%Y").fillna("")
    scores.to_csv(OUT / "4_scores_2025_26.csv", index=False)

    # 5. broadsheet: JHS 2 end-of-term-2 exam, one column per subject (as teachers keep it)
    j2 = sst[(sst.class_label == "JHS 2") & (sst.term == 2)]
    bs = j2.pivot(index="student_id", columns="subject", values="exam_score_100")
    short = {"English Language": "English", "Religious & Moral Education": "RME",
             "Creative Arts & Design": "Creative Arts & Design", "Career Technology": "Career Tech"}
    bs = bs.rename(columns=short)
    bs.index = bs.index.map(new_id)
    bs.insert(0, "Name", names.loc[bs.index, "First Name"] + " " + names.loc[bs.index, "Surname"])
    bs.reset_index().rename(columns={"student_id": "Student ID"}).to_excel(
        OUT / "5_broadsheet_JHS2_term2_exam.xlsx", index=False)

    # 6. marks per question (end-of-term papers, terms 1-2)
    q = qr[qr.term <= 2]
    pd.DataFrame({"student_id": q.student_id.map(new_id), "term": q.term, "subject": q.subject,
                  "assessment_name": "End of term", "question_no": q.question_no, "topic": q.topic,
                  "max_marks": q.max_marks, "marks_awarded": q.marks_awarded, "error_type": q.error_type.fillna(""),
                  "difficulty": q.difficulty}).to_csv(OUT / "6_question_marks_2025_26.csv", index=False)

    # 7. engagement ratings once per term per subject
    pd.DataFrame({"student_id": sst.student_id.map(new_id), "term": sst.term, "subject": sst.subject,
                  "participation_rating": sst.participation_rating, "answers_per_week": sst.answers_per_week,
                  "group_work_rating": sst.group_work_rating, "help_seeking_rating": sst.help_seeking_rating,
                  "time_on_task_pct": sst.time_on_task_pct, "observation_rating": sst.observation_rating,
                  "peer_learning_rating": sst.peer_learning_rating,
                  "classwork_completion_pct": (100 * sst.classwork_completion_rate).round(0),
                  "improvement_after_feedback": sst.improvement_after_feedback}).to_excel(
        OUT / "7_engagement_ratings_2025_26.xlsx", index=False)

    # 8. discipline incidents and 9. interventions
    inc = ["Fighting", "Lateness to class", "Bullying", "Using phone in class", "Disrespect to teacher",
           "Not in proper uniform", "Copying in a test", "Leaving school without permission"]
    drows = []
    for _, r in sst.drop_duplicates(["student_id", "term"]).iterrows():
        for _ in range(int(r.discipline_incidents)):
            d = DAYS[int(r.term)][rng.integers(0, len(DAYS[int(r.term)]))]
            drows.append((d.strftime("%d/%m/%Y"), new_id(r.student_id), rng.choice(inc),
                          rng.choice(["Warning", "Parent called", "Detention", "Letter to parent"])))
    pd.DataFrame(drows, columns=["date", "student_id", "incident", "action_taken"]).to_csv(
        OUT / "8_discipline_2025_26.csv", index=False)
    pd.DataFrame({"date": pd.to_datetime(iv.intervention_date).dt.strftime("%d/%m/%Y"),
                  "student_id": iv.student_id.map(new_id), "intervention_type": iv.intervention_type,
                  "notes": iv.reason_flagged}).to_csv(OUT / "9_interventions_2025_26.csv", index=False)
    for f in sorted(OUT.iterdir()):
        print(f"{f.name}: {f.stat().st_size / 1024:,.0f} KB")


if __name__ == "__main__":
    main()
