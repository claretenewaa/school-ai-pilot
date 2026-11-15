"""Import school data from Excel/CSV files: templates, column matching, cleaning and validation.

Designed for a school's data collector: forgiving about column names, class names ("B7", "Basic 7", "JHS1"),
subject names ("Maths", "Int. Science", "RME"), attendance marks (P, A, L, E, ✓, x) and day-first dates,
and strict about the things that break the AI (unknown student IDs, scores above the maximum, bad dates).
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from curriculum import ERROR_TYPES, LEVEL_LABEL, SUBJECTS, band

# --------------------------------------------------------------------------- value normalisers
ALL_SUBJECTS = sorted({s for v in SUBJECTS.values() for s in v} | {"French"})
SUBJECT_ALIASES = {
    "math": "Mathematics", "maths": "Mathematics", "mathematics": "Mathematics", "numeracy": "Mathematics",
    "english": "English Language", "eng": "English Language", "english language": "English Language",
    "literacy": "English Language", "science": "Science", "int science": "Science", "integrated science": "Science",
    "general science": "Science", "social studies": "Social Studies", "social": "Social Studies",
    "soc studies": "Social Studies", "rme": "Religious & Moral Education",
    "religious and moral education": "Religious & Moral Education",
    "religious moral education": "Religious & Moral Education", "computing": "Computing", "ict": "Computing",
    "computer": "Computing", "owop": "Our World Our People", "our world our people": "Our World Our People",
    "ghanaian language": "Ghanaian Language", "gh lang": "Ghanaian Language", "twi": "Ghanaian Language",
    "ga": "Ghanaian Language", "ewe": "Ghanaian Language", "fante": "Ghanaian Language", "dagbani": "Ghanaian Language",
    "career technology": "Career Technology", "career tech": "Career Technology", "cartech": "Career Technology",
    "creative arts and design": "Creative Arts & Design", "cad": "Creative Arts & Design",
    "creative arts design": "Creative Arts & Design", "creative arts": "Creative Arts", "history": "History",
    "french": "French",
}


def norm_subject(v):
    if pd.isna(v) or str(v).strip() == "":
        return None
    k = re.sub(r"[^a-z ]", " ", str(v).lower().replace("&", " and ")).split()
    k = " ".join(k)
    return SUBJECT_ALIASES.get(k) or SUBJECT_ALIASES.get(k.replace(" and ", " ")) or (
        str(v).strip() if str(v).strip() in ALL_SUBJECTS else None)


def norm_class(v):
    if pd.isna(v):
        return None
    s = re.sub(r"[\s._-]+", "", str(v).lower())
    m = re.fullmatch(r"(class|basic|primary|b|p|jhs|form)(\d)", s)
    if not m:
        return None
    kind, n = m.group(1), int(m.group(2))
    if kind in ("jhs", "form"):
        n += 6
    if kind in ("primary", "p", "class") and n > 6:
        return None
    return LEVEL_LABEL.get(n)


def norm_gender(v):
    s = str(v).strip().lower()
    return {"m": "Male", "male": "Male", "boy": "Male", "f": "Female", "female": "Female", "girl": "Female"}.get(s)


ATT_CODES = {"p": "P", "present": "P", "✓": "P", "✔": "P", "/": "P", "1": "P", "y": "P",
             "a": "A", "absent": "A", "x": "A", "0": "A", "n": "A",
             "l": "L", "late": "L", "e": "E", "left early": "E", "le": "E"}


def norm_att(v):
    if pd.isna(v) or str(v).strip() == "":
        return None
    s = str(v).strip().lower()
    if s.endswith(".0"):
        s = s[:-2]
    return ATT_CODES.get(s, "?")


def norm_yesno(v, default=True):
    if pd.isna(v) or str(v).strip() == "":
        return default
    s = str(v).strip().lower()
    if s in ("y", "yes", "1", "true", "submitted", "✓"):
        return True
    if s in ("n", "no", "0", "false", "not submitted", "x"):
        return False
    return None


def norm_term(v):
    s = str(v).strip().lower()
    m = re.search(r"([123])", s)
    if m:
        return int(m.group(1))
    return {"first": 1, "second": 2, "third": 3}.get(s.replace("term", "").strip())


def norm_error(v):
    if pd.isna(v) or str(v).strip() == "":
        return ""
    s = str(v).lower()
    for pat, e in [("concept", 0), ("calc|procedur|method", 1), ("careless|slip", 2), ("misread|read", 3),
                   ("incomplete|partial", 4), ("no attempt|not attempted|blank|skipped", 5),
                   ("language|vocab|spelling", 6)]:
        if re.search(pat, s):
            return ERROR_TYPES[e]
    return None


ASSESSMENT_TYPES = ["class_test", "mid_term", "quiz", "homework", "classwork", "project", "exam", "previous_exam"]


def norm_atype(v):
    s = re.sub(r"[\s-]+", "_", str(v).strip().lower())
    aliases = {"test": "class_test", "class_test": "class_test", "classtest": "class_test", "midterm": "mid_term",
               "mid_term": "mid_term", "mid_term_test": "mid_term", "quiz": "quiz", "homework": "homework",
               "home_work": "homework", "assignment": "homework", "exercise": "classwork", "classwork": "classwork",
               "class_exercise": "classwork", "project": "project", "exam": "exam", "examination": "exam",
               "end_of_term": "exam", "end_of_term_exam": "exam", "previous_exam": "previous_exam",
               "last_year_exam": "previous_exam"}
    return aliases.get(s)


def to_date(v):
    if isinstance(v, (pd.Timestamp, np.datetime64)) or hasattr(v, "year"):
        return pd.Timestamp(v).normalize()
    if pd.isna(v) or str(v).strip() == "":
        return pd.NaT
    s = str(v).strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}.*", s):
        return pd.to_datetime(s[:10], errors="coerce")
    return pd.to_datetime(s, dayfirst=True, errors="coerce")


# --------------------------------------------------------------------------- templates
@dataclass
class Col:
    name: str
    required: bool
    kind: str            # str, date, int, float, term, subject, class, gender, att, yesno, rating, pct, error, atype, choice
    help: str
    example: object = ""
    aliases: tuple = ()
    lo: float | None = None
    hi: float | None = None


@dataclass
class Template:
    key: str
    table: str
    title: str
    purpose: str
    cols: list = field(default_factory=list)
    wide: bool = False


SID = Col("student_id", True, "str", "The learner's ID. Must match the student register exactly.", "BF-0012",
          ("id", "student_no", "student_number", "learner_id", "pupil_id", "admission_no", "admission_number",
           "index_no"))
TERM = Col("term", True, "term", "1, 2 or 3", 1, ("term_no",))
SUBJ = Col("subject", True, "subject", "Subject name, e.g. Mathematics, English, Science, Social Studies, RME",
           "Mathematics", ("subject_name", "course"))

TEMPLATES = {t.key: t for t in [
    Template("students", "students", "Student register", "Every learner in the school. Import this first.", [
        SID, Col("first_name", True, "str", "First name(s)", "Ama", ("firstname", "first", "given_name", "other_names")),
        Col("surname", True, "str", "Surname / family name", "Mensah", ("last_name", "lastname", "family_name")),
        Col("gender", True, "gender", "Male or Female (M/F also accepted)", "Female", ("sex",)),
        Col("class_label", True, "class", "Class 1 to Class 6, JHS 1 to JHS 3 (B1-B9 and Basic 1-9 also accepted)",
            "JHS 2", ("class", "class_name", "level", "grade_level", "form")),
        Col("date_of_birth", False, "date", "Day/Month/Year", "14/03/2013", ("dob", "birth_date", "birthday")),
        Col("home_language", False, "str", "Main language at home", "Twi", ("language",)),
        Col("guardian_name", False, "str", "Parent or guardian", "Mr Kwame Mensah", ("parent_name", "guardian")),
        Col("guardian_phone", False, "str", "Phone number", "024 000 0000", ("parent_phone", "phone", "contact")),
        Col("guardian_occupation", False, "str", "Occupation", "Trader", ("parent_occupation", "occupation")),
        Col("special_needs", False, "str", "Any special educational need (or blank)", ""),
        Col("status", False, "str", "Active, Transferred out, Withdrawn (blank = Active)", "Active"),
    ]),
    Template("attendance_register", "attendance", "Attendance register (one column per day)",
             "Copy of the class register: one row per learner, one column per school day. "
             "Mark P = present, A = absent, L = late, E = left early. Leave holidays blank.", [
                 SID, Col("name", False, "str", "Learner's name (for your reference; not imported)", "Ama Mensah")],
             wide=True),
    Template("attendance_daily", "attendance", "Attendance (one row per learner per day or lesson)",
             "Use this if your attendance is already in a list. Leave subject blank for whole-day attendance, "
             "or fill it for lesson-by-lesson attendance.", [
                 Col("date", True, "date", "Day/Month/Year", "08/09/2026", ("day",)), SID,
                 Col("status", True, "att", "P, A, L or E", "A", ("mark", "attendance", "present")),
                 Col("reason", False, "str", "Reason for absence or lateness, if known", "Malaria",
                     ("absence_reason", "remarks", "remark")),
                 Col("subject", False, "subject", "Only for lesson attendance; blank = whole day", "")]),
    Template("assessments", "assessments", "Scores: tests, quizzes, homework, classwork, exams",
             "One row per learner per assessment. Mark every test, quiz and assignment as it happens.", [
                 SID, TERM, SUBJ,
                 Col("assessment_type", True, "atype",
                     "class_test, mid_term, quiz, homework, classwork, project, exam, previous_exam", "class_test",
                     ("type", "assessment")),
                 Col("assessment_name", True, "str", "A short name, e.g. 'Test 1', 'Homework 3', 'End of term'",
                     "Test 1", ("name", "title")),
                 Col("date", False, "date", "Day/Month/Year", "25/09/2026"),
                 Col("score", False, "float", "Marks obtained (blank if not submitted)", 14,
                     ("mark", "marks", "marks_obtained", "score_obtained"), 0, None),
                 Col("max_score", True, "float", "Total marks possible", 20,
                     ("out_of", "total_marks", "max_marks", "maximum", "total"), 0.5, None),
                 Col("submitted", False, "yesno", "Yes/No (homework and classwork); blank = Yes", "Yes"),
                 Col("time_taken_min", False, "float", "Minutes the learner took (tests)", 24, (), 0, 300),
                 Col("time_allowed_min", False, "float", "Minutes allowed (tests)", 30, (), 1, 300),
                 Col("teacher_comment", False, "str", "Remark (end-of-term exam)", "Good effort",
                     ("remark", "remarks", "comment"))]),
    Template("broadsheet", "assessments", "Broadsheet (one column per subject)",
             "One row per learner, one column per subject, for a single assessment (e.g. the end-of-term exam). "
             "You choose the term, type and maximum score when you upload.", [
                 SID, Col("name", False, "str", "Learner's name (for your reference; not imported)", "Ama Mensah")],
             wide=True),
    Template("question_marks", "question_marks", "Marks per question",
             "For each test or exam, the marks for every question, tagged to a topic. Needed for learning profiles "
             "and gap analysis.", [
                 SID, TERM, SUBJ, Col("assessment_name", True, "str", "Same name as in Scores", "End of term"),
                 Col("question_no", True, "int", "1, 2, 3 ...", 1, ("question", "q_no", "qno", "item"), 1, 200),
                 Col("topic", True, "str", "Curriculum topic of the question", "Fractions", ("strand", "sub_strand")),
                 Col("max_marks", True, "float", "Marks the question is worth", 3, ("out_of", "max_score"), 0.5, 50),
                 Col("marks_awarded", True, "float", "Marks the learner got", 2, ("marks", "score", "mark"), 0, 50),
                 Col("error_type", False, "error", "Main mistake: " + ", ".join(ERROR_TYPES), "Careless slip",
                     ("mistake", "error")),
                 Col("difficulty", False, "str", "Easy, Medium or Hard", "Medium")]),
    Template("engagement", "engagement", "Classroom engagement (teacher ratings, once per term)",
             "Each subject teacher rates each learner once a term. Ratings are 1 (very low) to 5 (very high).", [
                 SID, TERM, SUBJ,
                 Col("participation_rating", False, "rating", "Takes part in class discussions (1-5)", 4,
                     ("participation",), 1, 5),
                 Col("answers_per_week", False, "float", "About how often they answer questions each week", 3, (), 0, 50),
                 Col("group_work_rating", False, "rating", "Takes part in group activities (1-5)", 3, ("group_work",), 1, 5),
                 Col("help_seeking_rating", False, "rating", "Asks for help when confused (1-5)", 3, ("help_seeking",), 1, 5),
                 Col("time_on_task_pct", False, "pct", "Share of lesson time spent on task (%)", 75,
                     ("time_on_task",), 0, 100),
                 Col("observation_rating", False, "rating", "Your overall classroom observation (1-5)", 4,
                     ("observation",), 1, 5),
                 Col("peer_learning_rating", False, "rating", "Learns with and helps classmates (1-5)", 3,
                     ("peer_learning",), 1, 5),
                 Col("classwork_completion_pct", False, "pct", "Share of class exercises completed (%)", 85,
                     ("classwork_completion", "classwork_completion_rate"), 0, 100),
                 Col("improvement_after_feedback", False, "float", "Marks gained when correcting work after feedback",
                     4, ("improvement",), -50, 50)]),
    Template("discipline", "discipline", "Discipline records", "Optional. One row per incident.", [
        Col("date", True, "date", "Day/Month/Year", "12/10/2026"), SID,
        Col("incident", True, "str", "What happened", "Fighting"), Col("action_taken", False, "str", "", "Parent called")]),
    Template("interventions", "interventions", "Interventions", "Support given to a learner, e.g. parent meeting, "
             "remedial classes. Used to measure attendance before and after.", [
                 Col("date", True, "date", "Day/Month/Year", "20/10/2026"), SID,
                 Col("intervention_type", True, "str", "e.g. Parent-teacher meeting, Remedial classes, Counselling",
                     "Remedial classes", ("type", "intervention")),
                 Col("notes", False, "str", "", "Weekly maths support")]),
    Template("question_bank", "question_bank", "Question bank (BECE past questions, textbooks, teacher questions)",
             "Questions your school owns or has the right to use. The question generator uses them as examples and "
             "can draw from them directly. Always record the source.", [
                 SUBJ, Col("class_label", True, "class", "Class the question suits, e.g. JHS 3", "JHS 3",
                           ("class", "level")),
                 Col("topic", True, "str", "Curriculum topic", "Linear equations"),
                 Col("question", True, "str", "Full question text", "Solve 3x - 5 = 16."),
                 Col("option_a", False, "str", "Multiple choice options (leave blank for written questions)", "5"),
                 Col("option_b", False, "str", "", "7"), Col("option_c", False, "str", "", "11"),
                 Col("option_d", False, "str", "", "21"),
                 Col("answer", True, "str", "Correct answer (letter or full answer)", "B"),
                 Col("explanation", False, "str", "Working or marking guide", "3x = 21, x = 7"),
                 Col("source", True, "str", "Where it came from, e.g. 'BECE 2019 Paper 1 Q14' or a textbook and page",
                     "Teacher-written"),
                 Col("difficulty", False, "str", "Easy, Medium or Hard", "Medium")]),
]}


# --------------------------------------------------------------------------- reading and templates
def read_file(name: str, data: bytes) -> pd.DataFrame:
    if name.lower().endswith((".xlsx", ".xlsm", ".xls")):
        xl = pd.ExcelFile(io.BytesIO(data))
        sheet = next((s for s in xl.sheet_names if s.strip().lower() == "data"), xl.sheet_names[0])
        df = xl.parse(sheet, dtype=object)
    else:
        for enc in ("utf-8-sig", "latin-1"):
            try:
                df = pd.read_csv(io.BytesIO(data), dtype=object, encoding=enc)
                break
            except UnicodeDecodeError:
                continue
    df = df.dropna(how="all").dropna(axis=1, how="all")
    return df


def _canon(h) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(h).strip().lower()).strip("_")


def template_excel(tpl: Template, students: pd.DataFrame | None = None, class_label: str | None = None,
                   dates: list | None = None, subjects: list | None = None) -> bytes:
    """An Excel template: 'Data' sheet to fill, plus 'Instructions' and 'Example'.
    Register/broadsheet templates come pre-filled with the class list and the school days or subjects."""
    out = io.BytesIO()
    cols = [c.name for c in tpl.cols]
    data = pd.DataFrame(columns=cols)
    if tpl.wide and students is not None and class_label:
        cl = students[students.class_label == class_label].sort_values(["surname", "first_name"])
        data = pd.DataFrame({"student_id": cl.student_id, "name": cl.first_name + " " + cl.surname})
        extra = [d.strftime("%d/%m/%Y") for d in dates] if tpl.key == "attendance_register" else (subjects or [])
        for e in extra:
            data[e] = ""
    example = pd.DataFrame([{c.name: c.example for c in tpl.cols}])
    if tpl.key == "attendance_register":
        example = pd.DataFrame([{"student_id": "BF-0012", "name": "Ama Mensah", "08/09/2026": "P", "09/09/2026": "L",
                                 "10/09/2026": "A", "11/09/2026": "P"}])
    if tpl.key == "broadsheet":
        example = pd.DataFrame([{"student_id": "BF-0012", "name": "Ama Mensah", "English Language": 62,
                                 "Mathematics": 48, "Science": 71, "Social Studies": 66}])
    instr = pd.DataFrame([{"column": c.name, "required": "Yes" if c.required else "No", "what to enter": c.help,
                           "also accepted as": ", ".join(c.aliases)} for c in tpl.cols])
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        data.to_excel(w, sheet_name="Data", index=False)
        pd.DataFrame({"How to use this template": [
            tpl.purpose, "Fill in the 'Data' sheet only. Keep the first row (column names).",
            "Use the same student IDs as in the student register.", "Dates: Day/Month/Year, e.g. 08/09/2026.",
            "Save as Excel (.xlsx) or CSV and upload it on the Import data page."]}).to_excel(
            w, sheet_name="Instructions", index=False, startrow=0)
        instr.to_excel(w, sheet_name="Instructions", index=False, startrow=7)
        example.to_excel(w, sheet_name="Example", index=False)
        for ws in w.book.worksheets:
            for col in ws.columns:
                ws.column_dimensions[col[0].column_letter].width = min(60, max(12, max(len(str(c.value or "")) for c in col) + 2))
    return out.getvalue()


# --------------------------------------------------------------------------- validation
@dataclass
class Result:
    clean: pd.DataFrame
    errors: pd.DataFrame         # row, column, value, problem
    notes: list
    total_rows: int

    @property
    def ok_rows(self) -> int:
        return len(self.clean)


def _match_columns(df: pd.DataFrame, tpl: Template):
    """Map the file's headers onto template columns. Returns (renamed df, unmatched headers)."""
    mapping, used = {}, set()
    for c in tpl.cols:
        names = {c.name, *c.aliases}
        for h in df.columns:
            if h not in used and _canon(h) in names:
                mapping[h] = c.name
                used.add(h)
                break
    unmatched = [h for h in df.columns if h not in used]
    return df.rename(columns=mapping), unmatched


def _melt_wide(df, tpl, opts, notes, errors):
    """Turn a register (dates as columns) or broadsheet (subjects as columns) into long rows."""
    id_cols = [c for c in ("student_id", "name") if c in df.columns]
    others = [c for c in df.columns if c not in id_cols]
    keep, skipped = [], []
    for h in others:
        if tpl.key == "attendance_register":
            d = to_date(h)
            (keep if not pd.isna(d) else skipped).append(h)
        else:
            (keep if norm_subject(h) else skipped).append(h)
    if skipped:
        notes.append(f"Ignored {len(skipped)} column(s) that are not "
                     f"{'dates' if tpl.key == 'attendance_register' else 'subjects'}: {', '.join(map(str, skipped[:8]))}")
    if not keep:
        errors.append(dict(row="-", column="-", value="",
                           problem="No date columns found." if tpl.key == "attendance_register" else
                           "No subject columns found."))
        return pd.DataFrame()
    long = df.reset_index().melt(id_vars=["index"] + id_cols, value_vars=keep, var_name="_col", value_name="_val")
    long = long[long["_val"].notna() & (long["_val"].astype(str).str.strip() != "")]
    if tpl.key == "attendance_register":
        long = long.assign(date=long["_col"].map(to_date), status=long["_val"], reason="",
                           subject=opts.get("subject") or "")
    else:
        long = long.assign(subject=long["_col"].map(norm_subject), score=long["_val"], term=opts["term"],
                           assessment_type=opts["assessment_type"], assessment_name=opts["assessment_name"],
                           max_score=opts["max_score"], date=opts.get("date") or "", submitted="Yes")
    return long.set_index("index").drop(columns=["_col", "_val"])


def validate(df: pd.DataFrame, tpl_key: str, known_students: set | None = None, opts: dict | None = None,
             term_dates: dict | None = None) -> Result:
    tpl = TEMPLATES[tpl_key]
    opts = opts or {}
    notes, errs = [], []
    total = len(df)
    df, unmatched = _match_columns(df, tpl)
    if tpl.wide:
        if "student_id" not in df.columns:
            return Result(pd.DataFrame(), pd.DataFrame([dict(row="-", column="student_id", value="",
                          problem="Column 'student_id' not found.")]), notes, total)
        df = _melt_wide(df, tpl, opts, notes, errs)
        target = TEMPLATES["attendance_daily" if tpl.key == "attendance_register" else "assessments"]
    else:
        target = tpl
        if unmatched:
            notes.append(f"Ignored columns not in the template: {', '.join(map(str, unmatched[:10]))}")
    missing = [c.name for c in target.cols if c.required and c.name not in df.columns]
    if missing:
        errs.append(dict(row="-", column=", ".join(missing), value="",
                         problem="Required column(s) missing. Download the template to see the expected columns."))
        return Result(pd.DataFrame(), pd.DataFrame(errs), notes, total)

    out = pd.DataFrame(index=df.index)
    bad = pd.Series(False, index=df.index)

    def flag(mask, col, problem):
        nonlocal bad
        for i in df.index[mask & ~bad]:
            errs.append(dict(row=int(i) + 2, column=col, value=str(df.at[i, col]) if col in df.columns else "",
                             problem=problem))
        bad = bad | mask

    for c in target.cols:
        if c.name not in df.columns:
            out[c.name] = None
            continue
        raw = df[c.name]
        blank = raw.isna() | (raw.astype(str).str.strip() == "")
        if c.required:
            flag(blank, c.name, "Required value is empty.")
        if c.kind == "str":
            v = raw.where(~blank, "").astype(str).str.strip()
        elif c.kind == "date":
            v = raw.map(to_date)
            flag(v.isna() & ~blank, c.name, "Not a valid date. Use Day/Month/Year, e.g. 08/09/2026.")
            v = v.dt.strftime("%Y-%m-%d").where(v.notna(), "")
        elif c.kind in ("int", "float", "rating", "pct"):
            v = pd.to_numeric(raw.astype(str).str.replace("%", "").str.replace(",", "").str.strip(), errors="coerce")
            flag(v.isna() & ~blank, c.name, "Not a number.")
            if c.lo is not None:
                flag(v.notna() & (v < c.lo), c.name, f"Too small (minimum {c.lo:g}).")
            if c.hi is not None:
                flag(v.notna() & (v > c.hi), c.name, f"Too large (maximum {c.hi:g}).")
            if c.kind in ("int", "rating"):
                v = v.round()
        elif c.kind == "term":
            v = raw.map(norm_term)
            flag(v.isna() & ~blank, c.name, "Term must be 1, 2 or 3.")
        elif c.kind == "subject":
            v = raw.map(norm_subject)
            flag(v.isna() & ~blank, c.name, "Unknown subject. Use names like Mathematics, English Language, "
                                            "Science, Social Studies, RME, Computing.")
            v = v.fillna("")
        elif c.kind == "class":
            v = raw.map(norm_class)
            flag(v.isna() & ~blank, c.name, "Unknown class. Use Class 1 to Class 6 or JHS 1 to JHS 3.")
        elif c.kind == "gender":
            v = raw.map(norm_gender)
            flag(v.isna() & ~blank, c.name, "Use Male or Female (M/F).")
        elif c.kind == "att":
            v = raw.map(norm_att)
            flag((v == "?"), c.name, "Unknown mark. Use P (present), A (absent), L (late) or E (left early).")
        elif c.kind == "yesno":
            v = raw.map(norm_yesno)
            flag(v.isna(), c.name, "Use Yes or No.")
        elif c.kind == "error":
            v = raw.map(norm_error)
            flag(v.isna(), c.name, "Unknown error type. Use one of: " + ", ".join(ERROR_TYPES))
        elif c.kind == "atype":
            v = raw.map(norm_atype)
            flag(v.isna() & ~blank, c.name, "Unknown type. Use: " + ", ".join(ASSESSMENT_TYPES))
        else:
            v = raw
        out[c.name] = v

    # cross-field checks
    if "score" in out and "max_score" in out:
        flag(out.score.notna() & out.max_score.notna() & (out.score > out.max_score), "score",
             "Score is higher than the maximum score.")
    if "marks_awarded" in out:
        flag(out.marks_awarded > out.max_marks, "marks_awarded", "Marks awarded are higher than the question's marks.")
    if "student_id" in out and tpl.table != "students" and tpl.table != "question_bank":
        if known_students is not None:
            unknown = ~out.student_id.astype(str).isin(known_students)
            flag(unknown & out.student_id.notna(), "student_id",
                 "Student ID not in the student register. Import the register first, or check the ID.")
    if tpl.table == "assessments" and "submitted" in out:
        flag(out.score.isna() & out.submitted.eq(True), "score", "Score is empty. If the work was not submitted, "
                                                                "put No in 'submitted'.")
    if "date" in out and term_dates and tpl.table in ("attendance",):
        d = pd.to_datetime(out.date, errors="coerce")
        in_term = pd.Series(False, index=out.index)
        for s, e in term_dates.values():
            in_term |= (d >= pd.Timestamp(s)) & (d <= pd.Timestamp(e))
        n_out = int((~in_term & d.notna()).sum())
        if n_out:
            notes.append(f"{n_out} row(s) have dates outside the term dates in School setup. They will still be "
                         "imported; check the dates if this is unexpected.")
        if (d.dt.weekday >= 5).any():
            notes.append(f"{int((d.dt.weekday >= 5).sum())} row(s) are on a Saturday or Sunday.")

    # duplicates inside the file
    from school_store import KEYS
    key = [k for k in KEYS[tpl.table] if k in out.columns]
    dup = out.duplicated(key, keep="last") & ~bad
    if dup.any():
        notes.append(f"{int(dup.sum())} duplicate row(s) in the file; the last one of each is kept.")
    clean = out[~bad & ~dup].copy()
    for c in target.cols:  # whole numbers stay whole (term 1, not 1.0) so re-imports match existing records
        if c.kind in ("term", "int") and c.name in clean:
            clean[c.name] = clean[c.name].astype("Int64")
    if tpl.table == "attendance":
        clean["subject"] = clean.get("subject", "").fillna("") if "subject" in clean else ""
    if tpl.table == "students":
        clean["status"] = clean["status"].replace("", "Active")
    if tpl.table == "assessments":
        clean["submitted"] = clean["submitted"].map({True: "Yes", False: "No"})
    errors = pd.DataFrame(errs, columns=["row", "column", "value", "problem"])
    errors = errors.iloc[errors.row.map(lambda r: r if isinstance(r, int) else -1).argsort(kind="stable")]
    if tpl.wide and not errors.empty:
        errors["problem"] = errors["problem"] + " (row number refers to the learner's row in your sheet)"
    return Result(clean.reset_index(drop=True), errors, notes, total)
