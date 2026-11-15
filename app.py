"""School AI Pilot: one school per installation.

Run:  .venv/bin/streamlit run app.py
"""
import hmac
import io
import json
import sys
import zipfile
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import features  # noqa: E402
import importer as imp  # noqa: E402
import school_store as store  # noqa: E402
from ai import learning_profile as lp  # noqa: E402
from ai import llm  # noqa: E402
from ai import parent_advisor as pa  # noqa: E402
from ai import question_generator as qg  # noqa: E402
from ai.knowledge_bot import KnowledgeBot  # noqa: E402
from curriculum import LABEL_TO_LEVEL, LEVEL_LABEL, SUBJECTS, band, topics_for  # noqa: E402

SERIES = "#2a78d6"  # single-series chart colour
DEMO_SCHOOL = "SCH05"
SAMPLES = ROOT / "sample_imports"
SCHOOL_MODEL = store.DATA_DIR / "early_warning_school.joblib"
SAMPLE_PLAN = [
    ("1_student_register.xlsx", "students", {}), ("2_attendance_daily_2025_26.csv", "attendance_daily", {}),
    ("3_attendance_register_JHS2_term3.xlsx", "attendance_register", {}), ("4_scores_2025_26.csv", "assessments", {}),
    ("5_broadsheet_JHS2_term2_exam.xlsx", "broadsheet",
     dict(term=2, assessment_type="exam", assessment_name="End of term", max_score=100, date="23/03/2026")),
    ("6_question_marks_2025_26.csv", "question_marks", {}), ("7_engagement_ratings_2025_26.xlsx", "engagement", {}),
    ("8_discipline_2025_26.csv", "discipline", {}), ("9_interventions_2025_26.csv", "interventions", {})]
st.set_page_config(page_title="School AI Pilot", page_icon="🎓", layout="wide")


def password_gate():
    """When APP_PASSWORD is set in Streamlit secrets (the hosted version), ask for it once per visit."""
    try:
        pw = st.secrets.get("APP_PASSWORD")
    except Exception:  # no secrets file: running locally
        pw = None
    if not pw or st.session_state.get("auth_ok"):
        return
    st.title("🎓 School AI Pilot")
    st.write("Trial version for the team. All students, schools and documents are fictional.")
    entered = st.text_input("Team password", type="password")
    if entered:
        if hmac.compare_digest(entered.encode(), str(pw).encode()):
            st.session_state.auth_ok = True
            st.rerun()
        st.error("Wrong password.")
    st.stop()


password_gate()


# --------------------------------------------------------------------------- data loading
@st.cache_data
def load_demo():
    d = ROOT / "data"
    students = pd.read_csv(d / "students.csv")
    students = students[students.school_id == DEMO_SCHOOL]
    sst = pd.read_csv(d / "student_subject_term.csv", dtype={"grade": str})
    sst = sst[sst.school_id == DEMO_SCHOOL]
    qr = pd.read_csv(d / "question_responses.csv")
    return dict(students=students, sst=sst, qr=qr[qr.student_id.isin(students.student_id)])


@st.cache_data
def load_school(version: str):
    return features.build()


def data_version() -> str:
    log = store.import_log()
    return f"{len(log)}|{log['at'].max() if len(log) else ''}"


@st.cache_resource
def load_model(path: str, mtime: float):
    try:
        return joblib.load(path)
    except Exception:  # e.g. saved with a different scikit-learn version: retrain the base model
        from train_models import train_early_warning
        train_early_warning()
        return joblib.load(ROOT / "models" / "early_warning.joblib")


@st.cache_resource
def load_bot(docs_dir: str, school: str, office: str, stamp: str):
    return KnowledgeBot(Path(docs_dir), school, office)


profile = store.get_profile()
has_school_data = store.has_data()

# --------------------------------------------------------------------------- sidebar
st.sidebar.title("🎓 School AI Pilot")
st.sidebar.caption(profile["school_name"] or "School not set up yet")
mode = st.sidebar.radio("Data", ["My school", "Demo school (synthetic)"], index=0 if has_school_data else 1,
                        help="'My school' uses the data your school has imported. The demo uses simulated data.")
DEMO = mode.startswith("Demo")
page = st.sidebar.radio("Go to", [
    "🏫 School setup", "📥 Import data", "📊 Overview", "🚨 Early-warning model", "🧭 Learning profile & gaps",
    "📝 Question generator", "💬 Parent knowledge bot", "👪 Parent advisor"], index=2 if (DEMO or has_school_data) else 0)
st.sidebar.divider()
st.sidebar.subheader("Claude API")
key = st.sidebar.text_input("API key (optional)", type="password",
                            help="Without a key, the generative features run in offline mode.")
if key:
    llm.set_api_key(key)
use_llm = st.sidebar.toggle("Use Claude", value=llm.credentials_available(), disabled=not llm.credentials_available())
st.sidebar.caption(f"Model: `{llm.MODEL}`" if use_llm else "Offline mode: templates, question bank and retrieval.")
st.sidebar.caption("Teachers have the final say on every AI output.")

if DEMO:
    D = load_demo()
    school_name = "Madina Bright Future Academy (demo, synthetic)"
else:
    D = load_school(data_version()) if has_school_data else {}
    school_name = profile["school_name"] or "My school"
students, sst, qr = D.get("students", pd.DataFrame()), D.get("sst", pd.DataFrame()), D.get("qr", pd.DataFrame())
if not students.empty:
    students = students.assign(name=students.first_name + " " + students.surname)

model_path = SCHOOL_MODEL if (not DEMO and SCHOOL_MODEL.exists()) else ROOT / "models" / "early_warning.joblib"


def need_data(what="student data") -> bool:
    if students.empty or sst.empty:
        st.info(f"No {what} yet for {school_name}. Go to **📥 Import data**, or switch the sidebar to the demo school.")
        return True
    return False


def pick_student(key: str):
    c1, c2 = st.columns(2)
    classes = [c for c in LEVEL_LABEL.values() if c in set(students.class_label)]
    cls = c1.selectbox("Class", classes, index=len(classes) - 2 if len(classes) > 1 else 0, key=f"{key}_cls")
    pool = students[students.class_label == cls].sort_values("name")
    pool = pool[pool.student_id.isin(sst.student_id)]
    names = pool.set_index("student_id").name
    return c2.selectbox("Student", pool.student_id, format_func=lambda s: f"{names[s]} ({s})", key=f"{key}_stu")


def show_questions(qs):
    for i, q in enumerate(qs, 1):
        st.markdown(f"**{i}. {q['question']}**  \n<small>{q.get('topic', '')} · {q.get('difficulty', '')} · "
                    f"{q.get('type', '').replace('_', ' ')} · <i>{q.get('source', '')}</i></small>", unsafe_allow_html=True)
        for o in q.get("options") or []:
            st.markdown(f"&nbsp;&nbsp;&nbsp;{o}")
        with st.expander("Answer and marking guide"):
            st.write(f"**Answer:** {q['answer']}")
            st.write(q["explanation"])
    text = "\n\n".join(f"{i}. {q['question']}\n" + "\n".join(q.get("options") or []) + f"\nAnswer: {q['answer']}"
                       for i, q in enumerate(qs, 1))
    st.download_button("Download as text", text, file_name="questions.txt")


# =========================================================================== pages
if page == "🏫 School setup":
    st.title("School setup")
    st.write("This installation serves **one school**. Set it up once; the data collector then imports data on the "
             "**Import data** page.")
    with st.form("profile"):
        c1, c2 = st.columns(2)
        name = c1.text_input("School name", profile["school_name"])
        area = c2.text_input("Town / area", profile["area"], placeholder="e.g. Madina, Accra")
        c3, c4, c5 = st.columns(3)
        stype = c3.selectbox("Type", ["Public", "Private"], index=["Public", "Private"].index(profile["school_type"]))
        langs = ["Twi", "Ga", "Ewe", "Fante", "Dagbani", "Hausa", "Other"]
        glang = c4.selectbox("Ghanaian language taught", langs, index=langs.index(profile["ghanaian_language"])
                             if profile["ghanaian_language"] in langs else 0)
        year = c5.text_input("Academic year", profile["academic_year"])
        st.markdown("**Term dates** (used to place attendance and scores in the right term)")
        td = {}
        cols = st.columns(3)
        for i, t in enumerate(("1", "2", "3")):
            s, e = profile["term_dates"][t]
            with cols[i]:
                a = st.date_input(f"Term {t} starts", pd.Timestamp(s), format="DD/MM/YYYY")
                b = st.date_input(f"Term {t} ends", pd.Timestamp(e), format="DD/MM/YYYY")
            td[t] = [str(a), str(b)]
        c6, c7, c8 = st.columns(3)
        ca = c6.selectbox("Class score (CA) weight", [30, 40, 50], index=[30, 40, 50].index(int(profile["ca_weight"])),
                          help="Share of the term result from continuous assessment; the rest is the exam.")
        phone = c7.text_input("Office phone", profile["office_phone"])
        email = c8.text_input("Office email", profile["office_email"])
        if st.form_submit_button("Save school details", type="primary"):
            store.save_profile(dict(profile, school_name=name, area=area, school_type=stype, ghanaian_language=glang,
                                    academic_year=year, term_dates=td, ca_weight=ca, office_phone=phone,
                                    office_email=email))
            st.cache_data.clear()
            st.success("Saved.")
            st.rerun()

    st.subheader("School documents for the parent bot")
    st.write("Upload the handbook, fee schedule, calendar and FAQs (Word .docx, .txt or .md). The parent bot answers "
             "only from these. Use headings in your documents for the best results.")
    store.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    ups = st.file_uploader("Add documents", type=["docx", "txt", "md"], accept_multiple_files=True)
    if ups and st.button("Save documents"):
        for u in ups:
            (store.DOCS_DIR / Path(u.name).name).write_bytes(u.getvalue())
        st.success(f"Saved {len(ups)} document(s).")
        st.rerun()
    docs = sorted(store.DOCS_DIR.glob("*"))
    if docs:
        for f in docs:
            c1, c2 = st.columns([5, 1])
            c1.write(f"📄 {f.name}")
            if c2.button("Remove", key=f"rm_{f.name}"):
                f.unlink()
                st.rerun()
    else:
        st.caption("No documents yet; the bot uses the sample documents until you add yours.")

    with st.expander("Danger zone"):
        st.write("Delete all imported data for this school (the school details and documents are kept).")
        if st.checkbox("I understand this cannot be undone"):
            if st.button("Delete all imported data", type="primary"):
                store.clear()
                if SCHOOL_MODEL.exists():
                    SCHOOL_MODEL.unlink()
                st.cache_data.clear()
                st.success("All imported data deleted.")
                st.rerun()

elif page == "📥 Import data":
    st.title("Import data")
    if not store.is_configured():
        st.warning("Set up the school first on **🏫 School setup** (name and term dates).")
    st.write(f"For the data collector at **{profile['school_name'] or 'your school'}**. Download a template, fill it "
             "in Excel, upload it, fix anything the check finds, then import. Re-importing the same record replaces it, "
             "so you can correct mistakes by uploading again.")

    st.subheader("This week's data check")
    h = features.health()
    st.dataframe(h.assign(OK=h.OK.map({True: "✅", False: "⚠️"})), hide_index=True, width="stretch")

    st.subheader("1. What are you importing?")
    order = ["students", "attendance_register", "attendance_daily", "assessments", "broadsheet", "question_marks",
             "engagement", "discipline", "interventions", "question_bank"]
    tkey = st.selectbox("Type of data", order, format_func=lambda k: imp.TEMPLATES[k].title)
    tpl = imp.TEMPLATES[tkey]
    st.caption(tpl.purpose)
    known = set(store.read("students").student_id) if tkey not in ("students", "question_bank") else None
    if known is not None and not known:
        st.warning("Import the **Student register** first. Every other file is checked against it.")

    st.subheader("2. Download the template")
    reg = store.read("students")
    if tkey in ("attendance_register", "broadsheet") and not reg.empty:
        c1, c2 = st.columns(2)
        cls = c1.selectbox("Class", [c for c in LEVEL_LABEL.values() if c in set(reg.class_label)])
        if tkey == "attendance_register":
            term_t = c2.selectbox("Term", ["1", "2", "3"])
            s, e = profile["term_dates"][term_t]
            data = imp.template_excel(tpl, reg, cls, dates=list(pd.bdate_range(s, e)))
            fname = f"register_{cls.replace(' ', '')}_term{term_t}.xlsx"
            st.caption("Pre-filled with the class list and every weekday of the term. Leave holidays blank.")
        else:
            data = imp.template_excel(tpl, reg, cls, subjects=SUBJECTS[band(LABEL_TO_LEVEL[cls])])
            fname = f"broadsheet_{cls.replace(' ', '')}.xlsx"
            st.caption("Pre-filled with the class list and the class's subjects.")
    else:
        data, fname = imp.template_excel(tpl), f"template_{tkey}.xlsx"
    st.download_button(f"⬇️ Download {fname}", data, file_name=fname)
    with st.expander("What goes in each column"):
        st.dataframe(pd.DataFrame([{"column": c.name, "required": "Yes" if c.required else "", "what to enter": c.help}
                                   for c in tpl.cols]), hide_index=True, width="stretch")

    st.subheader("3. Upload the filled file")
    opts = {}
    if tkey == "broadsheet":
        c = st.columns(5)
        opts = dict(term=int(c[0].selectbox("Term", [1, 2, 3])),
                    assessment_type=c[1].selectbox("Type", ["exam", "mid_term", "class_test", "previous_exam"]),
                    assessment_name=c[2].text_input("Name", "End of term"),
                    max_score=c[3].number_input("Marked out of", 1, 500, 100),
                    date=str(c[4].date_input("Date", format="DD/MM/YYYY")))
    if tkey == "attendance_register":
        sub = st.selectbox("Register for", ["Whole day"] + sorted({s for v in SUBJECTS.values() for s in v}),
                           help="Choose a subject only if this is a lesson register.")
        opts = {"subject": "" if sub == "Whole day" else sub}
    up = st.file_uploader("Excel (.xlsx) or CSV file", type=["xlsx", "xls", "csv"], key=f"up_{tkey}")
    if up is not None:
        try:
            raw = imp.read_file(up.name, up.getvalue())
        except Exception as e:  # unreadable file
            st.error(f"Could not read the file: {e}")
            st.stop()
        res = imp.validate(raw, tkey, known, opts, profile["term_dates"])
        st.subheader("4. Check")
        c = st.columns(3)
        c[0].metric("Rows in file", f"{res.total_rows:,}")
        c[1].metric("Records ready to import", f"{res.ok_rows:,}")
        c[2].metric("Problems found", len(res.errors))
        for n in res.notes:
            st.info(n)
        if len(res.errors):
            st.error("Some rows have problems. Fix them in your file and upload again, or import only the good rows.")
            st.dataframe(res.errors.head(200), hide_index=True, width="stretch")
            st.download_button("Download the list of problems", res.errors.to_csv(index=False), "problems.csv")
        if res.ok_rows:
            with st.expander("Preview the first 20 records"):
                st.dataframe(res.clean.head(20), hide_index=True, width="stretch")
            label = (f"Import {res.ok_rows:,} records" if not len(res.errors) else
                     f"Import {res.ok_rows:,} good records and skip the problem rows")
            if st.button(label, type="primary"):
                added, replaced = store.upsert(tpl.table, res.clean, up.name, len(res.errors))
                st.cache_data.clear()
                st.success(f"Done: {added:,} new records, {replaced:,} updated.")

    st.subheader("Practise with sample files")
    st.write("Sample files for a fictional school, in the layouts schools keep: register, daily attendance, the class "
             "register, scores, a broadsheet, marks per question, engagement ratings, discipline and interventions.")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for f in sorted(SAMPLES.glob("*")):
            z.write(f, f.name)
    st.download_button("⬇️ Download sample files (zip)", buf.getvalue(), "sample_imports.zip")
    if st.button("Import all sample files into this installation"):
        if not store.is_configured():
            store.save_profile(dict(profile, school_name="Madina Bright Future Academy", area="Madina, Accra",
                                    academic_year="2025/2026", term_dates={
                                        "1": ["2025-09-09", "2025-12-19"], "2": ["2026-01-13", "2026-04-02"],
                                        "3": ["2026-04-28", "2026-07-24"]}))
            profile = store.get_profile()
        bar = st.progress(0.0)
        for i, (f, k, o) in enumerate(SAMPLE_PLAN):
            df = imp.read_file(f, (SAMPLES / f).read_bytes())
            kn = set(store.read("students").student_id) if k != "students" else None
            r = imp.validate(df, k, kn, o, profile["term_dates"])
            store.upsert(imp.TEMPLATES[k].table, r.clean, f, len(r.errors))
            bar.progress((i + 1) / len(SAMPLE_PLAN), text=f"{f}: {r.ok_rows:,} records")
        st.cache_data.clear()
        st.success("Sample files imported. Switch the sidebar to 'My school' to use them.")

    with st.expander("Import history"):
        st.dataframe(store.import_log(), hide_index=True, width="stretch")

elif page == "📊 Overview":
    st.title(f"Overview: {school_name}")
    if not need_data():
        terms = sorted(sst.term.unique())
        cur = sst[sst.term == max(terms)]
        c = st.columns(5)
        c[0].metric("Learners", f"{students.shape[0]:,}")
        c[1].metric("Classes", students.class_label.nunique())
        c[2].metric("Records (student × subject × term)", f"{len(sst):,}")
        c[3].metric(f"Attendance, Term {max(terms)}", f"{cur.drop_duplicates('student_id').attendance_rate.mean():.0%}")
        done = sst[sst.exam_score_100.notna()]
        c[4].metric("At-risk records (with exam)", f"{done.at_risk.mean():.0%}" if len(done) else "–")
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Average exam score by class")
            if len(done):
                by = done.groupby("class_level").exam_score_100.mean().rename("Average exam score")
                by.index = [LEVEL_LABEL[i] for i in by.index]
                st.bar_chart(by, color=SERIES, height=280)
        with c2:
            st.subheader(f"Attendance by class, Term {max(terms)}")
            att = cur.drop_duplicates("student_id").groupby("class_level").attendance_rate.mean().rename("Attendance rate")
            att.index = [LEVEL_LABEL[i] for i in att.index]
            st.bar_chart(att, color=SERIES, height=280)
        st.subheader("Records")
        st.dataframe(sst.head(500), width="stretch")
        st.caption("Showing the first 500 rows.")

elif page == "🚨 Early-warning model":
    st.title("Early-warning model")
    st.write("Flags learners likely to **score below 50 or drop 15+ points** in the end-of-term exam, from what "
             "teachers know before the exam. Demographics, home background and fees are not used.")
    if not need_data():
        ew = load_model(str(model_path), model_path.stat().st_mtime)
        st.caption("Model: " + ("retrained with this school's results" if model_path == SCHOOL_MODEL
                                else "trained on simulated students"))
        with st.expander("Model card: how well does it work?"):
            m = ew.metrics
            h = m["held_out_50_students"]
            c = st.columns(4)
            c[0].metric("ROC-AUC (50 unseen simulated students)", h["roc_auc"])
            c[1].metric("At-risk caught (recall)", f"{h['recall']:.0%}")
            c[2].metric("Flags correct (precision)", f"{h['precision']:.0%}")
            c[3].metric("Threshold", ew.threshold)
            if "school_retrain" in m:
                st.write("**On this school's own students:**")
                st.json(m["school_retrain"])
            st.caption("Accuracy must be checked on real students the model never saw before it is trusted.")
        if not DEMO:
            labelled = int(sst.at_risk.notna().sum())
            with st.expander("Retrain with our school's results"):
                st.write(f"{labelled:,} records with exam results are available. Retraining mixes them with the "
                         "simulated students and tests on 50 of your own students held out.")
                if st.button("Retrain now", disabled=labelled < 200):
                    from train_models import train_with_school
                    with st.spinner("Training (about a minute)..."):
                        res = train_with_school(sst, SCHOOL_MODEL)
                    st.json(res)
                    st.cache_resource.clear()

        c1, c2 = st.columns(2)
        terms = sorted(sst.term.unique(), reverse=True)
        term = c1.selectbox("Term", terms, format_func=lambda t: f"Term {t}" + (
            " (no exam yet: prediction)" if sst[sst.term == t].exam_score_100.isna().all() else ""))
        cls = c2.selectbox("Class", ["All"] + [c for c in LEVEL_LABEL.values() if c in set(students.class_label)])
        rows = sst[sst.term == term]
        if cls != "All":
            rows = rows[rows.class_label == cls]
        rows = rows.assign(risk=ew.predict(rows))
        rows["flag"] = rows.risk >= ew.threshold
        per = rows.groupby("student_id").agg(class_label=("class_label", "first"), max_risk=("risk", "max"),
                                             subjects_flagged=("flag", "sum"), attendance=("attendance_rate", "first"),
                                             avg_ca=("ca_score_30", "mean")).reset_index()
        per = per.merge(students[["student_id", "name"]]).sort_values("max_risk", ascending=False)
        st.subheader(f"{(per.subjects_flagged > 0).sum()} of {len(per)} learners flagged in at least one subject")
        st.dataframe(per[["student_id", "name", "class_label", "subjects_flagged", "max_risk", "attendance", "avg_ca"]]
                     .style.format({"max_risk": "{:.0%}", "attendance": "{:.0%}", "avg_ca": "{:.1f}/30"}, na_rep="–"),
                     width="stretch", height=300, hide_index=True)
        st.download_button("Download flag list", per.to_csv(index=False), f"flags_term{term}.csv")

        st.subheader("Why is a learner flagged?")
        names = per.set_index("student_id").name
        sid = st.selectbox("Learner", per.student_id, format_func=lambda s: f"{names[s]} ({s})")
        srows = rows[rows.student_id == sid].sort_values("risk", ascending=False)
        for _, r in srows.iterrows():
            icon = "🔴" if r.risk >= ew.threshold else ("🟡" if r.risk >= ew.threshold * 0.6 else "🟢")
            with st.expander(f"{icon} {r.subject}: risk {r.risk:.0%}", expanded=bool(r.flag)):
                for reason in ew.reasons(r):
                    st.markdown(f"- {reason}")
                if not pd.isna(r.exam_score_100):
                    st.caption(f"Actual exam result: {r.exam_score_100:.0f}.")

elif page == "🧭 Learning profile & gaps":
    st.title("Learning profile & gap analysis")
    if not need_data():
        sid = pick_student("lp")
        s = students.set_index("student_id").loc[sid]
        st.markdown(f"### {s['name']} · {s.class_label}")
        subj = lp.subject_summary(sst, sid)
        if not subj.empty:
            st.subheader("Results by subject")
            st.dataframe(subj.style.format({c: "{:.1f}" for c in subj.columns if c.startswith("Term")} |
                                           {"Trend": "{:+.1f}", "Lesson attendance": "{:.0%}"}, na_rep="–"),
                         width="stretch", hide_index=True)
        mastery = lp.topic_mastery(qr, sid) if not qr.empty else pd.DataFrame()
        if mastery.empty:
            st.info("No marks per question for this learner yet. Import **Marks per question** to see topic mastery "
                    "and gaps.")
        else:
            st.subheader("Topic mastery")
            subject = st.radio("Subject", sorted(mastery.subject.unique()), horizontal=True)
            ms = mastery[mastery.subject == subject].sort_values("mastery")
            st.bar_chart(ms.set_index("topic").mastery.rename("Mastery (0-1)"), color=SERIES, horizontal=True, height=320)
            st.dataframe(ms[["topic", "mastery", "class_average", "status", "main_error", "questions"]]
                         .style.format({"mastery": "{:.0%}", "class_average": "{:.0%}"}, na_rep="–"),
                         width="stretch", hide_index=True)
            st.subheader("Gap analysis and practice plan")
            gaps = lp.gap_analysis(mastery)
            for g in gaps[:5]:
                st.markdown(f"**{g['subject']} · {g['topic']}**: mastery {g['mastery']:.0%} (class "
                            f"{g['class_average']:.0%}). Main error: *{g['main_error']}*.  \n➡️ {g['action']}")
            if gaps:
                g0 = gaps[0]
                if st.button(f"Generate practice questions for: {g0['topic']}"):
                    try:
                        qs, src = qg.generate(g0["subject"], int(s.class_level), g0["topic"], "Easy", 5,
                                              error_focus=g0["main_error"], use_llm=use_llm,
                                              school_bank=None if DEMO else store.read("question_bank"))
                        st.caption("A teacher must review before use.")
                        show_questions(qs)
                    except llm.LLMError as e:
                        st.error(str(e))

elif page == "📝 Question generator":
    st.title("Question generator")
    bank_df = pd.DataFrame() if DEMO else store.read("question_bank")
    st.write("Curriculum-aligned questions in BECE and textbook style, set by difficulty and mixed in form and context. "
             + ("Claude writes new questions, guided by examples from the question bank." if use_llm else
                "Offline: question bank plus randomised Mathematics templates.")
             + (f" Your school's bank has **{len(bank_df)}** questions." if len(bank_df) else
                " Import your own BECE past questions and textbook items on the Import data page to use them here."))
    c1, c2, c3 = st.columns(3)
    cls = c1.selectbox("Class", list(LEVEL_LABEL.values()), index=8)
    level = LABEL_TO_LEVEL[cls]
    subject = c2.selectbox("Subject", SUBJECTS[band(level)], index=1)
    topic = c3.selectbox("Topic", topics_for(subject, level) or ["General"])
    c4, c5, c6 = st.columns(3)
    diff = c4.select_slider("Difficulty", ["Easy", "Medium", "Hard"], value="Medium")
    n = c5.number_input("Number of questions", 1, 15, 6)
    mix = c6.selectbox("Cognitive mix", ["Mostly reasoning and application", "Balanced", "Mostly recall"])
    err = st.selectbox("Target a common mistake (optional)", ["None"] + list(lp.ERROR_ACTIONS))
    if "q_history" not in st.session_state:
        st.session_state.q_history = []
    if st.button("Generate", type="primary"):
        with st.spinner("Generating..."):
            try:
                qs, src = qg.generate(subject, level, topic, diff, int(n), mix, None if err == "None" else err,
                                      use_llm=use_llm, avoid=st.session_state.q_history, school_bank=bank_df)
                st.session_state.q_history += [q["question"] for q in qs]
                st.session_state.q_last = qs
            except llm.LLMError as e:
                st.error(str(e))
    if st.session_state.get("q_last"):
        st.caption("Press Generate again for a different set; questions already shown are avoided. "
                   "A teacher must review every question before use.")
        show_questions(st.session_state.q_last)

elif page == "💬 Parent knowledge bot":
    st.title("Parent knowledge bot")
    own_docs = store.DOCS_DIR.exists() and any(store.DOCS_DIR.glob("*"))
    use_own = own_docs and not DEMO
    docs_dir = store.DOCS_DIR if use_own else ROOT / "school_docs"
    bot_school = profile["school_name"] if use_own else "Madina Bright Future Academy"
    office = (f"Please contact the school office: {profile['office_phone']} {profile['office_email']}".strip()
              if use_own and (profile["office_phone"] or profile["office_email"]) else "")
    st.write(f"Answers parents' questions from **{bot_school}**'s documents, showing the source, and says when the "
             "documents don't cover a question.")
    if not use_own:
        st.caption("Using the sample documents. Upload your own on **🏫 School setup**.")
    if not use_llm:
        st.info("Offline mode shows the closest passages. With Claude, answers are written from those passages and the "
                "bot refuses when the documents don't contain the answer.")
    stamp = "|".join(f"{f.name}{f.stat().st_mtime}" for f in sorted(Path(docs_dir).glob("*")))
    bot = load_bot(str(docs_dir), bot_school, office, stamp)
    if "chat" not in st.session_state:
        st.session_state.chat = []
    examples = ["How much are school fees for JHS 3?", "When does school reopen in January?",
                "My child has malaria. What should I do?", "Can I pay with mobile money?"]
    cols = st.columns(len(examples))
    clicked = next((q for c, q in zip(cols, examples) if c.button(q)), None)
    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    q = st.chat_input("Ask a question about the school...") or clicked
    if q:
        st.session_state.chat.append(dict(role="user", content=q))
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            try:
                res = bot.answer(q, use_llm=use_llm)
                st.markdown(res["answer"])
                if res["sources"]:
                    with st.expander("Documents searched"):
                        for h in res["sources"]:
                            st.markdown(f"**{h['source']} > {h['section']}** (match {h['score']:.2f})")
                st.session_state.chat.append(dict(role="assistant", content=res["answer"]))
            except llm.LLMError as e:
                st.error(str(e))

elif page == "👪 Parent advisor":
    st.title("Parent advisor")
    st.write("Explains a learner's progress in plain language. Every summary is a **draft that a teacher must approve**.")
    if not need_data():
        sid = pick_student("pa")
        lang = st.selectbox("Language", pa.LANGUAGES, disabled=not use_llm,
                            help="Other languages need Claude. Have a fluent speaker check translations.")
        profile_p = lp.build_profile(students, sst, qr, sid)
        with st.expander("Profile sent to the advisor"):
            st.code(json.dumps(profile_p, indent=1, default=str), language="json")
        if st.button("Draft parent summary", type="primary"):
            with st.spinner("Writing..."):
                try:
                    text, src = pa.summarise(profile_p, lang if use_llm else "English", use_llm=use_llm)
                    st.session_state[f"draft_{sid}"] = (text, src)
                except llm.LLMError as e:
                    st.error(str(e))
        if f"draft_{sid}" in st.session_state:
            text, src = st.session_state[f"draft_{sid}"]
            edited = st.text_area("Draft (the teacher can edit)", text, height=380)
            if st.checkbox("I am the class teacher and I approve this summary"):
                st.success("Approved. Ready to send to the family.")
                st.download_button("Download approved summary", edited, file_name=f"{sid}_summary.txt")
