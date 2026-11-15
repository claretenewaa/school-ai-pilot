# School AI Pilot: Accra basic schools (Class 1 to JHS 3)

Synthetic data, AI models and a small web app to try the AI features in the pilot plan
(`AI_Pilot_Priorities_and_Timeline.docx`). **Every student, school and document here is fictional.**

## Run it
```bash
cd "school_management_ai_project "
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # once
.venv/bin/python src/generate_data.py    # rebuild the data (about 10 s, fixed random seed)
.venv/bin/python src/train_models.py     # train and evaluate models -> models/
.venv/bin/python src/make_sample_imports.py   # sample import files for one school -> sample_imports/
.venv/bin/streamlit run app.py           # open http://localhost:8501
```
Everything works offline. Paste a Claude API key in the sidebar (or set `ANTHROPIC_API_KEY`) to switch the
question generator, knowledge bot and parent advisor to Claude (`claude-opus-5-5`).

## The data (`data/`)
| File | Rows | What it is |
|---|---|---|
| `student_subject_term.csv` | **50,000** | One row per student, subject and term, 2025/2026. All variables from the brief (A, B, C) plus outcomes |
| `students.csv` | 2,192 | Background: names, gender, date of birth, home language, guardian, occupation, distance, transport |
| `question_responses.csv` | ~187,000 | Per-question marks on end-of-term papers (Maths, English, Science), tagged to topic, difficulty, cognitive level and error type |
| `absence_events.csv` | ~21,000 | Each absence spell with dates and reason |
| `interventions.csv` | ~770 | Interventions with attendance 20 school days before and after |
| `schools.csv` | 8 | 4 public, 4 private (low, mid and high fee) in Adenta, Dansoman, Teshie, Nima, Madina, Lapaz, Kaneshie and East Legon |
| `parent_questions.csv` | 50 | Test questions for the knowledge bot |

How the data is made realistic:
- **Calendar:** the real 2025/26 Ghana term structure, with public holidays (Farmers' Day, Independence Day,
  Eid, Republic Day), mid-term breaks and exam weeks.
- **Attendance:** simulated day by day. Illness spells last several days. There is more malaria in the rainy
  months, Fridays are slightly worse, and flood days in June hit flood-prone areas.
- **Subjects:** the GES Standards-Based Curriculum for Class 1 to 6 and the Common Core Programme for JHS 1 to 3.
  Lessons missed per subject follow a weekly timetable.
- **Scores:** CA out of 30 plus 70% of the exam. Classes 1 to 6 get proficiency levels (HP/P/AP/D/E) and JHS
  gets grades 1 to 9.
- **School gaps:** public schools average about 55 in exams and private schools 65 to 75.
- **Names and languages:** Ghanaian names (Akan day names, Ga, Ewe, Northern) and home languages.
- **Events:** about 9% of students have a disruptive event in Term 2 or 3. Interventions cut absence afterwards.
- **Messiness:** some values are missing, as happens in real school data.

Key columns in `student_subject_term.csv`:
- **A. Academic:** `prev_exam_score`, `class_test_avg`, `quiz_avg`, `homework_avg`, `ca_score_30`,
  `exam_questions_attempted/correct`, `common_error_type`, `avg_test_time_min`
- **B. Attendance:** `days_present/absent`, `attendance_rate`, `late_arrivals`, `max_consecutive_days_absent`,
  `days_absent_before_tests`, `main_absence_reason`, `times_left_early`, `lesson_attendance_rate`,
  `intervention_this_term` (before/after in `interventions.csv`)
- **C. Engagement:** `participation_rating`, `answers_per_week`, `classwork_completion_rate`, `group_work_rating`,
  `help_seeking_rating`, `assignment_submission_rate`, `time_on_task_pct`, `observation_rating`,
  `peer_learning_rating`, `improvement_after_feedback`, `discipline_incidents`
- **Outcomes:** `exam_score_100`, `total_score_100`, `grade`, `at_risk` (exam below 50 or a drop of 15+
  points), `teacher_comment`

## One school per installation
Each installation of the app serves **one school**. Its data lives in one file, `school_data/school.db`, and the
school's documents live in `school_data/docs/`. The synthetic 8-school dataset is used only for the demo and as
extra training data.

1. **🏫 School setup:** enter the school name, term dates, the CA weight (30, 40 or 50) and office contacts, and
   upload the handbook, fee schedule and calendar (.docx, .txt or .md) for the parent bot.
2. **📥 Import data:** the data collector downloads an Excel template, fills it in and uploads it. The app checks
   every row and imports the good ones. Re-importing a record replaces it, so mistakes are fixed by uploading again.

| What | Template | Notes |
|---|---|---|
| Student register | `students` | Import first; every other file is checked against its IDs |
| Attendance | `attendance_register` (one column per day) or `attendance_daily` (a list) | Register template comes pre-filled with the class list and every weekday of the term. P / A / L / E, and ✓ / x also work |
| Scores | `assessments` (a list) or `broadsheet` (one column per subject) | Tests, quizzes, homework (with Yes/No submitted), classwork, exams, last year's exam |
| Marks per question | `question_marks` | Needed for learning profiles and gap analysis |
| Engagement | `engagement` | Teacher ratings once a term |
| Optional | `discipline`, `interventions` | |
| Question bank | `question_bank` | The school's own BECE past questions, textbook items and teacher questions, with their source |

The importer accepts common variations: headers like "Learner ID" or "Out of", classes like "B7", "Basic 7" or
"JHS1", subjects like "Maths", "Int. Science" or "RME", and day-first dates (08/09/2026). It reports each problem
with its row number in plain words, e.g. a score above the maximum, an unknown student ID, a bad date, an
unknown subject or a bad term.

`sample_imports/` holds a full year of files for one fictional school in these layouts. The data collector can
practise with them: the Import page has a button that imports them all. The early-warning page can **retrain**
with the school's own results and reports accuracy on 50 of its own students it never saw.

## Question bank and generator
- `src/ai/bank_data.py`: about 150 questions **written for this pilot in the style of** the BECE and Ghanaian
  textbooks, across Maths, English, Science, Social Studies, RME, Computing, History, OWOP, CAD and Career Tech.
  They are *not* copies of WAEC papers or textbooks: those are copyrighted. Schools import real past questions
  they own through the `question_bank` template, and these are preferred.
- Offline, a set mixes bank questions with about 60 randomised Maths templates (3 to 6 per topic: word
  problems, fill-in-the-blank, spot-the-mistake, multi-part, table reading). Questions are never repeated in a
  set or across "Generate" clicks.
- With Claude, bank questions are given as style examples, and the prompt requires different forms, contexts and
  numbers, avoiding questions already generated.

## The AI features
| Pilot priority | Module | Method |
|---|---|---|
| 1. Question generator | `src/ai/question_generator.py` | Claude with structured JSON output; offline: randomised Ghana-context Maths templates |
| 1. Learning profile and gap analysis | `src/ai/learning_profile.py` | Topic mastery from per-question marks, smoothed towards the class average and weighted to recent terms; errors mapped to practice actions |
| 2. Early-warning model | `src/ai/early_warning.py` | Gradient boosting for risk; logistic regression for plain-language reasons. No demographics or fees used |
| 3. Knowledge bot | `src/ai/knowledge_bot.py` | Retrieval-augmented: TF-IDF over document lines, then Claude answers only from the excerpts and cites them |
| 3. Parent advisor | `src/ai/parent_advisor.py` | Claude turns the learning profile into a plain-language summary (6 languages); a teacher approves it |

Model results are in `models/metrics.json`. All results are on simulated data. As the pilot plan says, accuracy
must be measured again on 50 real students the model never saw.

School documents for the bot are in `school_docs/`. Replace them with the real handbook, fee schedule and calendar.
