"""Generate a realistic synthetic dataset for basic schools in Accra (Class 1 to JHS 3).

Output (data/):
  student_subject_term.csv  exactly 50,000 rows: one per student, subject and term
  students.csv              one row per student (background)
  question_responses.csv    per-question marks (Maths, English, Science) tagged to topics
  absence_events.csv        one row per absence spell, with reason
  interventions.csv         interventions and attendance before/after
  schools.csv               the 8 schools

Everything is simulated. No real student is represented.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from curriculum import (ERROR_TYPES, LEVEL_LABEL, PERIODS_PER_WEEK, SUBJECTS, TOPICS, band,
                        jhs_grade, primary_proficiency)

SEED = 2026
TARGET_ROWS = 50_000
ACADEMIC_YEAR = "2025/2026"
OUT = Path(__file__).resolve().parent.parent / "data"
rng = np.random.default_rng(SEED)

# --------------------------------------------------------------------------- schools
SCHOOLS = pd.DataFrame([
    # id, name, area, type, fee tier, quality, avg class size, local language, flood-prone
    ("SCH01", "Adenta Hilltop Basic School", "Adenta", "Public", "Free", -0.30, 37, "Twi", False),
    ("SCH02", "Dansoman Seaview M/A Basic School", "Dansoman", "Public", "Free", -0.35, 38, "Ga", True),
    ("SCH03", "Teshie Coastal Cluster Basic School", "Teshie", "Public", "Free", -0.45, 39, "Ga", False),
    ("SCH04", "Nima Unity M/A Basic School", "Nima", "Public", "Free", -0.50, 40, "Twi", True),
    ("SCH05", "Madina Bright Future Academy", "Madina", "Private", "Low-fee", 0.00, 24, "Twi", False),
    ("SCH06", "Lapaz Wisdom Gate School", "Lapaz", "Private", "Low-fee", 0.05, 22, "Twi", False),
    ("SCH07", "Kaneshie Grace Preparatory School", "Kaneshie", "Private", "Mid-fee", 0.25, 21, "Ga", True),
    ("SCH08", "East Legon Pinecrest Academy", "East Legon", "Private", "High-fee", 0.55, 18, "Twi", False),
], columns=["school_id", "school_name", "area", "school_type", "fee_tier", "quality",
            "avg_class_size", "ghanaian_language_taught", "flood_prone"])
TERM_FEES = {"Free": 0, "Low-fee": 1450, "Mid-fee": 2600, "High-fee": 7800}  # GHS per term
SCHOOLS["term_fees_ghs"] = SCHOOLS["fee_tier"].map(TERM_FEES)
SCHOOLS["pta_levy_ghs"] = np.where(SCHOOLS["school_type"] == "Public", 60, 0)

# --------------------------------------------------------------------------- calendar
TERMS = {
    1: (dt.date(2025, 9, 9), dt.date(2025, 12, 19)),
    2: (dt.date(2026, 1, 13), dt.date(2026, 4, 2)),
    3: (dt.date(2026, 4, 28), dt.date(2026, 7, 24)),
}
HOLIDAYS = {
    dt.date(2025, 9, 22): "Kwame Nkrumah Memorial Day (observed)",
    dt.date(2025, 10, 23): "Mid-term break", dt.date(2025, 10, 24): "Mid-term break",
    dt.date(2025, 12, 5): "Farmers' Day",
    dt.date(2026, 2, 26): "Mid-term break", dt.date(2026, 2, 27): "Mid-term break",
    dt.date(2026, 3, 6): "Independence Day", dt.date(2026, 3, 20): "Eid al-Fitr",
    dt.date(2026, 5, 1): "May Day", dt.date(2026, 5, 27): "Eid al-Adha",
    dt.date(2026, 6, 11): "Mid-term break", dt.date(2026, 6, 12): "Mid-term break",
    dt.date(2026, 7, 1): "Republic Day",
}
EXAM_START = {1: dt.date(2025, 12, 8), 2: dt.date(2026, 3, 23), 3: dt.date(2026, 7, 13)}
FLOOD_DAYS = [dt.date(2026, 6, 15), dt.date(2026, 6, 16)]  # heavy rains in flood-prone areas


def school_days(term):
    start, end = TERMS[term]
    d, days = start, []
    while d <= end:
        if d.weekday() < 5 and d not in HOLIDAYS:
            days.append(d)
        d += dt.timedelta(days=1)
    return days


DAYS = {t: school_days(t) for t in TERMS}
ALL_DAYS = [d for t in TERMS for d in DAYS[t]]
DAY_TERM = np.array([t for t in TERMS for _ in DAYS[t]])
DAY_INDEX = {d: i for i, d in enumerate(ALL_DAYS)}
N_DAYS = len(ALL_DAYS)
WEEKDAY = np.array([d.weekday() for d in ALL_DAYS])
MONTH = np.array([d.month for d in ALL_DAYS])


def weekly_timetable(b):
    """Spread each subject's weekly periods over Mon-Fri. Returns {subject: [periods Mon..Fri]}."""
    tt = {}
    for k, (subj, n) in enumerate(PERIODS_PER_WEEK[b].items()):
        per_day = [n // 5] * 5
        for j in range(n % 5):
            per_day[(k + j * 2) % 5] += 1
        tt[subj] = per_day
    return tt


TIMETABLE = {b: weekly_timetable(b) for b in ("lower", "upper", "jhs")}

# --------------------------------------------------------------------------- names
AKAN_DAY = {  # weekday born: (male, female)
    0: ("Kwadwo", "Adwoa"), 1: ("Kwabena", "Abena"), 2: ("Kwaku", "Akua"), 3: ("Yaw", "Yaa"),
    4: ("Kofi", "Afua"), 5: ("Kwame", "Ama"), 6: ("Kwesi", "Akosua"),
}
CHRISTIAN_M = ["Emmanuel", "Prince", "Michael", "Daniel", "Joseph", "Samuel", "Godfred", "Ebenezer", "Isaac",
               "Richard", "Kelvin", "Bernard", "Elijah", "Nathaniel", "Caleb", "Jeffrey", "Desmond", "Bright",
               "Justice", "Clement", "Felix", "Jesse", "Benedict", "Collins", "David", "Francis", "Kingsley"]
CHRISTIAN_F = ["Gifty", "Blessing", "Precious", "Esther", "Priscilla", "Comfort", "Mercy", "Felicia",
               "Jennifer", "Grace", "Rita", "Vida", "Linda", "Patience", "Deborah", "Rebecca", "Mavis", "Nhyira",
               "Princess", "Abigail", "Joyce", "Josephine", "Benedicta", "Christabel", "Hannah", "Elizabeth",
               "Getrude", "Philomena", "Belinda", "Doreen"]
SURNAMES = {
    "Akan": ["Mensah", "Owusu", "Asante", "Boateng", "Osei", "Appiah", "Agyeman", "Ofori", "Darko", "Amoah",
             "Antwi", "Frimpong", "Sarpong", "Opoku", "Gyamfi", "Acheampong", "Asamoah", "Danquah", "Bonsu",
             "Ansah", "Adomako", "Kyei", "Nkansah", "Ampofo", "Yeboah", "Oppong", "Kusi", "Badu", "Amponsah",
             "Fosu", "Asiedu", "Owusu-Ansah", "Takyi", "Quaicoe", "Arthur", "Essien", "Eshun", "Baidoo"],
    "Ga": ["Tetteh", "Lamptey", "Quaye", "Okine", "Ashong", "Laryea", "Nortey", "Aryee", "Kotey", "Addo",
           "Quarshie", "Lartey", "Ankrah", "Sackey", "Odoi", "Tagoe", "Annan", "Bortey", "Amartey",
           "Mensah-Bonsu", "Allotey", "Adjetey", "Lomotey", "Nikoi", "Clottey", "Ayitey"],
    "Ewe": ["Agbeko", "Ahiable", "Amegah", "Dzokoto", "Kpodo", "Agbenyega", "Gbedemah", "Adzraku", "Kudjawu",
            "Fiadzo", "Tsikata", "Atsu", "Dogbe", "Ametefe", "Avornyo", "Kumordzi", "Segbefia", "Akpabli"],
    "Northern": ["Abdul-Rahman", "Iddrisu", "Mahama", "Fuseini", "Alhassan", "Salifu", "Zakaria", "Abubakar",
                 "Mohammed", "Yakubu", "Issahaku", "Sulemana", "Seidu", "Haruna", "Osman", "Abdulai", "Musah"],
}
LOCAL_FIRST = {
    "Ga": (["Nii", "Nii Armah", "Nii Okai", "Tetteh", "Odarkwei", "Ayi", "Kwei", "Lartey"],
           ["Naa", "Naa Adjeley", "Naa Odarkor", "Ayele", "Dede", "Korkoi", "Naa Ayeley", "Akuorkor"]),
    "Ewe": (["Selorm", "Elikem", "Edem", "Senyo", "Mawuli", "Delali", "Kekeli", "Etornam", "Dela"],
            ["Dzifa", "Kafui", "Enyonam", "Esinam", "Selasi", "Akpene", "Dzigbordi", "Elorm", "Afi"]),
    "Northern": (["Abdul", "Rashid", "Hamza", "Ibrahim", "Yusif", "Mustapha", "Shaibu", "Fatawu", "Nuhu"],
                 ["Fatima", "Mariam", "Ayisha", "Hawa", "Zuweira", "Rahinatu", "Sadia", "Amina", "Salamatu"]),
}
FOREIGN = [("Nigerian", ["Chukwuemeka", "Tunde", "Ifeanyi", "Chidinma", "Adaeze", "Temitope"],
            ["Okafor", "Adeyemi", "Eze", "Balogun", "Nwosu"]),
           ("Togolese", ["Komlan", "Kossi", "Afi", "Ablavi"], ["Agbodjan", "Kponton", "Lawson"])]

OCCUPATIONS = [  # name, socio-economic index, weight
    ("Trader", -0.2, 22), ("Artisan (mason/carpenter/mechanic)", -0.3, 12), ("Driver (trotro/taxi/ride-hailing)", -0.3, 9),
    ("Seamstress/Tailor", -0.25, 6), ("Hairdresser/Barber", -0.25, 5), ("Fisherman/Fishmonger", -0.5, 3),
    ("Civil servant", 0.3, 8), ("Teacher", 0.3, 6), ("Nurse/Health worker", 0.35, 4), ("Police/Military", 0.2, 3),
    ("Small business owner", 0.1, 8), ("Banker/Accountant", 0.7, 3), ("Engineer/IT professional", 0.8, 3),
    ("Security guard", -0.4, 3), ("Unemployed", -0.7, 3), ("Okada/delivery rider", -0.45, 2),
]
OCC_NAMES = [o[0] for o in OCCUPATIONS]
OCC_SES = {o[0]: o[1] for o in OCCUPATIONS}
OCC_W = np.array([o[2] for o in OCCUPATIONS], dtype=float)


def pick(seq, p=None):
    return seq[rng.choice(len(seq), p=p)]


# --------------------------------------------------------------------------- students
def make_students():
    rows = []
    for _, s in SCHOOLS.iterrows():
        counter = 0
        for level in range(1, 10):
            n = max(10, int(round(rng.normal(s.avg_class_size, s.avg_class_size * 0.12))))
            if level >= 7 and s.school_type == "Public":
                n = int(n * 1.05)  # JHS classes slightly larger in public schools
            for _ in range(n):
                counter += 1
                rows.append(make_student(s, level, counter))
    return pd.DataFrame(rows)


def make_student(s, level, counter):
    gender = pick(["Male", "Female"])
    # Ethnic background shapes names and home language (Accra is very mixed)
    eth_p = {"Nima": [0.30, 0.15, 0.10, 0.45], "Teshie": [0.25, 0.55, 0.12, 0.08],
             "Dansoman": [0.35, 0.40, 0.15, 0.10], "Kaneshie": [0.38, 0.35, 0.17, 0.10]}.get(
        s.area, [0.50, 0.20, 0.18, 0.12])
    eth = pick(["Akan", "Ga", "Ewe", "Northern"], eth_p)
    foreign = rng.random() < (0.035 if s.school_type == "Private" else 0.015)
    birth_year = 2025 - (level + 5) - (1 if rng.random() < (0.18 if s.school_type == "Public" else 0.07) else 0)
    dob = dt.date(birth_year, 1, 1) + dt.timedelta(days=int(rng.integers(0, 365)))
    if dob > dt.date(birth_year, 9, 1):
        dob -= dt.timedelta(days=int(rng.integers(30, 200)))
    male = gender == "Male"
    if foreign:
        nat, firsts, surs = FOREIGN[0] if rng.random() < 0.7 else FOREIGN[1]
        first, surname, home_lang, nationality = pick(firsts), pick(surs), "English", nat
    else:
        nationality = "Ghanaian"
        surname = pick(SURNAMES[eth])
        r = rng.random()
        if eth == "Akan" and r < 0.45:
            day_name = AKAN_DAY[dob.weekday()][0 if male else 1]
            first = day_name if rng.random() < 0.5 else f"{pick(CHRISTIAN_M if male else CHRISTIAN_F)} {day_name}"
        elif eth in LOCAL_FIRST and r < (0.85 if eth == "Northern" else 0.45):
            first = pick(LOCAL_FIRST[eth][0 if male else 1])
        else:
            first = pick(CHRISTIAN_M if male else CHRISTIAN_F)
        if eth == "Northern" and not any(first.startswith(x) for x in LOCAL_FIRST["Northern"][0 + (0 if male else 1)]):
            first = pick(LOCAL_FIRST["Northern"][0 if male else 1])
        home_lang = {"Akan": "Twi", "Ga": "Ga", "Ewe": "Ewe", "Northern": pick(["Hausa", "Dagbani"])}[eth]
        if s.fee_tier in ("Mid-fee", "High-fee") and rng.random() < 0.45:
            home_lang = "English"
        elif rng.random() < 0.10:
            home_lang = "English"
    religion = "Muslim" if eth == "Northern" and rng.random() < 0.9 else pick(
        ["Christian", "Muslim", "Traditional/Other"], [0.86, 0.11, 0.03])

    # Socio-economic background, tilted by school fee tier
    tier_shift = {"Free": -0.15, "Low-fee": 0.0, "Mid-fee": 0.3, "High-fee": 0.7}[s.fee_tier]
    w = OCC_W.copy()
    for i, o in enumerate(OCCUPATIONS):
        w[i] *= np.exp(4.0 * tier_shift * o[1])
    occ = OCC_NAMES[rng.choice(len(OCC_NAMES), p=w / w.sum())]
    ses = float(np.clip(OCC_SES[occ] + tier_shift + rng.normal(0, 0.2), -1.2, 1.5))
    guardian = pick(["Both parents", "Mother only", "Father only", "Grandparent", "Other relative"],
                    [0.58, 0.22, 0.06, 0.08, 0.06] if ses < 0.3 else [0.75, 0.13, 0.05, 0.04, 0.03])
    distance = float(np.round(np.clip(rng.gamma(2.0, 1.1 if s.school_type == "Public" else 2.4), 0.2, 18), 1))
    if distance < 1.2:
        transport = "Walk"
    elif s.fee_tier == "High-fee":
        transport = pick(["Private car", "School bus", "Ride-hailing"], [0.6, 0.32, 0.08])
    elif s.school_type == "Private":
        transport = pick(["School bus", "Trotro", "Walk", "Private car"], [0.4, 0.3, 0.15, 0.15])
    else:
        transport = pick(["Walk", "Trotro", "Okada/Taxi"], [0.45, 0.45, 0.10])
    device = rng.random() < np.clip(0.35 + 0.35 * ses, 0.05, 0.98)
    internet = device and rng.random() < np.clip(0.55 + 0.3 * ses, 0.1, 0.98)
    special = pick(["None", "Visual impairment", "Hearing impairment", "Learning difficulty"],
                   [0.965, 0.01, 0.007, 0.018])
    enrolled = int(np.clip(2025 - rng.integers(0, level + 2), 2016, 2025))

    return dict(
        student_id=f"{s.school_id}-{counter:04d}", school_id=s.school_id, first_name=first, surname=surname,
        gender=gender, date_of_birth=dob.isoformat(), age_at_sept_2025=(dt.date(2025, 9, 1) - dob).days // 365,
        class_level=level, class_label=LEVEL_LABEL[level], nationality=nationality, home_language=home_lang,
        religion=religion, guardian_type=guardian, guardian_occupation=occ,
        household_size=int(np.clip(rng.poisson(5.2 - 0.8 * ses), 2, 12)), distance_to_school_km=distance,
        transport_mode=transport, has_device_at_home=device, has_internet_at_home=internet,
        special_needs=special, school_feeding_programme=(s.school_type == "Public" and level <= 6),
        year_enrolled=enrolled, exempt_ghanaian_language=foreign,
        # latent traits (dropped before saving)
        _ses=ses, _eth=eth,
    )


# --------------------------------------------------------------------------- daily attendance simulation
ABSENCE_REASONS = ["Illness (malaria/fever)", "Illness (stomach/cold)", "Hospital appointment",
                   "Family funeral/function", "Travelled to hometown", "Sent home for fees",
                   "Helping at family business", "Transport problem", "Flooding/heavy rain",
                   "Religious festival", "No reason given"]


def simulate_attendance(st):
    """Simulate the whole year day by day for every student.

    Returns absent[N, D], late[N, D], left_early[N, D], reason_code[N, D] (-1 if present)
    and records of spells/interventions.
    """
    n = len(st)
    sch = st.merge(SCHOOLS, on="school_id", how="left")
    public = (sch.school_type == "Public").to_numpy()
    ses = st._ses.to_numpy()
    chronic = rng.random(n) < np.where(public, 0.08, 0.04)
    base = np.clip(0.028 + 0.015 * public - 0.012 * ses + chronic * rng.uniform(0.07, 0.17, n)
                   + rng.normal(0, 0.01, n), 0.006, 0.35)
    # A minority suffer a disruptive event (family crisis, illness, relocation) in term 2 or 3
    shock_term = np.where(rng.random(n) < 0.09, rng.choice([2, 3], n), 0)
    st["_chronic"] = chronic
    st["_shock_term"] = shock_term
    st["_base_abs"] = base

    # Joined late (transfer in) / left early (transfer out) decided later; simulate all days here
    absent = np.zeros((n, N_DAYS), dtype=bool)
    late = np.zeros((n, N_DAYS), dtype=bool)
    left = np.zeros((n, N_DAYS), dtype=bool)
    reason = np.full((n, N_DAYS), -1, dtype=np.int8)
    dist = st.distance_to_school_km.to_numpy()
    trotro = st.transport_mode.isin(["Trotro", "Okada/Taxi"]).to_numpy()
    p_late = np.clip(0.02 + 0.006 * dist + 0.04 * trotro + 0.02 * public + rng.normal(0, 0.01, n), 0.003, 0.3)
    p_left = np.clip(0.008 + 0.01 * chronic + rng.normal(0, 0.004, n), 0.0, 0.08)
    fee_payer = (~public)
    fee_risk = np.clip(0.4 - 0.5 * ses, 0, 1) * fee_payer
    flood_prone = sch.flood_prone.to_numpy()
    muslim = (st.religion == "Muslim").to_numpy()
    low_ses = ses < -0.3

    # Interventions: after term 1 and term 2, decided from previous-term attendance (filled in below)
    intervention_day = np.full(n, -1)

    prev_abs = np.zeros(n, dtype=bool)
    cur_reason = np.full(n, -1, dtype=np.int8)
    term_start_idx = {t: DAY_INDEX[DAYS[t][0]] for t in TERMS}
    rates = base.copy()
    for d in range(N_DAYS):
        t = DAY_TERM[d]
        if d in term_start_idx.values() and t > 1:
            # Decide interventions using attendance so far in the previous term
            prev_mask = DAY_TERM[:d] == t - 1
            prev_rate = 1 - absent[:, :d][:, prev_mask].mean(axis=1)
            flagged = prev_rate < 0.90
            chosen = flagged & (rng.random(n) < 0.45) & (intervention_day < 0)
            offs = rng.integers(5, 25, n)
            intervention_day = np.where(chosen, d + offs, intervention_day)
        r = base.copy()
        r = np.where(shock_term == t, r + 0.09, r)
        r = np.where(shock_term == t - 1, r + 0.03, r)  # partial recovery the term after
        r = np.where((intervention_day >= 0) & (d >= intervention_day), r * 0.6, r)
        if MONTH[d] in (5, 6, 10):  # rainy season: more malaria
            r = r * 1.25
        if WEEKDAY[d] == 4:  # Fridays slightly worse
            r = r * 1.15
        if ALL_DAYS[d] in FLOOD_DAYS:
            r = np.where(flood_prone, np.maximum(r, 0.45), r)
        # Markov persistence: illness spells last a few days
        p_abs = np.where(prev_abs, 0.48, r)
        a = rng.random(n) < p_abs
        start = a & ~prev_abs
        # Assign reason at spell start
        if start.any():
            idx = np.where(start)[0]
            for i in idx:
                if ALL_DAYS[d] in FLOOD_DAYS and flood_prone[i]:
                    code = 8
                else:
                    w = np.array([30, 12, 5, 9, 6, 9 * fee_risk[i] * (1 + (d - term_start_idx[t] > 20)),
                                  6 * low_ses[i], 4 + 3 * trotro[i], 2 * (MONTH[d] in (5, 6)),
                                  (4 if muslim[i] and ALL_DAYS[d].month in (3, 5) else 0.3), 18], dtype=float)
                    code = rng.choice(len(w), p=w / w.sum())
                cur_reason[i] = code
        reason[a, d] = cur_reason[a]
        absent[:, d] = a
        prev_abs = a
        present = ~a
        late[:, d] = present & (rng.random(n) < p_late * (1.4 if MONTH[d] in (5, 6) else 1.0))
        left[:, d] = present & (rng.random(n) < p_left)
        # term boundaries break spells
        if d + 1 < N_DAYS and DAY_TERM[d + 1] != t:
            prev_abs[:] = False
    st["_intervention_day"] = intervention_day
    return absent, late, left, reason


# --------------------------------------------------------------------------- main build
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


SUBJECT_DIFFICULTY = {"Mathematics": -0.38, "Science": -0.18, "English Language": -0.05, "Computing": -0.15,
                      "Social Studies": 0.05, "History": 0.0, "Our World Our People": 0.15,
                      "Religious & Moral Education": 0.30, "Creative Arts": 0.32, "Creative Arts & Design": 0.18,
                      "Career Technology": 0.10, "Ghanaian Language": 0.0}

COMMENTS = [
    (80, ["Excellent performance. Keep it up!", "Outstanding work. Maintain this standard.",
          "Very impressive. Keep shining."]),
    (70, ["Very good. Aim higher.", "Good performance. Keep working hard.", "Very good work. Well done."]),
    (60, ["Good. There is room for improvement.", "Credit performance. Can do better with more practice.",
          "Good effort. Keep pushing."]),
    (50, ["Average performance. Needs to put in more effort.", "Fair. Must learn harder.",
          "Can do better. More practice needed."]),
    (40, ["Below average. Needs serious improvement.", "Must sit up and take studies seriously.",
          "Needs to work harder and ask questions in class."]),
    (0, ["Weak performance. Needs extra support and close supervision.", "Poor. Must attend extra classes.",
         "Needs urgent help in this subject. Parents should support at home."]),
]


def teacher_comment(total, att_rate, eng):
    for cut, opts in COMMENTS:
        if total >= cut:
            c = pick(opts)
            break
    if att_rate < 0.85:
        c += " Irregular attendance is affecting performance."
    elif eng > 0.9 and rng.random() < 0.5:
        c += " Participates actively in class."
    elif eng < -0.9 and rng.random() < 0.5:
        c += " Must pay attention in class."
    return c


def assign_error_type(z, speed, eng, english_skill, subject):
    w = np.array([
        2.5 * sigmoid(-1.5 * z),                                      # conceptual
        (1.6 if subject in ("Mathematics", "Science") else 0.4) * sigmoid(-0.5 * z),  # procedural
        1.6 * sigmoid(3 * (0.75 - speed)) * sigmoid(z),                # careless (fast, able)
        0.8 * sigmoid(-1.2 * english_skill),                           # misread
        0.9 * sigmoid(-1.2 * eng),                                     # incomplete
        0.7 * sigmoid(-2.0 * eng - 0.8 * z - 1),                       # no attempt
        (1.2 if subject in ("English Language", "Ghanaian Language") else 0.5) * sigmoid(-1.3 * english_skill),
    ])
    return ERROR_TYPES[rng.choice(len(w), p=w / w.sum())]


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    st = make_students()
    n = len(st)
    sch = st.merge(SCHOOLS, on="school_id", how="left")
    print(f"students: {n}")

    # Latent traits
    ses = st._ses.to_numpy()
    quality = sch.quality.to_numpy()
    ability = 0.85 * rng.normal(0, 1, n) + 0.28 * ses + 0.75 * quality + 0.1
    sn = st.special_needs.to_numpy()
    ability = ability - 0.6 * (sn == "Learning difficulty") - 0.2 * np.isin(sn, ["Visual impairment", "Hearing impairment"])
    engagement = 0.45 * ability + 0.85 * rng.normal(0, 1, n)
    engagement = (engagement - engagement.mean()) / engagement.std()
    english_skill = ability + 0.25 * (st.home_language == "English") + rng.normal(0, 0.4, n)
    careless = rng.random(n) < 0.12          # fast, makes slips
    skipper = (st.class_level >= 7).to_numpy() & (rng.random(n) < 0.05)  # skips a specific subject
    term_trend = rng.normal(0, 0.12, (n, 4))
    st["_ability"] = ability

    absent, late, left, reason = simulate_attendance(st)
    shock_term = st._shock_term.to_numpy()
    int_day = st._intervention_day.to_numpy()

    # Enrolment changes: some join in term 2, some leave after term 2
    joined_t2 = rng.random(n) < 0.025
    left_after_t2 = (~joined_t2) & (rng.random(n) < 0.02 + 0.03 * (shock_term == 2))
    st["status"] = np.where(joined_t2, "Transferred in (Term 2)",
                            np.where(left_after_t2, "Transferred out (after Term 2)", "Active"))

    # Each student has weak and strong topics in the three core subjects
    weak_topic = {}
    for i in range(n):
        b = band(int(st.class_level.iat[i]))
        for subj in TOPICS:
            topics = [t[1] for t in TOPICS[subj][b]]
            effects = rng.normal(0, 0.25, len(topics))
            k = rng.choice([1, 2, 3], p=[0.45, 0.4, 0.15])
            for j in rng.choice(len(topics), k, replace=False):
                effects[j] -= rng.uniform(0.7, 1.4)
            weak_topic[(i, subj)] = dict(zip(topics, effects))

    subj_aff = {}
    rows, q_rows, spells, interventions = [], [], [], []
    disc_rate = 0.22 * np.exp(-0.55 * engagement + 0.35 * (st.class_level.to_numpy() >= 7))

    for i in range(n):
        lvl = int(st.class_level.iat[i])
        b = band(lvl)
        sid = st.student_id.iat[i]
        school = sch.iloc[i]
        subjects = [s for s in SUBJECTS[b] if not (s == "Ghanaian Language" and st.exempt_ghanaian_language.iat[i])]
        skip_subj = pick(["Mathematics", "Science", "Ghanaian Language", "Career Technology"]) if skipper[i] else None
        terms = [t for t in (1, 2, 3) if not (t == 1 and joined_t2[i]) and not (t == 3 and left_after_t2[i])]
        prev_exam = {}
        for subj in subjects:
            subj_aff[(i, subj)] = rng.normal(0, 0.35)
            # last year's term 3 exam score (none for Class 1: they came from KG)
            if lvl > 1 and not joined_t2[i]:
                z0 = ability[i] + subj_aff[(i, subj)] + SUBJECT_DIFFICULTY[subj] - term_trend[i, 0]
                prev_exam[subj] = float(np.clip(np.round(62 + {"lower": 6, "upper": 0, "jhs": -3}[b] + 15 * z0 + rng.normal(0, 6)), 4, 99))
            elif joined_t2[i]:
                z0 = ability[i] + subj_aff[(i, subj)] + SUBJECT_DIFFICULTY[subj]
                prev_exam[subj] = float(np.clip(np.round(57 + 17 * z0 + rng.normal(0, 10)), 4, 99)) if rng.random() < 0.6 else np.nan
            else:
                prev_exam[subj] = np.nan

        cum_trend = 0.0
        for t in terms:
            cum_trend += term_trend[i, t]
            mask = DAY_TERM == t
            day_idx = np.where(mask)[0]
            a_t = absent[i, mask]
            n_days = int(mask.sum())
            days_absent = int(a_t.sum())
            att_rate = 1 - days_absent / n_days
            # longest run of consecutive absences
            runs, cur = 0, 0
            for v in a_t:
                cur = cur + 1 if v else 0
                runs = max(runs, cur)
            # absence spells -> event log
            j = 0
            while j < n_days:
                if a_t[j]:
                    k = j
                    while k + 1 < n_days and a_t[k + 1]:
                        k += 1
                    spells.append(dict(student_id=sid, term=t, start_date=ALL_DAYS[day_idx[j]].isoformat(),
                                       end_date=ALL_DAYS[day_idx[k]].isoformat(), school_days=k - j + 1,
                                       reason=ABSENCE_REASONS[reason[i, day_idx[j]]]))
                    j = k + 1
                else:
                    j += 1
            r_codes = reason[i, mask][a_t]
            main_reason = ABSENCE_REASONS[np.bincount(r_codes, minlength=len(ABSENCE_REASONS)).argmax()] if days_absent else "None"
            late_n = int(late[i, mask].sum())
            left_n = int(left[i, mask].sum())
            # intervention window
            got_int = int_day[i] >= 0 and DAY_TERM[min(int_day[i], N_DAYS - 1)] == t
            if got_int:
                d0 = int_day[i]
                before = 1 - absent[i, max(0, d0 - 20):d0].mean()
                after = 1 - absent[i, d0:min(N_DAYS, d0 + 20)].mean()
                itype = pick(["Parent-teacher meeting", "Remedial classes", "Counselling", "Peer tutoring",
                              "Home visit", "Attendance contract"], [0.3, 0.25, 0.15, 0.12, 0.1, 0.08])
                interventions.append(dict(student_id=sid, term=t, intervention_date=ALL_DAYS[d0].isoformat(),
                                          intervention_type=itype,
                                          reason_flagged="Low attendance in previous term",
                                          attendance_rate_20d_before=round(before, 3),
                                          attendance_rate_20d_after=round(after, 3)))
            int_effect = 0.15 if (int_day[i] >= 0 and DAY_TERM[min(int_day[i], N_DAYS - 1)] <= t) else 0.0
            shock = -0.55 if shock_term[i] == t else (-0.25 if shock_term[i] and shock_term[i] < t else 0.0)
            eng_t = engagement[i] + rng.normal(0, 0.25) + (0.8 * shock) + (0.2 if int_effect else 0)
            fees_paid = 100.0
            if school.school_type == "Private":
                fees_paid = float(np.clip(np.round(100 * (1 - rng.beta(1.2, 6) * (1.6 - ses[i]) * (1.5 if shock else 1)), 0), 0, 100))
            discipline = int(rng.poisson(disc_rate[i] * (1.6 if shock else 1)))

            # assessment dates for class tests (weeks 3, 6 and 10) and the end-of-term exam
            test_days = [day_idx[min(len(day_idx) - 1, w)] for w in (12, 27, 45)] + [DAY_INDEX[EXAM_START[t]]]
            before_tests = int(sum(absent[i, max(0, td - 2):td].sum() for td in test_days))

            for subj in subjects:
                per_day = np.array(TIMETABLE[b][subj])
                sched = int(per_day[WEEKDAY[mask]].sum())
                missed = int(per_day[WEEKDAY[mask]][a_t].sum())
                if subj == skip_subj:
                    present_lessons = np.where(~a_t)[0]
                    extra = per_day[WEEKDAY[mask]][present_lessons]
                    missed += int((extra * (rng.random(len(extra)) < 0.35)).sum())
                missed = min(missed, sched)
                lesson_att = 1 - missed / sched
                z = (ability[i] + subj_aff[(i, subj)] + SUBJECT_DIFFICULTY[subj] + cum_trend + shock + int_effect
                     + (0.25 if subj == "Ghanaian Language" and st.home_language.iat[i] == school.ghanaian_language_taught else 0)
                     - (0.25 if subj == "Computing" and school.school_type == "Public" else 0))
                level_adj = {"lower": 6, "upper": 0, "jhs": -3}[b]
                mu = 66 + level_adj + 15 * z + 3.2 * eng_t - 55 * max(0, (1 - lesson_att) - 0.04) - 1.5 * before_tests
                exam = float(np.clip(np.round(mu + rng.normal(0, 6)), 2, 99))
                test_avg = float(np.clip(np.round(mu + 4 + rng.normal(0, 8.5)), 0, 100))
                quiz_avg = float(np.clip(np.round(mu + 6 + rng.normal(0, 10)), 0, 100))
                hw_given = int(rng.integers(6, 11))
                sub_rate = float(np.clip(0.86 + 0.07 * eng_t - 1.2 * (1 - att_rate) + rng.normal(0, 0.07), 0.15, 1))
                hw_done = int(round(hw_given * sub_rate))
                hw_avg = float(np.clip(np.round(mu + 14 + rng.normal(0, 9)), 5, 100)) if hw_done else np.nan
                cw_rate = float(np.clip(0.82 + 0.08 * eng_t + 0.04 * z + rng.normal(0, 0.07), 0.2, 1))
                ca = 30 * (0.40 * test_avg + 0.20 * quiz_avg + 0.25 * (hw_avg if hw_done else 0) * (hw_done / hw_given)
                           + 0.15 * cw_rate * 100) / 100
                ca = float(np.round(np.clip(ca, 0, 30), 1))
                total = float(np.round(ca + 0.7 * exam, 1))
                n_q = {"lower": 20, "upper": 30, "jhs": 40}[b]
                attempted = int(np.clip(n_q - rng.poisson(max(0.1, 1.2 - 1.0 * eng_t)), n_q // 2, n_q))
                correct = int(np.clip(rng.binomial(attempted, np.clip(exam / 100 * n_q / attempted, 0.02, 0.99)), 0, attempted))
                speed = float(np.clip(0.8 - 0.05 * z + (-0.25 if careless[i] else 0) + rng.normal(0, 0.1), 0.35, 1.1))
                allowed = 20 if b == "lower" else 30
                err = assign_error_type(z, speed, eng_t, english_skill[i], subj)
                rating = lambda base_: int(np.clip(np.round(3 + base_ + rng.normal(0, 0.6)), 1, 5))
                prev = prev_exam[subj]
                at_risk = int(exam < 50 or (not np.isnan(prev) and prev - exam >= 15))
                if b == "jhs":
                    g, gdesc = jhs_grade(total)
                    grade = str(g)
                else:
                    grade, gdesc = primary_proficiency(total)
                rows.append(dict(
                    record_id=None, student_id=sid, school_id=school.school_id, academic_year=ACADEMIC_YEAR,
                    term=t, class_level=lvl, class_label=LEVEL_LABEL[lvl], subject=subj,
                    teacher_id=f"{school.school_id}-T{lvl:02d}" if b != "jhs" else
                    f"{school.school_id}-T{lvl:02d}{SUBJECTS[b].index(subj):02d}",
                    # A. academic
                    prev_exam_score=prev, class_test_avg=test_avg, quiz_avg=quiz_avg,
                    homework_given=hw_given, homework_submitted=hw_done, homework_avg=hw_avg,
                    ca_score_30=ca, exam_questions_total=n_q, exam_questions_attempted=attempted,
                    exam_questions_correct=correct, common_error_type=err,
                    avg_test_time_min=float(np.round(allowed * speed, 1)), test_time_allowed_min=allowed,
                    # B. attendance (term-level for the student)
                    school_days_in_term=n_days, days_present=n_days - days_absent, days_absent=days_absent,
                    attendance_rate=round(att_rate, 3), late_arrivals=late_n, max_consecutive_days_absent=runs,
                    days_absent_before_tests=before_tests, main_absence_reason=main_reason,
                    times_left_early=left_n,
                    # B. attendance (subject-level)
                    lessons_scheduled=sched, lessons_attended=sched - missed, lesson_attendance_rate=round(lesson_att, 3),
                    # intervention
                    intervention_this_term=got_int,
                    # C. engagement and behaviour
                    participation_rating=rating(0.95 * eng_t), answers_per_week=int(rng.poisson(np.exp(0.7 + 0.45 * eng_t))),
                    classwork_completion_rate=round(cw_rate, 3), group_work_rating=rating(0.8 * eng_t),
                    help_seeking_rating=rating(0.45 * eng_t - 0.3 * z),
                    assignment_submission_rate=round(hw_done / hw_given, 3),
                    time_on_task_pct=float(np.clip(np.round(73 + 9 * eng_t + 3 * z + rng.normal(0, 6)), 20, 99)),
                    observation_rating=rating(0.7 * eng_t + 0.3 * z), peer_learning_rating=rating(0.75 * eng_t),
                    improvement_after_feedback=float(np.round(np.clip(rng.normal(4 + 2.5 * eng_t, 3.5), -6, 25), 1)),
                    discipline_incidents=discipline, fees_paid_pct=fees_paid,
                    # outcomes
                    exam_score_100=exam, total_score_100=total, grade=grade, grade_description=gdesc,
                    at_risk=at_risk, teacher_comment=teacher_comment(total, att_rate, eng_t),
                ))
                # per-question responses on the end-of-term paper (core subjects only)
                if subj in TOPICS:
                    q_rows.extend(question_paper(i, sid, school.school_id, lvl, t, subj, z - SUBJECT_DIFFICULTY[subj],
                                                 eng_t, careless[i], english_skill[i], weak_topic[(i, subj)]))
                prev_exam[subj] = exam

    df = pd.DataFrame(rows)
    # Trim to exactly TARGET_ROWS: remove Term-3 blocks of a few more students who transferred out
    excess = len(df) - TARGET_ROWS
    print(f"rows before trim: {len(df)}")
    if excess < 0:
        raise SystemExit("Not enough rows - increase class sizes")
    candidates = st.loc[st.status == "Active", "student_id"].sample(frac=1, random_state=SEED).tolist()
    drop_ids = []
    for c in candidates:
        if excess <= 0:
            break
        k = int(((df.student_id == c) & (df.term == 3)).sum())
        if k <= excess:
            drop_ids.append(c)
            excess -= k
    df = df[~((df.student_id.isin(drop_ids)) & (df.term == 3))]
    st.loc[st.student_id.isin(drop_ids), "status"] = "Transferred out (after Term 2)"
    if excess > 0:  # a handful of records lost, as happens with real paper registers
        df = df.drop(df.sample(excess, random_state=SEED).index)
    df = df.reset_index(drop=True)
    df["record_id"] = [f"R{k:06d}" for k in range(1, len(df) + 1)]
    # Realistic missing values: some homework marks and test averages not typed up
    for col, frac in (("homework_avg", 0.02), ("quiz_avg", 0.015), ("time_on_task_pct", 0.03)):
        df.loc[df.sample(frac=frac, random_state=hash(col) % 1000).index, col] = np.nan

    keep_ids = set(df.student_id)
    qr = pd.DataFrame(q_rows)
    qr = qr.merge(df[["student_id", "term", "subject"]], on=["student_id", "term", "subject"])
    ab = pd.DataFrame(spells)
    ab = ab.merge(df[["student_id", "term"]].drop_duplicates(), on=["student_id", "term"])
    iv = pd.DataFrame(interventions)
    iv = iv.merge(df[["student_id", "term"]].drop_duplicates(), on=["student_id", "term"])

    st_out = st[st.student_id.isin(keep_ids)].drop(columns=[c for c in st.columns if c.startswith("_")])
    st_out.to_csv(OUT / "students.csv", index=False)
    df.to_csv(OUT / "student_subject_term.csv", index=False)
    qr.to_csv(OUT / "question_responses.csv", index=False)
    ab.to_csv(OUT / "absence_events.csv", index=False)
    iv.to_csv(OUT / "interventions.csv", index=False)
    SCHOOLS.drop(columns=["quality"]).to_csv(OUT / "schools.csv", index=False)
    print(f"student_subject_term: {len(df):,} rows | students: {len(st_out):,} | question_responses: {len(qr):,} "
          f"| absence_events: {len(ab):,} | interventions: {len(iv):,}")


DIFF = {"Easy": -0.9, "Medium": 0.0, "Hard": 0.9}
MARKS = {"Easy": 2, "Medium": 3, "Hard": 5}
_PAPERS = {}


def paper_for(school_id, lvl, t, subj):
    """Same 10-question section-B paper for everyone in a class (school, level, term, subject)."""
    key = (school_id, lvl, t, subj)
    if key not in _PAPERS:
        topics = [tp for tp in TOPICS[subj][band(lvl)] if tp[2] == t]
        qs = []
        for q in range(10):
            strand, topic, _ = topics[q % len(topics)]
            diff = pick(["Easy", "Medium", "Hard"], [0.35, 0.4, 0.25])
            bloom = pick(["Recall", "Understanding", "Application", "Reasoning"],
                         [0.25, 0.25, 0.3, 0.2] if lvl > 3 else [0.4, 0.35, 0.2, 0.05])
            qs.append((f"{school_id}-B{lvl}-T{t}-{subj[:3].upper()}-Q{q + 1:02d}", q + 1, strand, topic, diff, bloom))
        _PAPERS[key] = qs
    return _PAPERS[key]


def question_paper(i, sid, school_id, lvl, t, subj, z, eng, careless, eng_skill, topic_eff):
    out = []
    for qid, qno, strand, topic, diff, bloom in paper_for(school_id, lvl, t, subj):
        bloom_pen = {"Recall": 0.0, "Understanding": 0.1, "Application": 0.25, "Reasoning": 0.45}[bloom]
        p = sigmoid(1.4 * (z + topic_eff[topic] - DIFF[diff] - bloom_pen) + 0.8)
        mx = MARKS[diff]
        no_attempt = rng.random() < sigmoid(-3.3 - 1.2 * eng + 0.8 * (diff == "Hard"))
        if no_attempt:
            marks, err = 0, "No attempt"
        elif rng.random() < p:
            marks, err = mx, ""
        else:
            marks = int(rng.binomial(mx - 1, 0.35 + 0.3 * p))
            w = np.array([2.2 * sigmoid(-1.5 * (z + topic_eff[topic])),
                          1.4 if subj != "English Language" else 0.4,
                          2.0 if careless else 0.5,
                          0.8 * sigmoid(-1.3 * eng_skill),
                          0.9 * sigmoid(-eng) + (0.5 if mx == 5 else 0),
                          0.0,
                          (1.3 if subj == "English Language" else 0.5) * sigmoid(-1.3 * eng_skill)])
            err = ERROR_TYPES[rng.choice(len(w), p=w / w.sum())]
        time_s = int(np.clip(rng.normal(60 * mx * (0.9 - 0.08 * z) * (0.65 if careless else 1), 25), 10, 900))
        out.append(dict(student_id=sid, term=t, subject=subj, class_level=lvl, question_id=qid, question_no=qno,
                        strand=strand, topic=topic, difficulty=diff, cognitive_level=bloom, max_marks=mx,
                        marks_awarded=marks, fully_correct=int(marks == mx), error_type=err,
                        time_spent_sec=0 if no_attempt else time_s))
    return out


if __name__ == "__main__":
    build()
