"""Shared Ghana basic-school curriculum reference (Class 1 to JHS 3).

Based on the structure of the GES/NaCCA Standards-Based Curriculum (B1-B6)
and the Common Core Programme (B7-B9). Topic lists are simplified for the pilot.
"""

LEVELS = list(range(1, 10))
LEVEL_LABEL = {1: "Class 1", 2: "Class 2", 3: "Class 3", 4: "Class 4", 5: "Class 5",
               6: "Class 6", 7: "JHS 1", 8: "JHS 2", 9: "JHS 3"}
LABEL_TO_LEVEL = {v: k for k, v in LEVEL_LABEL.items()}


def band(level: int) -> str:
    """Lower primary (B1-B3), upper primary (B4-B6) or JHS (B7-B9)."""
    return "lower" if level <= 3 else ("upper" if level <= 6 else "jhs")


SUBJECTS = {
    "lower": ["English Language", "Mathematics", "Science", "Our World Our People",
              "Ghanaian Language", "Creative Arts", "Religious & Moral Education"],
    "upper": ["English Language", "Mathematics", "Science", "Our World Our People",
              "Ghanaian Language", "History", "Computing", "Religious & Moral Education"],
    "jhs": ["English Language", "Mathematics", "Science", "Social Studies",
            "Religious & Moral Education", "Computing", "Career Technology",
            "Creative Arts & Design", "Ghanaian Language"],
}

# Periods per week for each subject (40-minute periods)
PERIODS_PER_WEEK = {
    "lower": {"English Language": 6, "Mathematics": 6, "Science": 3, "Our World Our People": 4,
              "Ghanaian Language": 3, "Creative Arts": 3, "Religious & Moral Education": 2},
    "upper": {"English Language": 6, "Mathematics": 6, "Science": 4, "Our World Our People": 3,
              "Ghanaian Language": 3, "History": 2, "Computing": 2, "Religious & Moral Education": 2},
    "jhs": {"English Language": 5, "Mathematics": 5, "Science": 4, "Social Studies": 3,
            "Religious & Moral Education": 2, "Computing": 2, "Career Technology": 3,
            "Creative Arts & Design": 3, "Ghanaian Language": 3},
}

# Fixed error categories, as the pilot plan requires
ERROR_TYPES = ["Conceptual misunderstanding", "Procedural/calculation error", "Careless slip",
               "Misread question", "Incomplete answer", "No attempt", "Language/vocabulary gap"]

# Topics per subject and band: (strand, topic, term taught)
TOPICS = {
    "Mathematics": {
        "lower": [("Number", "Counting and number names", 1), ("Number", "Place value", 1),
                  ("Number", "Addition", 1), ("Number", "Subtraction", 2),
                  ("Algebra", "Patterns and relationships", 2), ("Geometry", "2D and 3D shapes", 2),
                  ("Measurement", "Length and capacity", 3),
                  ("Measurement", "Money: Ghana cedis and pesewas", 3), ("Data", "Pictographs", 3)],
        "upper": [("Number", "Whole numbers and place value", 1), ("Number", "Multiplication and division", 1),
                  ("Number", "Fractions", 1), ("Number", "Decimals", 2), ("Number", "Percentages", 2),
                  ("Algebra", "Patterns and algebraic expressions", 2),
                  ("Measurement", "Perimeter and area", 3), ("Geometry", "Angles and shapes", 3),
                  ("Data", "Data handling: bar graphs and mean", 3)],
        "jhs": [("Number", "Number sets and operations", 1), ("Number", "Rational numbers and fractions", 1),
                ("Number", "Indices and standard form", 1), ("Algebra", "Algebraic expressions", 2),
                ("Algebra", "Linear equations and inequalities", 2),
                ("Number", "Ratio, proportion and percentages", 2),
                ("Geometry", "Angles and constructions", 3), ("Measurement", "Mensuration: area and volume", 3),
                ("Data", "Statistics and probability", 3)],
    },
    "English Language": {
        "lower": [("Oral language", "Listening and oral language", 1), ("Reading", "Phonics and letter sounds", 1),
                  ("Reading", "Sight words and vocabulary", 1), ("Reading", "Reading comprehension", 2),
                  ("Grammar", "Nouns and pronouns", 2), ("Writing", "Handwriting and spelling", 2),
                  ("Grammar", "Verbs and simple tenses", 3), ("Writing", "Writing simple sentences", 3),
                  ("Oral language", "Story telling", 3)],
        "upper": [("Reading", "Reading comprehension", 1), ("Grammar", "Parts of speech", 1),
                  ("Reading", "Vocabulary and word meanings", 1), ("Grammar", "Tenses and concord", 2),
                  ("Writing", "Letter writing", 2), ("Writing", "Punctuation", 2),
                  ("Writing", "Summary writing", 3), ("Writing", "Narrative composition", 3),
                  ("Writing", "Spelling and dictation", 3)],
        "jhs": [("Reading", "Comprehension", 1), ("Grammar", "Concord and tenses", 1),
                ("Reading", "Vocabulary: synonyms, antonyms and idioms", 1),
                ("Writing", "Formal and informal letters", 2), ("Writing", "Summary writing", 2),
                ("Literature", "Prose and poetry", 2), ("Grammar", "Clauses and phrases", 3),
                ("Writing", "Argumentative essay", 3), ("Oral language", "Oral English: vowels and consonants", 3)],
    },
    "Science": {
        "lower": [("Diversity of matter", "Living and non-living things", 1),
                  ("Systems", "Parts of the human body", 1), ("Diversity of matter", "Plants around us", 1),
                  ("Cycles", "Water and its uses", 2), ("Cycles", "Weather and seasons", 2),
                  ("Energy", "Sources of energy", 2), ("Humans and environment", "Personal hygiene and health", 3),
                  ("Diversity of matter", "Animals and their homes", 3), ("Energy", "Light and sound", 3)],
        "upper": [("Diversity of matter", "Materials and their properties", 1),
                  ("Cycles", "Life cycles of organisms", 1), ("Systems", "Human digestive system", 1),
                  ("Forces", "Forces and motion", 2), ("Energy", "Electricity and magnetism", 2),
                  ("Energy", "Energy forms and conversion", 2), ("Humans and environment", "Ecosystems and food chains", 3),
                  ("Humans and environment", "Malaria and cholera prevention", 3),
                  ("Humans and environment", "Soil and conservation", 3)],
        "jhs": [("Diversity of matter", "Elements, compounds and mixtures", 1), ("Systems", "Cells and cell division", 1),
                ("Diversity of matter", "Measurement and SI units", 1), ("Forces", "Forces, motion and pressure", 2),
                ("Energy", "Electrical circuits and electronics", 2), ("Systems", "Photosynthesis and plant nutrition", 2),
                ("Cycles", "Reproduction in humans and plants", 3),
                ("Humans and environment", "Ecosystems and pollution", 3),
                ("Diversity of matter", "Acids, bases and salts", 3)],
    },
}

# Extra topic lists so the question generator covers other subjects
OTHER_TOPICS = {
    "Social Studies": ["The environment and its resources", "Our culture and identity",
                       "Population and development", "Governance and citizenship",
                       "Ghana's independence and leaders", "Tourism in Ghana", "Road safety"],
    "Our World Our People": ["My family and community", "Festivals in Ghana", "Our national symbols",
                             "Occupations in my community", "Keeping our environment clean"],
    "Religious & Moral Education": ["God the creator", "Religious festivals", "Honesty and integrity",
                                    "Respect for elders", "The three major religions in Ghana"],
    "Computing": ["Parts of a computer", "Using a keyboard and mouse", "Internet safety",
                  "Word processing basics", "Introduction to coding with Scratch"],
    "History": ["How Ghana got its name", "Early migrations into Ghana", "European arrivals on the Gold Coast",
                "Road to independence", "Ghana's republics"],
    "Career Technology": ["Health and safety in the workshop", "Food and nutrition",
                          "Sewing and textiles basics", "Tools and materials", "Entrepreneurship"],
    "Creative Arts & Design": ["Elements of design", "Drawing and painting", "Ghanaian music and dance",
                               "Adinkra symbols", "Drama and performance"],
    "Creative Arts": ["Drawing and colouring", "Songs and rhymes", "Craft with local materials", "Dance and movement"],
    "Ghanaian Language": ["Letter sounds and reading", "Proverbs", "Folktales (Ananse stories)",
                          "Letter writing", "Grammar: nouns and verbs"],
}


def topics_for(subject: str, level: int):
    """Topic names for a subject at a class level."""
    if subject in TOPICS:
        return [t[1] for t in TOPICS[subject][band(level)]]
    return OTHER_TOPICS.get(subject, [])


def jhs_grade(total: float) -> tuple[int, str]:
    """JHS grading scale (1 = highest, 9 = lowest)."""
    cuts = [(80, 1, "Highest"), (70, 2, "Higher"), (60, 3, "High"), (55, 4, "High Average"),
            (50, 5, "Average"), (45, 6, "Low Average"), (40, 7, "Low"), (35, 8, "Lower"), (0, 9, "Lowest")]
    for cut, g, desc in cuts:
        if total >= cut:
            return g, desc
    return 9, "Lowest"


def primary_proficiency(total: float) -> tuple[str, str]:
    """Standards-Based Curriculum proficiency levels for B1-B6."""
    if total >= 80:
        return "HP", "Highly Proficient"
    if total >= 68:
        return "P", "Proficient"
    if total >= 54:
        return "AP", "Approaching Proficiency"
    if total >= 40:
        return "D", "Developing"
    return "E", "Emerging"
