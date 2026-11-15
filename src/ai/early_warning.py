"""Early-warning model: flags student-subject records likely to fail (<50) or drop sharply (15+ points)
in the end-of-term exam, using only information available before the exam.

Gradient boosting gives the risk score. A logistic regression on the same features
gives the reasons behind each flag in plain words.
Demographics, home background and fee payments are left out on purpose (fairness and pilot ground rules).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

NUMERIC = [
    "class_level", "prev_exam_score", "class_test_avg", "quiz_avg", "homework_avg", "ca_score_30",
    "assignment_submission_rate", "classwork_completion_rate", "lesson_attendance_rate", "attendance_rate",
    "late_arrivals", "max_consecutive_days_absent", "days_absent_before_tests", "times_left_early",
    "participation_rating", "answers_per_week", "group_work_rating", "help_seeking_rating", "time_on_task_pct",
    "observation_rating", "peer_learning_rating", "improvement_after_feedback", "discipline_incidents",
    "test_time_ratio", "has_prev_score", "intervention_this_term",
]
CATEGORICAL = ["subject", "common_error_type"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "at_risk"

LABELS = {
    "prev_exam_score": ("Last term's exam score", "{:.0f}%"),
    "class_test_avg": ("Class test average", "{:.0f}%"),
    "quiz_avg": ("Quiz average", "{:.0f}%"),
    "homework_avg": ("Homework average", "{:.0f}%"),
    "ca_score_30": ("Class score (CA)", "{:.1f}/30"),
    "assignment_submission_rate": ("Assignments submitted", "{:.0%}"),
    "classwork_completion_rate": ("Classwork completed", "{:.0%}"),
    "lesson_attendance_rate": ("Attendance in this subject's lessons", "{:.0%}"),
    "attendance_rate": ("School attendance this term", "{:.0%}"),
    "late_arrivals": ("Late arrivals this term", "{:.0f}"),
    "max_consecutive_days_absent": ("Longest run of days absent", "{:.0f} days"),
    "days_absent_before_tests": ("Days missed just before tests", "{:.0f}"),
    "times_left_early": ("Times left school early", "{:.0f}"),
    "participation_rating": ("Participation in discussions", "{:.0f}/5"),
    "answers_per_week": ("Questions answered in class per week", "{:.0f}"),
    "group_work_rating": ("Group work participation", "{:.0f}/5"),
    "help_seeking_rating": ("Asks for help when confused", "{:.0f}/5"),
    "time_on_task_pct": ("Time on task in lessons", "{:.0f}%"),
    "observation_rating": ("Teacher observation rating", "{:.0f}/5"),
    "peer_learning_rating": ("Peer learning participation", "{:.0f}/5"),
    "improvement_after_feedback": ("Improvement after feedback", "{:+.1f} pts"),
    "discipline_incidents": ("Discipline incidents", "{:.0f}"),
    "test_time_ratio": ("Time used on tests", "{:.0%} of time allowed"),
}


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["test_time_ratio"] = df["avg_test_time_min"] / df["test_time_allowed_min"]
    df["has_prev_score"] = df["prev_exam_score"].notna().astype(int)
    df["intervention_this_term"] = df["intervention_this_term"].astype(int)
    return df


def make_gbm() -> Pipeline:
    pre = ColumnTransformer([
        ("num", "passthrough", NUMERIC),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CATEGORICAL),
    ])
    cat_mask = [False] * len(NUMERIC) + [True] * len(CATEGORICAL)
    return Pipeline([("pre", pre), ("clf", HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=1.0,
        categorical_features=cat_mask, early_stopping=True, random_state=0))])


def make_explainer() -> Pipeline:
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000, C=0.5))])


class EarlyWarning:
    def __init__(self, gbm, explainer, threshold, reference, direction, metrics):
        self.gbm, self.explainer, self.threshold = gbm, explainer, threshold
        self.reference = reference  # median of each feature among students not at risk
        self.direction = direction  # +1 if higher values go with more risk, -1 if lower values do
        self.metrics = metrics

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.gbm.predict_proba(add_features(df)[FEATURES])[:, 1]

    def reasons(self, row: pd.Series, k: int = 3) -> list[str]:
        """Top reasons pushing this record's risk up, in plain words."""
        x = add_features(row.to_frame().T)
        pre = self.explainer.named_steps["pre"]
        coef = self.explainer.named_steps["clf"].coef_[0]
        z = pre.transform(x[FEATURES])
        z = z.toarray()[0] if hasattr(z, "toarray") else np.asarray(z)[0]
        contrib = coef * z
        names = pre.get_feature_names_out()
        out = []
        for j in np.argsort(-contrib):
            if contrib[j] <= 0.05 or len(out) >= k:
                break
            name = names[j].split("__", 1)[1]
            if name in LABELS:
                label, fmt = LABELS[name]
                val = x[name].iloc[0]
                if pd.isna(val):
                    continue
                typical = self.reference.get(name)
                # only report a feature when this learner is on the risky side of typical
                if typical is not None and not pd.isna(typical) and (val - typical) * self.direction.get(name, 0) <= 0:
                    continue
                txt = f"{label}: {fmt.format(val)}"
                if typical is not None and not pd.isna(typical):
                    txt += f" (typical for students doing well: {fmt.format(typical)})"
                out.append(txt)
            elif name.startswith("common_error_type_"):
                out.append(f"Most common mistake: {name.replace('common_error_type_', '')}")
        return out or ["Several small signals together; no single strong reason"]
