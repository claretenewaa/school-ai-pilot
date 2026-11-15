"""Train and evaluate the AI models. Saves to models/ and prints a metrics report."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ai.early_warning import (FEATURES, NUMERIC, TARGET, EarlyWarning, add_features, make_explainer,  # noqa: E402
                              make_gbm)
from ai.knowledge_bot import MIN_SCORE, KnowledgeBot  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA, MODELS = ROOT / "data", ROOT / "models"


def scores(y, p, thr):
    yhat = (p >= thr).astype(int)
    return dict(roc_auc=round(roc_auc_score(y, p), 3), pr_auc=round(average_precision_score(y, p), 3),
                precision=round(precision_score(y, yhat), 3), recall=round(recall_score(y, yhat), 3),
                f1=round(f1_score(y, yhat), 3), flagged_share=round(yhat.mean(), 3), base_rate=round(y.mean(), 3))


def pick_threshold(y, p, target_recall=0.80):
    """Highest threshold that still catches at least 80% of at-risk records."""
    for thr in np.arange(0.9, 0.05, -0.01):
        if recall_score(y, (p >= thr).astype(int)) >= target_recall:
            return float(round(thr, 2))
    return 0.5


def train_early_warning():
    df = add_features(pd.read_csv(DATA / "student_subject_term.csv", dtype={"grade": str}))
    X, y, groups = df[FEATURES], df[TARGET].to_numpy(), df["student_id"]

    # 1) Hold out 20% of students entirely (no leakage of the same child across splits)
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(X, y, groups))
    tr2, val = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=7).split(X.iloc[tr], y[tr], groups.iloc[tr]))
    gbm = make_gbm().fit(X.iloc[tr[tr2]], y[tr[tr2]])
    thr = pick_threshold(y[tr[val]], gbm.predict_proba(X.iloc[tr[val]])[:, 1])
    gbm = make_gbm().fit(X.iloc[tr], y[tr])
    expl = make_explainer().fit(X.iloc[tr], y[tr])
    p_te = gbm.predict_proba(X.iloc[te])[:, 1]
    metrics = {"threshold": thr,
               "held_out_students_20pct": scores(y[te], p_te, thr),
               "logistic_explainer_held_out": scores(y[te], expl.predict_proba(X.iloc[te])[:, 1], thr)}

    # 2) Pilot rule: accuracy on 50 students the model never saw
    test_students = groups.iloc[te].drop_duplicates().sample(50, random_state=1)
    m50 = groups.iloc[te].isin(test_students).to_numpy()
    metrics["held_out_50_students"] = scores(y[te][m50], p_te[m50], thr) | {"records": int(m50.sum())}

    # 3) Time-based check: train on Terms 1-2, test on Term 3
    t12, t3 = (df.term <= 2).to_numpy(), (df.term == 3).to_numpy()
    gbm_t = make_gbm().fit(X[t12], y[t12])
    metrics["train_terms_1_2_test_term_3"] = scores(y[t3], gbm_t.predict_proba(X[t3])[:, 1], thr)

    # fairness check: flag rates and recall by gender and school type (attributes NOT used by the model)
    st = pd.read_csv(DATA / "students.csv")[["student_id", "gender"]]
    sc = pd.read_csv(DATA / "schools.csv")[["school_id", "school_type"]]
    te_df = df.iloc[te].assign(p=p_te).merge(st, on="student_id").merge(sc, on="school_id")
    fair = {}
    for col in ("gender", "school_type"):
        for k, g in te_df.groupby(col):
            fair[f"{col}={k}"] = dict(recall=round(recall_score(g.at_risk, g.p >= thr), 3),
                                      flagged_share=round(float((g.p >= thr).mean()), 3),
                                      base_rate=round(float(g.at_risk.mean()), 3))
    metrics["fairness_held_out"] = fair

    # Final model on all data
    gbm_all = make_gbm().fit(X, y)
    expl_all = make_explainer().fit(X, y)
    reference = df.loc[df.at_risk == 0, NUMERIC].median().to_dict()
    direction = np.sign(df[NUMERIC].corrwith(df[TARGET], method="spearman")).fillna(0).to_dict()
    # top coefficients for transparency
    names = expl_all.named_steps["pre"].get_feature_names_out()
    coefs = pd.Series(expl_all.named_steps["clf"].coef_[0], index=[n.split("__", 1)[1] for n in names])
    metrics["strongest_signals"] = coefs.reindex(coefs.abs().sort_values(ascending=False).index)[:12].round(3).to_dict()
    MODELS.mkdir(exist_ok=True)
    joblib.dump(EarlyWarning(gbm_all, expl_all, thr, reference, direction, metrics), MODELS / "early_warning.joblib")
    return metrics


def eval_knowledge_bot():
    bot = KnowledgeBot()
    qs = pd.read_csv(DATA / "parent_questions.csv")
    hit1 = hit3 = n_ans = abstain_ok = n_no = false_abstain = 0
    for _, q in qs.iterrows():
        hits = bot.retrieve(q.question)
        if q.answerable == "yes":
            n_ans += 1
            srcs = [h["source"] for h in hits[:3]]
            hit1 += srcs[0] == q.expected_source
            hit3 += q.expected_source in srcs
            false_abstain += hits[0]["score"] < MIN_SCORE
        else:
            n_no += 1
            abstain_ok += hits[0]["score"] < MIN_SCORE
    return dict(answerable_questions=n_ans, right_document_top1=round(hit1 / n_ans, 3),
                right_document_top3=round(hit3 / n_ans, 3), wrongly_refused_offline=false_abstain,
                unanswerable_questions=n_no, correctly_refused_offline=abstain_ok,
                note="Offline refusal uses a score threshold only. With Claude, refusal is decided by the model from the excerpts.")


if __name__ == "__main__":
    m = {"early_warning": train_early_warning(), "knowledge_bot_retrieval": eval_knowledge_bot()}
    (MODELS / "metrics.json").write_text(json.dumps(m, indent=2))
    print(json.dumps(m, indent=2))


def train_with_school(school_sst: pd.DataFrame, out_path: Path, school_weight: float = 3.0) -> dict:
    """Retrain on simulated students + the school's own labelled records (weighted higher).
    Accuracy is reported on 50 of the school's own students the model never saw (pilot rule)."""
    syn = add_features(pd.read_csv(DATA / "student_subject_term.csv", dtype={"grade": str}))
    sch = add_features(school_sst[school_sst.at_risk.notna()].copy())
    sch["at_risk"] = sch.at_risk.astype(int)
    n_students = sch.student_id.nunique()
    metrics = {"school_labelled_records": int(len(sch)), "school_labelled_students": int(n_students)}
    base = joblib.load(MODELS / "early_warning.joblib")

    def fit(df_s):
        X = pd.concat([syn[FEATURES], df_s[FEATURES]], ignore_index=True)
        y = np.concatenate([syn[TARGET].to_numpy(), df_s[TARGET].to_numpy()])
        w = np.concatenate([np.ones(len(syn)), np.full(len(df_s), school_weight)])
        return (make_gbm().fit(X, y, clf__sample_weight=w), make_explainer().fit(X, y, clf__sample_weight=w))

    if n_students >= 80:
        test_ids = sch.student_id.drop_duplicates().sample(50, random_state=3)
        te = sch.student_id.isin(test_ids)
        gbm, _ = fit(sch[~te])
        y = sch.loc[te, TARGET].to_numpy()
        metrics["held_out_50_school_students"] = scores(y, gbm.predict_proba(sch.loc[te, FEATURES])[:, 1], base.threshold)
        metrics["simulated_only_model_same_50"] = scores(y, base.gbm.predict_proba(sch.loc[te, FEATURES])[:, 1],
                                                         base.threshold)
    else:
        metrics["note"] = "Fewer than 80 school students with exam results: not enough to hold out 50 for testing."
    gbm, expl = fit(sch)
    allx = pd.concat([syn, sch], ignore_index=True)
    reference = allx.loc[allx.at_risk == 0, NUMERIC].median().to_dict()
    direction = np.sign(allx[NUMERIC].corrwith(allx[TARGET], method="spearman")).fillna(0).to_dict()
    model_metrics = dict(base.metrics, school_retrain=metrics)
    joblib.dump(EarlyWarning(gbm, expl, base.threshold, reference, direction, model_metrics), out_path)
    return metrics
