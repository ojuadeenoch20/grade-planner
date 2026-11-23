"""
Train the two prediction stages on data/*.csv (synthetic now, real later: same schema).

Stage A (pre-semester): predicts a course's final total score before any CA exists.
Stage B (mid-semester): same, plus CA fraction and assignment completion. Only for
                         courses that have a CA (ca_max > 0).
Each stage saves three models: a point estimate and a 10th/90th percentile pair that
gives an 80% range, so risk tags can use uncertainty, not just a single number.

NOTE: on synthetic data these metrics only show the pipeline works. They are NOT
real-world accuracy. Re-run on real records and trust only those numbers.

Usage:  python train_models.py --data data --out models
"""
import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import schema as S

CATS = ["department", "course_category", "course_type"]
NUM_A = ["level", "units", "perceived_difficulty", "ca_max", "study_hours_per_week",
         "sleep_hours", "outside_workload", "attendance_pct", "prev_gpa", "prev_cgpa",
         "gpa_trend", "carryover_count", "semester_units", "entry_score",
         "category_hist_mean", "n_prior_semesters"]
NUM_B = NUM_A + ["ca_fraction", "assignment_completion"]
TARGET = "total_score"
BANDS = [g for _, g, _ in S.GRADE_SCALE][::-1]   # F, E, D, C, B, A


def build_table(data_dir):
    students = pd.read_csv(f"{data_dir}/students.csv")
    sem = pd.read_csv(f"{data_dir}/semesters.csv").sort_values(["anon_id", "session", "semester"])
    crs = pd.read_csv(f"{data_dir}/courses.csv")

    g = sem.groupby("anon_id")
    sem["n_prior_semesters"] = g.cumcount()
    sem["gpa_trend"] = sem["prev_gpa"] - g["prev_gpa"].shift(1)

    df = crs.merge(sem.drop(columns=["level"]), on=["anon_id", "session", "semester"], how="left")
    df = df.merge(students[["anon_id", "entry_score"]], on="anon_id", how="left")
    df = df.sort_values(["anon_id", "session", "semester"]).reset_index(drop=True)

    # Past performance in the same course category, using strictly earlier semesters only.
    # Built from letter grades (S.GRADE_REP) because that is what students enter in the app.
    hist, out = {}, np.full(len(df), np.nan)
    for _, idx in df.groupby(["anon_id", "session", "semester"], sort=False).groups.items():
        idx = list(idx)
        sid = df.at[idx[0], "anon_id"]
        stats = hist.setdefault(sid, {})
        for i in idx:
            s, n = stats.get(df.at[i, "course_category"], (0.0, 0))
            out[i] = s / n if n else np.nan
        for i in idx:
            cat = df.at[i, "course_category"]
            s, n = stats.get(cat, (0.0, 0))
            stats[cat] = (s + S.GRADE_REP[df.at[i, "grade"]], n + 1)
    df["category_hist_mean"] = out

    df["ca_fraction"] = np.where(df["ca_max"] > 0, df["ca_score"] / df["ca_max"].replace(0, np.nan), np.nan)
    df["assignment_completion"] = df["assignments_done"] / df["assignments_total"]
    for c in CATS:
        df[c] = df[c].astype("category")
    return df


def gbm(**kw):
    return HistGradientBoostingRegressor(max_iter=250, learning_rate=0.07, max_leaf_nodes=20,
                                         min_samples_leaf=40, random_state=0,
                                         categorical_features="from_dtype", **kw)


def ridge(num):
    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATS)])
    return make_pipeline(pre, Ridge(alpha=5.0))


def band_distance(pred, actual):
    p = pd.Series(pred).apply(S.score_to_grade).map(BANDS.index).to_numpy()
    a = pd.Series(actual).apply(S.score_to_grade).map(BANDS.index).to_numpy()
    return np.abs(p - a)


def run_stage(name, df, num, seed):
    cols = num + CATS
    train_i, test_i = next(GroupShuffleSplit(test_size=0.2, random_state=seed).split(df, groups=df["anon_id"]))
    tr, te = df.iloc[train_i], df.iloc[test_i]   # no student appears in both sets
    Xtr, Xte, ytr, yte = tr[cols], te[cols], tr[TARGET], te[TARGET]

    point, lo, hi = gbm(), gbm(loss="quantile", quantile=0.1), gbm(loss="quantile", quantile=0.9)
    for m in (point, lo, hi):
        m.fit(Xtr, ytr)
    base = ridge(num).fit(Xtr, ytr)

    pred = np.clip(point.predict(Xte), 0, 100)
    lo_p, hi_p = np.clip(lo.predict(Xte), 0, 100), np.clip(hi.predict(Xte), 0, 100)
    dist = band_distance(pred, yte)
    metrics = {
        "rows_train": len(tr), "rows_test": len(te),
        "mae_mean_baseline": round(float(np.abs(yte - ytr.mean()).mean()), 2),
        "mae_ridge": round(float(np.abs(yte - base.predict(Xte)).mean()), 2),
        "mae_gbm": round(float(np.abs(yte - pred).mean()), 2),
        "exact_grade_pct": round(float((dist == 0).mean() * 100), 1),
        "within_one_grade_pct": round(float((dist <= 1).mean() * 100), 1),
        "interval_80_coverage_pct": round(float(((yte >= lo_p) & (yte <= hi_p)).mean() * 100), 1),
        "avg_interval_width": round(float((hi_p - lo_p).mean()), 1),
    }
    bundle = {"point": point, "lo": lo, "hi": hi, "features": cols,
              "categories": {c: list(df[c].cat.categories) for c in CATS},
              "trained_on": "synthetic"}
    return bundle, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="models")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    df = build_table(a.data)
    os.makedirs(a.out, exist_ok=True)
    all_metrics = {}
    stages = {"stage_a": (df, NUM_A), "stage_b": (df[df["ca_max"] > 0], NUM_B)}
    for name, (frame, num) in stages.items():
        bundle, metrics = run_stage(name, frame.reset_index(drop=True), num, a.seed)
        joblib.dump(bundle, f"{a.out}/{name}.joblib")
        all_metrics[name] = metrics
        print(f"\n{name}: {metrics}")

    with open(f"{a.out}/metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    print("\nREMINDER: trained on SIMULATED data. These numbers are not real-world accuracy.")


if __name__ == "__main__":
    main()