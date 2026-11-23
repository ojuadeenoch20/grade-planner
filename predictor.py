"""
Turns what a student enters in the app into model features, and model output into
predicted scores, grades, ranges and risk tags.

Everything about past performance is derived from the grades the student typed in,
so they enter their history once and the rest is calculated.
"""
import math
import os

import joblib
import numpy as np
import pandas as pd

import schema as S
import calculators as C

# Risk thresholds are PROVISIONAL. Re-tune on real data.
AT_RISK_SCORE, AT_RISK_LOW = 50, 40      # predicted below 50, or the 80% range reaches a fail
WATCH_SCORE, WATCH_LOW = 60, 50
HEAVY_LOAD_UNITS = 24

_BUNDLES = {}


def load_models(models_dir="models"):
    for stage in ("stage_a", "stage_b"):
        _BUNDLES[stage] = joblib.load(os.path.join(models_dir, f"{stage}.joblib"))
    return _BUNDLES


def risk_tag(score, low):
    if score < AT_RISK_SCORE or low < AT_RISK_LOW:
        return "At-Risk"
    if score < WATCH_SCORE or low < WATCH_LOW:
        return "Watch"
    return "Safe"


def history_features(history):
    """history: list of semesters, each a list of {units, category, grade} (oldest first)."""
    sem_units = [[c["units"] for c in sem] for sem in history]
    sem_points = [[S.grade_to_point(c["grade"]) for c in sem] for sem in history]
    all_u = [u for sem in sem_units for u in sem]
    all_p = [p for sem in sem_points for p in sem]

    gpas = [S.gpa(u, p) for u, p in zip(sem_units, sem_points)]
    prev_gpa = gpas[-1] if gpas else np.nan
    gpa_trend = gpas[-1] - gpas[-2] if len(gpas) >= 2 else np.nan

    by_cat = {}
    for sem in history:
        for c in sem:
            by_cat.setdefault(c["category"], []).append(S.GRADE_REP[c["grade"]])

    return {
        "prev_gpa": prev_gpa,
        "prev_cgpa": S.gpa(all_u, all_p) if all_u else np.nan,
        "gpa_trend": gpa_trend,
        "carryover_count": sum(1 for p in all_p if p == 0),
        "n_prior_semesters": len(history),
        "past_units": all_u,
        "past_points": all_p,
        "cat_means": {k: float(np.mean(v)) for k, v in by_cat.items()},
    }


def predict_semester(student, courses, models_dir="models"):
    """
    student: department, level, entry_score?, study_hours, sleep_hours, outside_workload,
             expected_attendance_pct, history
    courses: code, units, category, course_type, difficulty, ca_max, ca_score?,
             attendance_pct?, assignments_done?, assignments_total?
    """
    if not _BUNDLES:
        load_models(models_dir)
    hf = history_features(student["history"])
    sem_units = sum(c["units"] for c in courses)

    rows, stages = [], []
    for c in courses:
        has_ca = c["ca_max"] > 0 and c.get("ca_score") is not None
        stage = "stage_b" if has_ca else "stage_a"
        a_done, a_tot = c.get("assignments_done"), c.get("assignments_total")
        rows.append({
            "level": student["level"], "units": c["units"],
            "perceived_difficulty": c.get("difficulty", 3), "ca_max": c["ca_max"],
            "study_hours_per_week": student["study_hours"], "sleep_hours": student["sleep_hours"],
            "outside_workload": int(student["outside_workload"]),
            "attendance_pct": c.get("attendance_pct") or student["expected_attendance_pct"],
            "prev_gpa": hf["prev_gpa"], "prev_cgpa": hf["prev_cgpa"], "gpa_trend": hf["gpa_trend"],
            "carryover_count": hf["carryover_count"], "semester_units": sem_units,
            "entry_score": student.get("entry_score") if student.get("entry_score") is not None else np.nan,
            "category_hist_mean": hf["cat_means"].get(c["category"], np.nan),
            "n_prior_semesters": hf["n_prior_semesters"],
            "ca_fraction": c["ca_score"] / c["ca_max"] if has_ca else np.nan,
            "assignment_completion": a_done / a_tot if a_done is not None and a_tot else np.nan,
            "department": student["department"], "course_category": c["category"],
            "course_type": c.get("course_type", "core"),
        })
        stages.append(stage)

    out = [None] * len(courses)
    for stage in ("stage_a", "stage_b"):
        idx = [i for i, s in enumerate(stages) if s == stage]
        if not idx:
            continue
        b = _BUNDLES[stage]
        X = pd.DataFrame([rows[i] for i in idx])[b["features"]].copy()
        for col, cats in b["categories"].items():
            X[col] = pd.Categorical(X[col], categories=cats)
        pt = np.clip(b["point"].predict(X), 0, 100)
        lo = np.clip(b["lo"].predict(X), 0, 100)
        hi = np.clip(b["hi"].predict(X), 0, 100)
        for j, i in enumerate(idx):
            lo_j, hi_j = min(lo[j], pt[j]), max(hi[j], pt[j])   # keep point inside its range
            out[i] = {"code": courses[i]["code"], "units": courses[i]["units"],
                      "stage": "B" if stage == "stage_b" else "A",
                      "predicted_score": round(float(pt[j]), 1),
                      "range_low": round(float(lo_j), 1), "range_high": round(float(hi_j), 1),
                      "predicted_grade": S.score_to_grade(pt[j]),
                      "risk": risk_tag(float(pt[j]), float(lo_j))}

    units = [o["units"] for o in out]
    pts = lambda key: [S.grade_to_point(S.score_to_grade(o[key])) for o in out]
    gpa_mid, gpa_lo, gpa_hi = (S.gpa(units, pts(k)) for k in ("predicted_score", "range_low", "range_high"))
    cg = lambda g_pts: C.projected_cgpa(hf["past_units"], hf["past_points"], units, g_pts)
    return {
        "courses": out,
        "semester": {
            "units": sem_units,
            "predicted_gpa": round(gpa_mid, 2), "gpa_low": round(gpa_lo, 2), "gpa_high": round(gpa_hi, 2),
            "predicted_cgpa": round(cg(pts("predicted_score")), 2),
            "current_cgpa": None if math.isnan(hf["prev_cgpa"]) else round(hf["prev_cgpa"], 2),
            "heavy_load": sem_units > HEAVY_LOAD_UNITS,
        },
        "model_note": "Prototype model trained on simulated data. Treat as guidance, not a verdict.",
    }
