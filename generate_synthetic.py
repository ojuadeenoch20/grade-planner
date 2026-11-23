"""
Synthetic data generator for the Student Performance Predictor (PROTOTYPE DATA).

Everything here is simulated from assumptions written in this file. A model trained
on it recovers these assumptions, not real student behaviour, so never quote its
accuracy as real-world accuracy. Validate only on real records.

Usage:  python generate_synthetic.py --n-students 1500 --seed 42 --out data
"""
import argparse
import os

import numpy as np
import pandas as pd

import schema as S

# Placeholder departments. CA share of the 100 marks is the assessment structure:
# 0 means no CA (exam /100); 30 means CA /30 + exam /70; 40 means CA /40 + exam /60.
DEPARTMENTS = {
    # "cats" order follows schema.CATEGORIES: calculation, applied, writing, theory, practical, general
    "CSC": {"ca": {30: 1.0}, "cats": [0.20, 0.40, 0.05, 0.25, 0.05, 0.05]},
    "EEE": {"ca": {30: 0.7, 40: 0.3}, "cats": [0.30, 0.25, 0.05, 0.15, 0.20, 0.05]},
    "ACC": {"ca": {0: 0.8, 30: 0.2}, "cats": [0.30, 0.00, 0.10, 0.45, 0.00, 0.15]},
    "MEE": {"ca": {0: 1.0}, "cats": [0.25, 0.20, 0.05, 0.20, 0.25, 0.05]},
    "LAW": {"ca": {30: 0.5, 0: 0.5}, "cats": [0.02, 0.05, 0.38, 0.45, 0.00, 0.10]},
    "BIO": {"ca": {30: 0.7, 40: 0.3}, "cats": [0.15, 0.10, 0.10, 0.35, 0.25, 0.05]},
}
CAT_BASE_DIFFICULTY = {"calculation": 3.6, "applied": 3.3, "writing": 2.7, "theory": 2.8,
                       "practical": 2.6, "general": 2.0}


def build_catalog(rng):
    """One fixed course list per (department, level, semester), shared by all students."""
    catalog = {}
    for dept, cfg in DEPARTMENTS.items():
        cas, cws = list(cfg["ca"].keys()), list(cfg["ca"].values())
        dept_harshness = rng.normal(0, 1.5)
        for lvl in range(1, 6):
            for sem in (1, 2):
                courses = []
                for i in range(rng.integers(6, 9)):
                    is_gst = i == 0 and lvl <= 2
                    cat = "general" if is_gst else rng.choice(S.CATEGORIES, p=cfg["cats"])
                    courses.append({
                        "course_code": f"{'GST' if is_gst else dept} {lvl}{sem}{i}",
                        "units": 2 if is_gst else int(rng.choice([1, 2, 3, 3, 4])),
                        "course_category": cat,
                        "course_type": "general" if is_gst else ("elective" if rng.random() < 0.12 else "core"),
                        "true_difficulty": float(np.clip(CAT_BASE_DIFFICULTY[cat] + rng.normal(0, 0.5), 1, 5)),
                        "course_effect": float(rng.normal(0, 3.5) - dept_harshness),
                        "ca_max": int(rng.choice(cas, p=cws)),
                    })
                catalog[(dept, lvl, sem)] = courses
    return catalog


def simulate(n_students, seed):
    rng = np.random.default_rng(seed)
    catalog = build_catalog(rng)
    students, semesters, courses = [], [], []

    for n in range(n_students):
        sid = f"S{n + 1:05d}"
        dept = rng.choice(list(DEPARTMENTS))
        ability = rng.normal(0, 1)
        diligence = 0.3 * ability + np.sqrt(1 - 0.09) * rng.normal(0, 1)
        aptitude = {c: rng.normal(0, 3.5) for c in S.CATEGORIES}
        sleep_mean = rng.normal(6.6, 0.9)
        entry = int(np.clip(215 + 25 * ability + rng.normal(0, 20), 160, 340))
        students.append({"anon_id": sid, "department": dept,
                         "entry_score": entry if rng.random() > 0.3 else np.nan})

        entry_year = int(rng.integers(2019, 2026))
        k_max = min(10, 2 * (2025 - entry_year + 1))
        n_sem = int(rng.integers(1, k_max + 1))

        past_units, past_points, carry = [], [], 0
        prev_gpa = np.nan
        for k in range(n_sem):
            lvl, sem = k // 2 + 1, k % 2 + 1
            session = f"{entry_year + k // 2}/{entry_year + k // 2 + 1}"
            cat_list = catalog[(dept, lvl, sem)]
            sem_units = sum(c["units"] for c in cat_list)

            study = float(np.clip(12 + 4 * diligence + rng.normal(0, 3), 1, 40))
            sleep = float(np.clip(sleep_mean + rng.normal(0, 0.5), 3.5, 9.5))
            workload = int(rng.random() < 0.25)
            att_mean = np.clip(78 + 7 * diligence, 40, 95)

            prev_cgpa = S.gpa(past_units, past_points) if past_units else np.nan
            sem_units_list, sem_points = [], []
            for c in cat_list:
                att = float(np.clip(rng.normal(att_mean, 8), 20, 100))
                mu = (58.5 + 8.0 * ability + aptitude[c["course_category"]]
                      + 0.45 * (study - 12) + 0.12 * (att - 75)
                      - 3.2 * (c["true_difficulty"] - 3)
                      - 0.8 * (sleep - 7) ** 2 / 2 - 3.0 * workload
                      - 0.25 * (sem_units - 18) + c["course_effect"])
                score = mu + rng.normal(0, 6)
                u = rng.random()  # deliberate real-world messiness
                if u < 0.03:
                    score -= rng.uniform(15, 30)   # illness, missed paper
                elif u < 0.05:
                    score += rng.uniform(10, 20)   # lucky or unusually easy paper
                total = float(np.clip(score, 0, 100))

                ca_max = c["ca_max"]
                exam_max = 100 - ca_max
                if ca_max > 0:
                    ca_ratio = np.clip(total / 100 + 0.08 + rng.normal(0, 0.07), 0, 1)
                    ca = round(float(ca_ratio * ca_max), 1)
                    exam = round(float(np.clip(total - ca, 0, exam_max)), 1)
                    ca_score = ca
                else:
                    ca_score, exam = np.nan, round(total, 1)
                total = round((0 if ca_score != ca_score else ca_score) + exam, 1)
                grade = S.score_to_grade(total)

                a_total = int(rng.integers(3, 9))
                a_done = int(rng.binomial(a_total, float(np.clip(att / 100 + 0.05, 0.1, 1))))
                perceived = int(np.clip(round(c["true_difficulty"] - 0.4 * ability + rng.normal(0, 0.8)), 1, 5))

                courses.append({
                    "anon_id": sid, "department": dept, "level": lvl * 100,
                    "session": session, "semester": sem,
                    "course_code": c["course_code"], "units": c["units"],
                    "course_category": c["course_category"], "course_type": c["course_type"],
                    "perceived_difficulty": perceived, "ca_max": ca_max, "exam_max": exam_max,
                    "ca_score": ca_score, "attendance_pct": round(att, 1),
                    "assignments_done": a_done, "assignments_total": a_total,
                    "exam_score": exam, "total_score": total, "grade": grade,
                    "weights_confirmed": bool(rng.random() < 0.8),
                })
                sem_units_list.append(c["units"])
                sem_points.append(S.grade_to_point(grade))

            sem_gpa = S.gpa(sem_units_list, sem_points)
            semesters.append({
                "anon_id": sid, "session": session, "semester": sem, "level": lvl * 100,
                "study_hours_per_week": round(study, 1), "sleep_hours": round(sleep, 1),
                "outside_workload": workload, "prev_gpa": prev_gpa, "prev_cgpa": prev_cgpa,
                "carryover_count": carry, "semester_units": sem_units,
                "semester_gpa": round(sem_gpa, 2),
            })
            carry += sum(1 for p in sem_points if p == 0)
            past_units += sem_units_list
            past_points += sem_points
            prev_gpa = round(sem_gpa, 2)

    return (pd.DataFrame(students)[S.STUDENT_COLS],
            pd.DataFrame(semesters)[S.SEMESTER_COLS],
            pd.DataFrame(courses)[S.COURSE_COLS])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-students", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data")
    a = ap.parse_args()

    students, semesters, courses = simulate(a.n_students, a.seed)
    issues = S.validate_courses(courses)
    if issues:
        raise SystemExit(f"Schema validation failed: {issues}")

    os.makedirs(a.out, exist_ok=True)
    students.to_csv(f"{a.out}/students.csv", index=False)
    semesters.to_csv(f"{a.out}/semesters.csv", index=False)
    courses.to_csv(f"{a.out}/courses.csv", index=False)
    print(f"students={len(students)} semesters={len(semesters)} courses={len(courses)}")
    print("grade mix (%):", (courses["grade"].value_counts(normalize=True).sort_index() * 100).round(1).to_dict())
    print("total_score mean/sd:", round(courses["total_score"].mean(), 1), round(courses["total_score"].std(), 1))
    print("semester_gpa mean/sd:", round(semesters["semester_gpa"].mean(), 2), round(semesters["semester_gpa"].std(), 2))
    print("share of courses with no CA (exam /100):", round((courses["ca_max"] == 0).mean() * 100, 1), "%")


if __name__ == "__main__":
    main()