"""
Shared schema for the Student Performance Predictor.

Synthetic data and real data (lecturer sheets, student survey) must both produce
these exact columns, so retraining on real records is a drop-in swap.
"""
import pandas as pd

# ---------------------------------------------------------------- grading scale
# Default Nigerian 5.0 scale. Boundaries are config, not code: change here only.
GRADE_SCALE = [  # (minimum total score, letter, grade point)
    (70, "A", 5),
    (60, "B", 4),
    (50, "C", 3),
    (45, "D", 2),
    (40, "E", 1),
    (0, "F", 0),
]


def score_to_grade(score: float) -> str:
    for minimum, letter, _ in GRADE_SCALE:
        if score >= minimum:
            return letter
    return "F"


# Representative score for each letter. The app collects past results as grades, not
# exact scores, so history features are built from these values in training AND in the app.
GRADE_REP = {"A": 80, "B": 65, "C": 55, "D": 47, "E": 42, "F": 20}


def grade_to_point(letter: str) -> int:
    return {g: p for _, g, p in GRADE_SCALE}[letter]


def gpa(units, points) -> float:
    """Unit-weighted GPA. Deterministic, no ML involved."""
    total_units = sum(units)
    if total_units == 0:
        return float("nan")
    return sum(u * p for u, p in zip(units, points)) / total_units


# ---------------------------------------------------------------- categories
CATEGORIES = ["calculation", "applied", "writing", "theory", "practical", "general"]
CATEGORY_LABELS = {"calculation": "calculation-based", "applied": "applied or technical", "writing": "writing-based",
                   "theory": "theory-heavy", "practical": "practical or lab", "general": "general studies"}
COURSE_TYPES = ["core", "elective", "general"]

# ---------------------------------------------------------------- tables
STUDENT_COLS = ["anon_id", "department", "entry_score"]

SEMESTER_COLS = [
    "anon_id", "session", "semester", "level",
    "study_hours_per_week", "sleep_hours", "outside_workload",
    "prev_gpa", "prev_cgpa", "carryover_count",
    "semester_units", "semester_gpa",
]

COURSE_COLS = [
    "anon_id", "department", "level", "session", "semester",
    "course_code", "units", "course_category", "course_type",
    "perceived_difficulty", "ca_max", "exam_max",
    "ca_score", "attendance_pct", "assignments_done", "assignments_total",
    "exam_score", "total_score", "grade", "weights_confirmed",
]


def validate_courses(df: pd.DataFrame) -> list:
    """Return a list of problems. Empty list means the table is valid.
    Run this on real data too, before training."""
    issues = []
    missing = [c for c in COURSE_COLS if c not in df.columns]
    if missing:
        return [f"missing columns: {missing}"]

    if not (df["ca_max"] + df["exam_max"] == 100).all():
        issues.append("ca_max + exam_max must equal 100 on every row")
    has_ca = df["ca_max"] > 0
    if (df.loc[has_ca, "ca_score"] > df.loc[has_ca, "ca_max"] + 1e-6).any():
        issues.append("ca_score exceeds ca_max")
    if (df["exam_score"] > df["exam_max"] + 1e-6).any():
        issues.append("exam_score exceeds exam_max")
    if ((df["attendance_pct"] < 0) | (df["attendance_pct"] > 100)).any():
        issues.append("attendance_pct outside 0-100")
    if not df["perceived_difficulty"].dropna().between(1, 5).all():
        issues.append("perceived_difficulty outside 1-5")
    if (df["units"] <= 0).any():
        issues.append("units must be positive")
    if not df["course_category"].isin(CATEGORIES).all():
        issues.append("unknown course_category value")
    if not df["course_type"].isin(COURSE_TYPES).all():
        issues.append("unknown course_type value")
    expected = df["total_score"].apply(score_to_grade)
    if (expected != df["grade"]).any():
        issues.append("grade does not match total_score on the configured scale")
    return issues