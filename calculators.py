"""
Deterministic calculators for the Student Performance Predictor (no ML).

- projected_cgpa:   CGPA after a semester, given grade points
- required_semester_gpa: what GPA is needed to reach a target CGPA
- plan_for_target:  per-course grades and scores needed to reach a target CGPA

Course dict keys: code, units, ca_max, exam_max, ca_score (optional), difficulty (1-5, optional).
"""
import math

import schema as S

MIN_SCORE = {point: minimum for minimum, _, point in S.GRADE_SCALE}   # grade point -> min total
LETTER = {point: letter for _, letter, point in S.GRADE_SCALE}
TOP, PASS = max(MIN_SCORE), 1   # planning baseline is "pass every course" (E on the default scale)


def quality_points(units, points):
    return sum(u * p for u, p in zip(units, points))


def cgpa(past_units, past_points):
    return S.gpa(past_units, past_points)


def projected_cgpa(past_units, past_points, sem_units, sem_points):
    return S.gpa(list(past_units) + list(sem_units), list(past_points) + list(sem_points))


def required_semester_gpa(target, past_units, past_points, sem_units_total):
    """Semester GPA needed so the new CGPA equals `target`."""
    total_units = sum(past_units) + sem_units_total
    needed_qp = target * total_units - quality_points(past_units, past_points)
    return needed_qp / sem_units_total


def _known(ca):
    """CA is unknown when missing or NaN (pandas writes NaN for blanks)."""
    return ca is not None and not (isinstance(ca, float) and math.isnan(ca))


def _max_point(course):
    """Highest grade point still reachable in this course."""
    ca = course.get("ca_score")
    best_total = (ca if _known(ca) else course["ca_max"]) + course["exam_max"]
    return max(p for p, m in MIN_SCORE.items() if m <= best_total)


def _exam_needed(course, min_total, assumed_ca_ratio):
    ca = course.get("ca_score")
    if course["ca_max"] == 0:
        return max(0.0, min_total), False
    if _known(ca):
        return max(0.0, min_total - ca), False
    return max(0.0, min_total - assumed_ca_ratio * course["ca_max"]), True   # CA unknown: assumption


def plan_for_target(target, past_units, past_points, courses, assumed_ca_ratio=0.7, difficulty_weight=1.0):
    """
    Spread effort across courses to reach `target` CGPA.
    difficulty_weight: 1 balanced, 2 leans on easier courses, 0 ignores difficulty (even spread).
    Heuristic: start every course at a pass, then upgrade the grade that costs the least
    effort per quality point, where effort grows with course difficulty and with the grade
    already planned. The rising cost spreads upgrades instead of piling them on one course.
    """
    sem_units = sum(c["units"] for c in courses)
    need_gpa = required_semester_gpa(target, past_units, past_points, sem_units)
    needed_qp = need_gpa * sem_units

    points = {c["code"]: PASS for c in courses}
    caps = {c["code"]: _max_point(c) for c in courses}
    qp = sum(c["units"] * PASS for c in courses)

    while qp < needed_qp - 1e-9:
        options = []
        for c in courses:
            g = points[c["code"]]
            if g < caps[c["code"]]:
                gap = MIN_SCORE[g + 1] - MIN_SCORE[g]
                cost = (c.get("difficulty", 3) ** difficulty_weight) * gap * (1 + 0.25 * g) / c["units"]
                options.append((cost, c["code"], c["units"]))
        if not options:
            break
        _, code, units = min(options)
        points[code] += 1
        qp += units

    plan = []
    for c in courses:
        g = points[c["code"]]
        exam, assumed = _exam_needed(c, MIN_SCORE[g], assumed_ca_ratio)
        plan.append({
            "code": c["code"], "units": c["units"], "target_grade": LETTER[g],
            "min_total_score": MIN_SCORE[g], "exam_score_needed": round(exam, 1),
            "exam_max": c["exam_max"], "assumes_typical_ca": assumed,
        })

    achieved = projected_cgpa(past_units, past_points,
                              [c["units"] for c in courses], [points[c["code"]] for c in courses])
    max_cgpa = projected_cgpa(past_units, past_points,
                              [c["units"] for c in courses], [caps[c["code"]] for c in courses])
    if need_gpa <= PASS:
        status = "secured_if_you_pass_everything"
    elif qp >= needed_qp - 1e-9:
        status = "reachable"
    else:
        status = "unreachable"
    return {
        "status": status, "required_semester_gpa": round(need_gpa, 2),
        "planned_cgpa": round(achieved, 2), "max_possible_cgpa": round(max_cgpa, 2),
        "plan": plan,
    }


if __name__ == "__main__":
    past_u, past_p = [3, 3, 2, 4, 3, 2], [4, 3, 5, 4, 3, 5]   # CGPA 3.88 on 17 units
    courses = [
        {"code": "CSC 301", "units": 4, "ca_max": 30, "exam_max": 70, "difficulty": 4},
        {"code": "CSC 305", "units": 3, "ca_max": 30, "exam_max": 70, "ca_score": 24, "difficulty": 3},
        {"code": "MTH 303", "units": 3, "ca_max": 0, "exam_max": 100, "difficulty": 5},
        {"code": "EEE 311", "units": 2, "ca_max": 30, "exam_max": 70, "difficulty": 2},
        {"code": "GST 301", "units": 2, "ca_max": 0, "exam_max": 100, "difficulty": 1},
    ]
    print("CGPA now:", round(cgpa(past_u, past_p), 2))
    for t in (4.0, 4.5, 4.9):
        r = plan_for_target(t, past_u, past_p, courses)
        print(f"\nTarget {t}: {r['status']}, need semester GPA {r['required_semester_gpa']}, "
              f"planned CGPA {r['planned_cgpa']}, max possible {r['max_possible_cgpa']}")
        if t == 4.5:
            for row in r["plan"]:
                print("  ", row)