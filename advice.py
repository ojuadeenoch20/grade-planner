"""
Advice engine: turns a prediction into specific, number-backed advice.

Rules are plain code so each message is predictable and testable. "Levers" come from
re-running the model with a changed habit, so they reflect what the model learned
(prototype: simulated data) rather than a hard-coded claim.
Advice is guidance, never a verdict.
"""
import copy

import schema as S
import calculators as C
import predictor as P

# Thresholds (provisional, tune on real data)
LOW_ATTENDANCE = 75
LOW_ASSIGNMENT_COMPLETION = 0.7
STRONG_CA = 0.7
WEAK_EXAM_FRACTION = 0.55
LOW_CA = 0.5
WEAK_CATEGORY_SCORE = 50          # average representative score in past courses of the category
MIN_LEVER_MARKS = 0.5             # ignore what-ifs that change the average by less than this
MAX_COURSE_ADVICE = 3

MIN_FOR = {p: m for m, _, p in S.GRADE_SCALE}   # grade point -> minimum score
PASS_MIN, C_MIN = MIN_FOR[1], MIN_FOR[3]


def _course_advice(c, r, cat_means, expected_attendance):
    out = []
    code, units = c["code"], c["units"]

    if r["stage"] == "B":
        ca, ca_max, exam_max = c["ca_score"], c["ca_max"], 100 - c["ca_max"]
        ca_frac = ca / ca_max
        exam_frac = (r["predicted_score"] - ca) / exam_max
        if ca_frac < LOW_CA:
            need_c, need_pass = C_MIN - ca, PASS_MIN - ca
            if need_c > exam_max:
                msg = (f"{code}: with {ca:g}/{ca_max} CA, a C is out of reach (it needs {need_c:g} in a "
                       f"{exam_max}-mark exam). You need {max(need_pass, 0):g}/{exam_max} to pass, so focus on securing the pass.")
            else:
                msg = (f"{code}: with {ca:g}/{ca_max} CA you need about {max(need_c, 0):g}/{exam_max} in the exam for a C "
                       f"and {max(need_pass, 0):g}/{exam_max} to pass. Plan your revision around that number.")
            out.append((1, "ca_gap", msg))
        elif ca_frac >= STRONG_CA and exam_frac < WEAK_EXAM_FRACTION:
            out.append((2, "exam_gap", f"{code}: your CA is strong ({ca:g}/{ca_max}) but the exam is where marks are at risk. "
                        "Practise past questions under timed conditions."))

    att = c.get("attendance_pct")
    if att is not None and att < LOW_ATTENDANCE:
        out.append((2, "attendance", f"{code}: attendance is {att:g}%. For a {units}-unit course, getting back above "
                    f"{LOW_ATTENDANCE}% is usually the cheapest improvement available."))

    done, total = c.get("assignments_done"), c.get("assignments_total")
    if done is not None and total and done / total < LOW_ASSIGNMENT_COMPLETION:
        out.append((2, "assignments", f"{code}: {done} of {total} assignments done. Unsubmitted work is CA marks you can still earn."))

    weak = cat_means.get(c["category"])
    if r["risk"] != "Safe" and weak is not None and weak < WEAK_CATEGORY_SCORE:
        out.append((3, "history", f"{code}: past {S.CATEGORY_LABELS[c['category']]} courses were difficult for you. "
                    "Start early, bring questions to your lecturer, and consider a study group."))

    if r["risk"] in ("At-Risk", "Watch") and not out:
        tail = "Give it extra study time early in the semester." if r["risk"] == "At-Risk" else \
               "A little extra revision early on would move it into the safe zone."
        out.append((3, "general_risk", f"{code}: flagged {r['risk']} (predicted {r['predicted_score']:g}, range "
                    f"{r['range_low']:g} to {r['range_high']:g}). {tail}"))
    if r["risk"] == "Safe":
        out.append((9, "on_track", f"{code}: on track. Keep your current routine."))

    out.sort(key=lambda x: x[0])
    return [{"priority": p, "type": t, "text": m} for p, t, m in out[:MAX_COURSE_ADVICE]]


def _avg_score(result):
    units = [c["units"] for c in result["courses"]]
    return sum(u * c["predicted_score"] for u, c in zip(units, result["courses"])) / sum(units)


def _levers(student, courses, base_result):
    """What-if: re-run the model with one habit improved; report the change in average predicted score."""
    base = _avg_score(base_result)
    trials = []

    s = copy.deepcopy(student)
    s["study_hours"] = min(40, s["study_hours"] + 5)
    if s["study_hours"] > student["study_hours"]:
        trials.append(("Study 5 more hours a week", s, courses))

    s = copy.deepcopy(student)
    s["expected_attendance_pct"] = min(95, s["expected_attendance_pct"] + 10)
    cs = copy.deepcopy(courses)
    for c in cs:
        if c.get("attendance_pct") is not None:
            c["attendance_pct"] = min(95, c["attendance_pct"] + 10)
    if s["expected_attendance_pct"] > student["expected_attendance_pct"] or cs != courses:
        trials.append(("Raise attendance by 10 points", s, cs))

    out = []
    for label, st, cs in trials:
        gain = _avg_score(P.predict_semester(st, cs)) - base
        if gain >= MIN_LEVER_MARKS:
            out.append({"lever": label, "avg_score_change": round(gain, 1)})
    return sorted(out, key=lambda x: -x["avg_score_change"])


def build_advice(student, courses, result):
    hf = P.history_features(student["history"])
    by_code = {r["code"]: r for r in result["courses"]}
    per_course = {c["code"]: _course_advice(c, by_code[c["code"]], hf["cat_means"], student["expected_attendance_pct"])
                  for c in courses}

    sem, info = [], result["semester"]
    risky = [r for r in result["courses"] if r["risk"] == "At-Risk"]
    if risky:
        order = sorted(risky, key=lambda r: -r["units"])
        names = ", ".join(f"{r['code']} ({r['units']} units)" for r in order)
        sem.append({"priority": 1, "type": "priorities",
                    "text": f"Put your study time into these first: {names}."
                            + (" Higher-unit courses move your GPA the most." if len(order) > 1 else "")})
    if info["heavy_load"]:
        sem.append({"priority": 2, "type": "load",
                    "text": f"{info['units']} units is a heavy load (above {P.HEAVY_LOAD_UNITS}). "
                            "Plan your week around the At-Risk and highest-unit courses and protect your sleep."})
    if info["current_cgpa"] is not None and info["predicted_cgpa"] < info["current_cgpa"] - 0.05:
        sem.append({"priority": 2, "type": "cgpa_dip",
                    "text": f"At the current plan your CGPA is predicted to move from {info['current_cgpa']} to "
                            f"{info['predicted_cgpa']}. The levers below show what can change that."})
    if student["expected_attendance_pct"] < LOW_ATTENDANCE:
        sem.append({"priority": 2, "type": "attendance",
                    "text": f"You expect to attend about {student['expected_attendance_pct']:g}% of classes. "
                            "Attendance often carries CA marks and is the easiest thing to improve."})

    levers = _levers(student, courses, result)
    if levers:
        top = levers[0]
        sem.append({"priority": 3, "type": "lever",
                    "text": f"Biggest lever in the prototype model: {top['lever'].lower()} raises your predicted "
                            f"average score by about {top['avg_score_change']:g} marks."})
    sem.sort(key=lambda x: x["priority"])
    return {"semester": sem, "courses": per_course, "levers": levers}


def target_advice(plan, target):
    out = []
    if plan["status"] == "unreachable":
        out.append(f"A CGPA of {target:g} is not reachable this semester. The best possible is {plan['max_possible_cgpa']:g}. "
                   "Aim for that, or spread the goal across two semesters.")
    elif plan["status"] == "secured_if_you_pass_everything":
        out.append("This target is already secured as long as you pass every course.")
    else:
        hard = sorted((p for p in plan["plan"] if p["exam_max"] and p["exam_score_needed"] / p["exam_max"] >= 0.6),
                      key=lambda p: -p["exam_score_needed"] / p["exam_max"])
        if hard:
            names = ", ".join(f"{p['code']} ({p['exam_score_needed']:g}/{p['exam_max']})" for p in hard[:3])
            out.append(f"The biggest asks in this plan: {names}. Start preparing for these first.")
        else:
            out.append("This target is within reach with steady work across all your courses.")
    if any(p["assumes_typical_ca"] for p in plan["plan"]):
        out.append("Some exam targets assume a typical CA score. They will update as you log real CA results.")
    return out