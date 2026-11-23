"""
Weekly study plan: splits the student's weekly study hours across courses by need,
then lays the sessions out over Monday to Saturday (Sunday stays free).

Need per course = units x (base + risk + gap to target grade) x difficulty/3.
Higher-unit, riskier, harder courses, and courses where the target grade is above the
predicted grade, get more hours. Sessions are at most 90 minutes, and the same course
is kept on different days where possible.
"""
import schema as S

RISK_NEED = {"At-Risk": 1.0, "Watch": 0.5, "Safe": 0.15}
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
FOCUS = {
    "ca_gap": "Plan revision around the exam score you need",
    "exam_gap": "Past questions under timed conditions",
    "assignments": "Finish pending assignments",
    "attendance": "Catch up on missed lectures",
    "history": "Revisit fundamentals and ask your lecturer",
    "general_risk": "Revise notes and practise early",
}


def build_plan(result, courses, advice, targets, hours):
    rows = []
    for c, r in zip(courses, result["courses"]):
        gap = 0
        if targets:
            gap = max(0, S.grade_to_point(targets[c["code"]]) - S.grade_to_point(r["predicted_grade"]))
        weight = c["units"] * (0.3 + RISK_NEED[r["risk"]] + 0.5 * gap) * (c.get("difficulty", 3) / 3)
        kinds = [a["type"] for a in advice["courses"].get(c["code"], [])]
        focus = next((FOCUS[k] for k in kinds if k in FOCUS), "Review notes and practise problems")
        rows.append({"code": c["code"], "units": c["units"], "risk": r["risk"], "weight": weight,
                     "focus": focus, "target_grade": targets.get(c["code"]) if targets else None})

    total = sum(x["weight"] for x in rows)
    raw = [hours * x["weight"] / total * 2 for x in rows]          # in half-hours
    halves = [int(v) for v in raw]
    spare = int(round(hours * 2)) - sum(halves)
    for i in sorted(range(len(rows)), key=lambda i: -(raw[i] - halves[i]))[:max(spare, 0)]:
        halves[i] += 1
    for x, h in zip(rows, halves):
        x["hours"] = h / 2

    sessions = []
    for x in rows:
        left = x["hours"]
        while left > 0:
            s = min(1.5, left)
            sessions.append((x["code"], int(s * 60), x["focus"]))
            left -= s

    days = {d: [] for d in DAYS}
    load = {d: 0 for d in DAYS}
    for code, mins, focus in sorted(sessions, key=lambda s: -s[1]):
        free = [d for d in DAYS if all(s["code"] != code for s in days[d])] or DAYS
        d = min(free, key=lambda d: load[d])
        days[d].append({"code": code, "minutes": mins, "focus": focus})
        load[d] += mins

    return {
        "weekly_hours": hours,
        "courses": [{k: x[k] for k in ("code", "hours", "risk", "focus", "target_grade")} for x in rows],
        "days": [{"day": d, "minutes": load[d], "sessions": days[d]} for d in DAYS],
        "note": "Hours follow risk, difficulty, units" + (" and the gap to your target" if targets else "") +
                ". It updates as you log CA scores and attendance.",
    }
