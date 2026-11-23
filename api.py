"""
FastAPI backend for the Student Performance Predictor.

Run:   uvicorn api:app --reload
Docs:  http://127.0.0.1:8000/docs   (try the endpoints in the browser)
"""
import os
from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

import advice as A
import calculators as C
import predictor as P
import studyplan as SP
import schema as S

Category = Literal["calculation", "applied", "writing", "theory", "practical", "general"]
CourseType = Literal["core", "elective", "general"]
Grade = Literal["A", "B", "C", "D", "E", "F"]

app = FastAPI(title="Student Performance Predictor", version="0.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class HistCourse(BaseModel):
    units: int = Field(gt=0, le=6)
    category: Category
    grade: Grade


class CourseIn(BaseModel):
    code: str
    units: int = Field(gt=0, le=6)
    category: Category
    course_type: CourseType = "core"
    difficulty: int = Field(3, ge=1, le=5)
    ca_max: int = Field(30, ge=0, le=60, description="Marks for CA. 0 means exam is out of 100.")
    ca_score: Optional[float] = Field(None, ge=0, description="CA score so far, if known")
    attendance_pct: Optional[float] = Field(None, ge=0, le=100)
    assignments_done: Optional[int] = Field(None, ge=0)
    assignments_total: Optional[int] = Field(None, ge=0, description="0 means no assignments")

    @property
    def exam_max(self):
        return 100 - self.ca_max

    @model_validator(mode="after")
    def check_ca(self):
        if self.ca_score is not None and self.ca_score > self.ca_max:
            raise ValueError(f"{self.code}: ca_score cannot exceed ca_max")
        return self


class StudentIn(BaseModel):
    department: str
    level: Literal[100, 200, 300, 400, 500]
    entry_score: Optional[int] = Field(None, ge=0, le=400)
    study_hours: float = Field(ge=0, le=80, description="Hours per week")
    sleep_hours: float = Field(ge=2, le=14, description="Hours per night")
    outside_workload: bool = False
    expected_attendance_pct: float = Field(80, ge=0, le=100)
    history: List[List[HistCourse]] = Field(default_factory=list, description="Past semesters, oldest first")


class PredictRequest(BaseModel):
    student: StudentIn
    courses: List[CourseIn] = Field(min_length=1)


class TargetRequest(PredictRequest):
    target_cgpa: float = Field(gt=0, le=5)


def _student_dict(s: StudentIn):
    d = s.model_dump()
    d["history"] = [[c for c in sem] for sem in d["history"]]
    return d


def _course_dicts(courses: List[CourseIn]):
    return [c.model_dump() for c in courses]


@app.get("/health")
def health():
    return {"status": "ok", "grade_scale": S.GRADE_SCALE}


@app.post("/predict")
def predict(req: PredictRequest):
    student, courses = _student_dict(req.student), _course_dicts(req.courses)
    result = P.predict_semester(student, courses)
    result["advice"] = A.build_advice(student, courses, result)
    return result


@app.post("/target")
def target(req: TargetRequest):
    hf = P.history_features(_student_dict(req.student)["history"])
    planner_courses = [{
        "code": c.code, "units": c.units, "ca_max": c.ca_max, "exam_max": c.exam_max,
        "ca_score": c.ca_score, "difficulty": c.difficulty,
    } for c in req.courses]
    styles = [("Balanced", "Effort is spread by how hard each course is.", 1.0),
              ("Lean on easier courses", "Push grades higher where courses are easier for you.", 2.0),
              ("Even spread", "Aim for similar grades across all courses.", 0.0)]
    options = []
    for label, blurb, dw in styles:
        p = C.plan_for_target(req.target_cgpa, hf["past_units"], hf["past_points"], planner_courses, difficulty_weight=dw)
        p.update(label=label, blurb=blurb, advice=A.target_advice(p, req.target_cgpa))
        options.append(p)
    out = {"options": options}
    if not hf["past_units"]:
        out["note"] = "No past results entered, so the target is treated as this semester's GPA."
    return out


class StudyPlanRequest(PredictRequest):
    target_cgpa: Optional[float] = Field(None, gt=0, le=5)
    weekly_hours: Optional[float] = Field(None, gt=0, le=80)


@app.post("/studyplan")
def studyplan(req: StudyPlanRequest):
    student, courses = _student_dict(req.student), _course_dicts(req.courses)
    result = P.predict_semester(student, courses)
    advice = A.build_advice(student, courses, result)
    targets = None
    if req.target_cgpa:
        hf = P.history_features(student["history"])
        pc = [{"code": c.code, "units": c.units, "ca_max": c.ca_max, "exam_max": c.exam_max,
               "ca_score": c.ca_score, "difficulty": c.difficulty} for c in req.courses]
        plan = C.plan_for_target(req.target_cgpa, hf["past_units"], hf["past_points"], pc)
        targets = {p["code"]: p["target_grade"] for p in plan["plan"]}
    return SP.build_plan(result, courses, advice, targets, req.weekly_hours or req.student.study_hours)


# Serve the web app from the same server (must stay last, after all API routes)
_WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
if os.path.isdir(_WEB):
    app.mount("/", StaticFiles(directory=_WEB, html=True), name="web")