"""PNCare-AI REST API.

Endpoints
  GET  /                 short description
  GET  /health           model status
  GET  /api/inputs       the 12 inputs and their allowed values
  POST /api/predict      one mother: risk, risk level, WHO-referenced actions and SHAP reasons
  POST /api/predict/batch  up to 20,000 mothers (JSON): risk and risk level for each
  POST /api/upload       CSV file with up to 20,000 mothers: risk and risk level for each
  POST /api/preprocess   checks the inputs and returns the 22 selected features the model uses
  GET  /api/model/{version}  model information (use "latest" or the version number)
Interactive documentation: /docs
"""
from typing import List

import io

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

import engine

app = FastAPI(
    title="PNCare-AI API",
    version=engine.APP_VERSION,
    description=("Predicts the risk that a mother in Bangladesh will not receive a postnatal check from a medically "
                 "trained provider within 48 hours of birth (soft-voting model, BDHS 2017-18 and 2022). "
                 "Decision support only. Inputs are processed in memory and are not stored."),
)


class Mother(BaseModel):
    place_of_delivery: str = Field(..., examples=["Private hospital"])
    anc_visits: float = Field(..., ge=0, le=20, examples=[5])
    mother_age_at_birth: float = Field(..., ge=13, le=49, examples=[26])
    education: str = Field(..., examples=["Secondary or higher"])
    division: str = Field(..., examples=["Dhaka"])
    residence: str = Field(..., examples=["Urban"])
    newspaper: str = Field(..., examples=["Not at all"])
    radio: str = Field(..., examples=["Not at all"])
    tv: str = Field(..., examples=["At least weekly"])
    poor_household: str = Field(..., examples=["No"])
    birth_order_4_plus: str = Field(..., examples=["No"])
    distance_big_problem: str = Field(..., examples=["No"])


class Batch(BaseModel):
    mothers: List[Mother] = Field(..., max_length=20000)


def _parse(m: Mother):
    try:
        return engine.parse_inputs(m.model_dump())
    except ValueError as err:
        raise HTTPException(status_code=422, detail=str(err))


@app.get("/")
def root():
    return {"name": "PNCare-AI API", "version": engine.APP_VERSION, "docs": "/docs",
            "endpoints": ["/health", "/api/inputs", "/api/predict", "/api/predict/batch", "/api/upload",
                          "/api/preprocess", "/api/model/latest"]}


@app.get("/health")
def health():
    return {"status": "ok", "model": "soft voting (LR + LightGBM + XGBoost + RF)", "threshold": engine.THRESHOLD,
            "survey_year": engine.YEAR, "features": len(engine.FEATURE_NAMES)}


@app.get("/api/inputs")
def inputs():
    out = []
    for s in engine.INPUTS:
        d = {"name": s["csv"], "label": s["label"], "type": s["kind"]}
        if s["kind"] == "cat":
            d["allowed"] = list(s["options"])
        elif s["kind"] == "bool":
            d["allowed"] = ["Yes", "No"]
        else:
            d["min"], d["max"] = s["min"], s["max"]
        out.append(d)
    return out


@app.post("/api/predict")
def predict_one(mother: Mother):
    return engine.assess(_parse(mother), with_explanation=True)


@app.post("/api/predict/batch")
def predict_batch(batch: Batch):
    import pandas as pd
    xs = [_parse(m) for m in batch.mothers]
    risks = engine.predict(pd.DataFrame(xs))
    return {"n": len(xs), "threshold": engine.THRESHOLD,
            "results": [{"row": i + 1, "risk": round(float(r), 4), "risk_percent": round(float(r) * 100, 1),
                         "risk_level": "high" if r >= engine.THRESHOLD else "low"} for i, r in enumerate(risks)]}


TEST_METRICS = {"roc_auc": 0.9405, "pr_auc": 0.9271, "brier": 0.0822, "accuracy_at_threshold": 0.8760,
                "recall_at_threshold": 0.8778, "specificity_at_threshold": 0.8745, "test_mothers": 1460}


@app.post("/api/preprocess")
def preprocess(mother: Mother):
    x = _parse(mother)
    return {"inputs": {s["csv"]: x[s["key"]] for s in engine.INPUTS},
            "vulnerability_score": engine.vulnerability(x), "media_exposure_index": engine.media_index(x),
            "selected_features": engine.selected_features(x)}


@app.get("/api/model/{version}")
def model_info(version: str):
    if version not in ("latest", engine.APP_VERSION):
        raise HTTPException(status_code=404, detail=f"Model version '{version}' not found; available: {engine.APP_VERSION}")
    return {"version": engine.APP_VERSION,
            "model": "soft voting of logistic regression, LightGBM, XGBoost and random forest (Optuna-tuned)",
            "data": "BDHS 2017-18 and 2022, 9,733 mothers; train/validation/test 70/15/15",
            "outcome": "no postnatal check by a medically trained provider within 48 hours of birth",
            "threshold": engine.THRESHOLD, "survey_year_used": engine.YEAR,
            "selected_features": list(engine.FEATURE_NAMES), "test_metrics": TEST_METRICS}


@app.post("/api/upload")
async def upload(file: UploadFile = File(..., description="CSV with the 12 input columns, one row per mother")):
    import pandas as pd
    try:
        raw = pd.read_csv(io.BytesIO(await file.read()))
    except Exception:
        raise HTTPException(status_code=400, detail="The file could not be read as CSV.")
    raw.columns = [c.strip().lower() for c in raw.columns]
    missing = [s["csv"] for s in engine.INPUTS if s["csv"] not in raw.columns]
    if missing:
        raise HTTPException(status_code=422, detail="Missing columns: " + ", ".join(missing))
    if len(raw) > 20000:
        raise HTTPException(status_code=422, detail="Please upload at most 20,000 rows at a time.")
    xs, bad = [], []
    for i, row in enumerate(raw.to_dict("records")):
        try:
            xs.append((i + 1, engine.parse_inputs(row)))
        except ValueError as err:
            bad.append({"row": i + 1, "error": str(err)})
    results = []
    if xs:
        risks = engine.predict(pd.DataFrame([x for _, x in xs]))
        results = [{"row": r, "risk": round(float(p), 4), "risk_percent": round(float(p) * 100, 1),
                    "risk_level": "high" if p >= engine.THRESHOLD else "low"} for (r, _), p in zip(xs, risks)]
    return {"rows": len(raw), "scored": len(results), "invalid_rows": bad, "threshold": engine.THRESHOLD,
            "results": results}
