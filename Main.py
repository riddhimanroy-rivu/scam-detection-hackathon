import os
import sys
import subprocess
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Dict, Optional, List

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from common.config import DATASETS
from common.url_features import extract_features, FEATURE_NAMES as URL_FEATURE_NAMES
from explainers.explain_util import (
    lime_explain_text, anchor_explain_text,
    
    lime_explain_tabular, anchor_explain_tabular,
)

app = FastAPI(title="Scam & Fraud Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_MODEL_CACHE: Dict[str, dict] = {}


def _load(key: str) -> dict:
    if key not in DATASETS:
        raise HTTPException(404, f"Unknown detector '{key}'")
    cfg = DATASETS[key]
    if key not in _MODEL_CACHE:
        if not os.path.exists(cfg["model_file"]):
            raise HTTPException(
                409,
                f"'{key}' model is not trained yet. Put its CSV in backend/data/ "
                f"and call POST /train, or run `python -m models.train_all {key}`.",
            )
        _MODEL_CACHE[key] = joblib.load(cfg["model_file"])
    return _MODEL_CACHE[key]


# --------------------------------------------------------------- status ---

@app.get("/api/status")
def status():
    out = {}
    for key, cfg in DATASETS.items():
        out[key] = {
            "dataset_present": os.path.exists(cfg["path"]),
            "dataset_path": cfg["path"],
            "model_trained": os.path.exists(cfg["model_file"]),
        }
    return out


class TrainRequest(BaseModel):
    targets: Optional[List[str]] = None


@app.post("/api/train")
def train(req: TrainRequest = TrainRequest()):
    targets = req.targets or list(DATASETS.keys())
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    result = subprocess.run(
        [sys.executable, "-m", "models.train_all", *targets],
        cwd=backend_dir, capture_output=True, text=True, timeout=3600,
    )
    for t in targets:
        _MODEL_CACHE.pop(t, None)
    return {
        "returncode": result.returncode,
        "log": result.stdout[-6000:],
        "errors": result.stderr[-3000:],
    }


# ------------------------------------------------------------------ SMS ---

class SmsRequest(BaseModel):
    text: str


@app.post("/api/predict/sms")
def predict_sms(req: SmsRequest):
    bundle = _load("sms")
    pipeline = bundle["pipeline"]
    class_names = bundle["class_names"]

    proba = pipeline.predict_proba([req.text])[0]
    pred_idx = int(np.argmax(proba))

    return {
        "prediction": class_names[pred_idx],
        "confidence": round(float(proba[pred_idx]), 4),
        "probabilities": {c: round(float(p), 4) for c, p in zip(class_names, proba)},
        "lime": lime_explain_text(pipeline, class_names, req.text),
        "anchor": anchor_explain_text(pipeline, class_names, req.text),
    }


# ------------------------------------------------------------------ URL ---

class UrlRequest(BaseModel):
    url: str


@app.post("/api/predict/url")
def predict_url(req: UrlRequest):
    bundle = _load("url")
    model = bundle["model"]
    feature_names = bundle["feature_names"]
    class_names = bundle["class_names"]
    train_sample = bundle["train_sample"]

    feats = extract_features(req.url)
    row = np.array([feats[f] for f in feature_names], dtype=float)

    proba = model.predict_proba([row])[0]
    pred_idx = int(np.argmax(proba))

    return {
        "prediction": class_names[pred_idx],
        "confidence": round(float(proba[pred_idx]), 4),
        "probabilities": {c: round(float(p), 4) for c, p in zip(class_names, proba)},
        "extracted_features": feats,
        "lime": lime_explain_tabular(model, feature_names, class_names, train_sample, row),
        "anchor": anchor_explain_tabular(model, feature_names, class_names, train_sample, row),
    }


# -------------------------------------------------------- bank / card ----

class TabularRequest(BaseModel):
    features: Dict[str, float]


def _predict_tabular(key: str, req: TabularRequest):
    bundle = _load(key)
    model = bundle["model"]
    feature_names = bundle["feature_names"]
    class_names = bundle["class_names"]
    train_sample = bundle["train_sample"]

    row = np.array(
        [float(req.features.get(f, 0.0)) for f in feature_names], dtype=float
    )

    proba = model.predict_proba([row])[0]
    pred_idx = int(np.argmax(proba))

    return {
        "prediction": class_names[pred_idx],
        "confidence": round(float(proba[pred_idx]), 4),
        "probabilities": {c: round(float(p), 4) for c, p in zip(class_names, proba)},
        "lime": lime_explain_tabular(model, feature_names, class_names, train_sample, row),
        "anchor": anchor_explain_tabular(model, feature_names, class_names, train_sample, row),
    }


@app.get("/api/schema/{key}")
def schema(key: str):
    bundle = _load(key)
    train_sample = bundle["train_sample"]
    return {
        "feature_names": bundle["feature_names"],
        "class_names": bundle["class_names"],
        "example_values": {
            f: round(float(train_sample[f].median()), 4) for f in bundle["feature_names"]
        },
    }


@app.post("/api/predict/bank")
def predict_bank(req: TabularRequest):
    return _predict_tabular("bank", req)


@app.post("/api/predict/creditcard")
def predict_creditcard(req: TabularRequest):
    return _predict_tabular("creditcard", req)


@app.get("/api/url-feature-names")
def url_feature_names():
    return {"feature_names": URL_FEATURE_NAMES}


# ------------------------------------------------------------- frontend ---

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")

if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/")
    def index():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))