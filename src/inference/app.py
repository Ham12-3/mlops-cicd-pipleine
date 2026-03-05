"""
Minimal inference API: /health and /predict.
Run with: uvicorn src.inference.app:app --host 0.0.0.0 --port 8000
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

# Model loaded at startup if MODEL_PATH is set
_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model once at startup when MODEL_PATH is set."""
    global _model
    path = os.environ.get("MODEL_PATH")
    if path and os.path.isfile(path):
        from src.model import load_model

        _model = load_model(path)
    yield
    _model = None


app = FastAPI(title="ML Inference API", lifespan=lifespan)


class PredictRequest(BaseModel):
    """Input for /predict: list of feature vectors (list of lists)."""

    features: list[list[float]]


class PredictResponse(BaseModel):
    """Predictions returned from /predict."""

    predictions: list[int]


@app.get("/health")
def health() -> dict:
    """Liveness/readiness: OK if app is up; model_loaded indicates if MODEL_PATH was set."""
    return {
        "status": "ok",
        "model_loaded": _model is not None,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    """Run model prediction on the provided feature vectors."""
    if _model is None:
        from fastapi import HTTPException

        raise HTTPException(status_code=503, detail="Model not loaded; set MODEL_PATH")
    import numpy as np

    X = np.array(req.features, dtype=np.float64)
    preds = _model.predict(X)
    return PredictResponse(predictions=preds.tolist())
