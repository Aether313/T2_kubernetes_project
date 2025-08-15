from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, List
import os
import pandas as pd
import tensorflow as tf  # for loading .h5 models

app = FastAPI()
DATA_ROOT = os.environ.get("DATA_ROOT", "/data")
MODEL_PATH = os.path.join(DATA_ROOT, "models", "model.h5")  # point to .h5

@app.get("/healthz")
def healthz():
    return {"ok": True}

class PredictPayload(BaseModel):
    input_path: Optional[str] = None
    row: Optional[Dict] = None

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"model not found: {MODEL_PATH}")
    return tf.keras.models.load_model(MODEL_PATH)

def predict_value(model_obj, row: Dict | None = None) -> float:
    # Convert row dict to DataFrame, ensure consistent shape
    if row is None:
        raise ValueError("No row data provided for prediction")
    df = pd.DataFrame([row])
    pred = model_obj.predict(df)
    return float(pred[0][0])  # adjust if model outputs differently

@app.post("/predict")
def predict(payload: PredictPayload):
    try:
        model = load_model()
        if payload.row is not None:
            y = predict_value(model, payload.row)
            return {"status": "ok", "n": 1, "predictions": [y]}
        if payload.input_path:
            if not os.path.exists(payload.input_path):
                return JSONResponse({"status": "error", "msg": f"input not found: {payload.input_path}"}, status_code=400)
            df = pd.read_csv(payload.input_path, encoding="utf-8-sig")
            preds: List[float] = [float(p[0]) for p in model.predict(df)]
            return {"status": "ok", "n": int(len(df)), "predictions": preds}
        return JSONResponse({"status": "error", "msg": "provide 'row' or 'input_path'"}, status_code=400)
    except Exception as e:
        return JSONResponse({"status": "error", "msg": str(e)}, status_code=500)

