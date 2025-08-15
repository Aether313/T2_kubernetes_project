from fastapi import FastAPI
from fastapi.responses import JSONResponse
import os, pickle, pandas as pd

app = FastAPI()
DATA_ROOT = os.environ.get("DATA_ROOT", "/data")
MODELS_DIR = os.path.join(DATA_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/train")
def train(input_path: str):
    try:
        if not os.path.exists(input_path):
            return JSONResponse({"status": "error", "msg": f"input not found: {input_path}"}, status_code=400)
        df = pd.read_csv(input_path, encoding="utf-8-sig")
        if "IMDb Score" in df.columns:
            model_obj = {"type": "constant_mean", "value": float(df["IMDb Score"].mean())}
        else:
            model_obj = {"type": "constant_count", "value": int(len(df))}
        model_path = os.path.join(MODELS_DIR, "model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model_obj, f)
        return {"status": "ok", "model_path": model_path}
    except Exception as e:
        return JSONResponse({"status": "error", "msg": str(e)}, status_code=500)
