from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict
import os, uuid, io
import pandas as pd
import requests

from clean import clean_df

app = FastAPI()

DATA_ROOT = os.environ.get("DATA_ROOT", "/data")
PROCESSED_DIR = os.path.join(DATA_ROOT, "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)


def sniff_delimiter(first_kb: bytes, default: str = ",") -> str:
    head = first_kb.decode("utf-8", errors="ignore")
    if "|" in head and ("," not in head or head.count("|") >= head.count(",")):
        return "|"
    return default


def get_urls() -> Dict[str, Optional[str]]:
    return {
        "TRAINING_URL": os.environ.get("TRAINING_URL"),
        "INFERENCE_URL": os.environ.get("INFERENCE_URL"),
    }


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/readyz")
def readyz():
    urls = get_urls()
    return {"ok": True, **urls}


@app.get("/config")
def config():
    urls = get_urls()
    return {"data_root": DATA_ROOT, "processed_dir": PROCESSED_DIR, **urls}


@app.post("/clean")
async def clean_endpoint(
    file: UploadFile = File(...),
    delimiter: Optional[str] = Form(None),
    output_name: Optional[str] = Form(None),
):
    try:
        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="empty file")
        sep = delimiter or sniff_delimiter(raw[:2048])
        df = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig", sep=sep)
        df_cleaned = clean_df(df)

        stem_base = output_name or f"{uuid.uuid4()}_{os.path.splitext(file.filename or 'data')[0]}_cleaned.csv"
        stem = stem_base if stem_base.endswith(".csv") else f"{stem_base}.csv"
        out_path = os.path.join(PROCESSED_DIR, stem)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df_cleaned.to_csv(out_path, index=False, encoding="utf-8-sig")

        return JSONResponse(
            {
                "status": "ok",
                "cleaned_path": out_path,
                "filename": os.path.basename(out_path),
                "rows": int(df_cleaned.shape[0]),
                "delimiter_used": sep,
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse({"status": "error", "msg": str(e)}, status_code=500)


class TrainRequest(BaseModel):
    cleaned_path: str


@app.post("/train")
def train_proxy(req: TrainRequest):
    urls = get_urls()
    if not urls["TRAINING_URL"]:
        return JSONResponse({"status": "error", "msg": "TRAINING_URL not set"}, status_code=500)
    try:
        if not req.cleaned_path.startswith(DATA_ROOT):
            return JSONResponse({"status": "error", "msg": "cleaned_path must be under DATA_ROOT"}, status_code=400)
        # Important: training service expects query param ?input_path=...
        r = requests.post(urls["TRAINING_URL"], params={"input_path": req.cleaned_path}, timeout=600)
        return JSONResponse(r.json(), status_code=r.status_code)
    except Exception as e:
        return JSONResponse({"status": "error", "msg": str(e)}, status_code=500)


class PredictRequest(BaseModel):
    cleaned_path: Optional[str] = None
    row: Optional[Dict] = None


@app.post("/predict")
def predict_proxy(req: PredictRequest):
    urls = get_urls()
    if not urls["INFERENCE_URL"]:
        return JSONResponse({"status": "error", "msg": "INFERENCE_URL not set"}, status_code=500)
    try:
        payload: Dict = {}
        if req.cleaned_path:
            if not req.cleaned_path.startswith(DATA_ROOT):
                return JSONResponse({"status": "error", "msg": "cleaned_path must be under DATA_ROOT"}, status_code=400)
            payload["input_path"] = req.cleaned_path
        if req.row:
            payload["row"] = req.row
        if not payload:
            return JSONResponse({"status": "error", "msg": "provide 'cleaned_path' or 'row'"}, status_code=400)
        r = requests.post(urls["INFERENCE_URL"], json=payload, timeout=120)
        return JSONResponse(r.json(), status_code=r.status_code)
    except Exception as e:
        return JSONResponse({"status": "error", "msg": str(e)}, status_code=500)
