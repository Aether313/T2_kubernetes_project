from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import os, uuid, pandas as pd
from clean import clean_df

app = FastAPI(title="preprocess-service", version="1.0.0")
DATA_ROOT = os.getenv("DATA_ROOT", "/data")
RAW_DIR = os.path.join(DATA_ROOT, "raw")
PROC_DIR = os.path.join(DATA_ROOT, "processed")
os.makedirs(RAW_DIR, exist_ok=True); os.makedirs(PROC_DIR, exist_ok=True)

@app.post("/clean")
async def clean_endpoint(file: UploadFile = File(...), output_name: str = Form(None)):
    raw_name = output_name or f"{uuid.uuid4()}_{file.filename}"
    raw_path = os.path.join(RAW_DIR, raw_name)
    with open(raw_path, "wb") as f: f.write(await file.read())

    df = pd.read_csv(raw_path, encoding="utf-8-sig")
    df_cleaned = clean_df(df)
    cleaned_name = raw_name.replace(".csv","") + "_cleaned.csv"
    cleaned_path = os.path.join(PROC_DIR, cleaned_name)
    df_cleaned.to_csv(cleaned_path, index=False, encoding="utf-8-sig")
    return JSONResponse({"status":"ok","cleaned_path": cleaned_path, "rows": int(df_cleaned.shape[0])})

@app.get("/healthz")
def healthz(): return {"ok": True}
