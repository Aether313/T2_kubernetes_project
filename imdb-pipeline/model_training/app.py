from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os, threading, time

app = FastAPI()
templates = Jinja2Templates(directory="templates")

DATA_ROOT = os.environ.get("DATA_ROOT", "/data")
MODEL_PATH = os.path.join(DATA_ROOT, "models", "model.h5")

training_progress = {"epoch": 0, "total": 0, "loss": None, "done": False}

def train_model(X, y, epochs=10):
    global training_progress
    training_progress.update({"epoch": 0, "total": epochs, "done": False})

    model = keras.Sequential([
        layers.Dense(64, activation="relu", input_shape=(X.shape[1],)),
        layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")

    for epoch in range(epochs):
        history = model.fit(X, y, epochs=1, verbose=0)
        loss = history.history["loss"][-1]
        training_progress.update({"epoch": epoch+1, "loss": float(loss)})
        time.sleep(0.5)  

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    training_progress["done"] = True


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/train", response_class=HTMLResponse)
async def get_train(request: Request):
    return templates.TemplateResponse("train.html", {"request": request})


@app.post("/train")
async def post_train(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)

    target_candidates = ["IMDb Score", "target", "score", "rating"]
    target_col = next((c for c in target_candidates if c in df.columns), df.columns[-1])

    X = df.drop(columns=[target_col]).select_dtypes(include=["number"]).fillna(0)
    y = df[target_col]

    threading.Thread(target=train_model, args=(X, y, 10)).start()
    return {"message": "Training started"}


@app.get("/progress")
async def progress():
    return training_progress
