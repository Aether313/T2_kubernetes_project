from flask import Flask, render_template, request
import requests
import os
import pandas as pd
import tensorflow as tf
import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ---------------------------
# Flask App for UI
# ---------------------------
app = Flask(__name__)

INFERENCE_URL = os.getenv("INFERENCE_URL", "http://inference-svc/predict")
TRAIN_URL = os.getenv("TRAIN_URL", "http://training-svc/train")

@app.route('/')
def predict_page():
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        imdb_votes = int(request.form.get("IMDb Votes", 0))
    except ValueError:
        imdb_votes = None

    try:
        runtime = int(request.form.get("Runtime", 0))
    except ValueError:
        runtime = None

    try:
        hidden_gem_score = float(request.form.get("Hidden Gem Score", 0))
    except ValueError:
        hidden_gem_score = None

    try:
        boxoffice = int(request.form.get("Boxoffice", 0))
    except ValueError:
        boxoffice = None

    payload = {
        "data": {
            "IMDb Votes": imdb_votes,
            "Runtime": runtime,
            "Hidden Gem Score": hidden_gem_score,
            "Boxoffice": boxoffice,
            "Genre": request.form.get("Genre", ""),
            "Language": request.form.get("Language", ""),
            "Director": request.form.get("Director", ""),
            "Writer": request.form.get("Writer", ""),
            "Production House": request.form.get("Production House", "")
        }
    }

    try:
        resp = requests.post(INFERENCE_URL, json=payload)
        if resp.status_code == 200:
            result = resp.json().get("prediction", "No result")
            prediction = f"Predicted IMDB Score: {result}"
        else:
            prediction = f"Error from inference service: {resp.text}"
    except Exception as e:
        prediction = f"Request failed: {str(e)}"

    return render_template("predict.html", prediction=prediction)

@app.route('/train_page')
def train_page():
    return render_template("train.html")

@app.route('/train', methods=['POST'])
def train():
    file = request.files.get("file")
    if not file:
        return render_template("train.html", accuracy={"status": "error", "msg": "No file uploaded"})

    try:
        resp = requests.post(TRAIN_URL, files={"file": file})
        if resp.status_code == 200:
            accuracy = resp.json()
        else:
            accuracy = {"status": "error", "msg": resp.text}
    except Exception as e:
        accuracy = {"status": "error", "msg": str(e)}

    return render_template("train.html", accuracy=accuracy)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8001, debug=True)


# ---------------------------
# FastAPI Backend for Model
# ---------------------------
fastapi_app = FastAPI()

DATA_ROOT = os.environ.get("DATA_ROOT", "/data")
MODEL_PATH = os.path.join(DATA_ROOT, "models", "model.h5")
PREPROC_PATH = os.path.join(DATA_ROOT, "models", "preproc.pkl")

model = None
preproc = None

def load_model():
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        model = tf.keras.models.load_model(MODEL_PATH)
    return model

def load_preproc():
    global preproc
    if preproc is None:
        if not os.path.exists(PREPROC_PATH):
            raise FileNotFoundError(f"Preprocessor not found at {PREPROC_PATH}")
        with open(PREPROC_PATH, "rb") as f:
            preproc = pickle.load(f)
    return preproc

def preprocess_input(df, preproc):
    director_map = preproc["director_map"]
    writer_map = preproc["writer_map"]
    genre_map = preproc["genre_map"]
    lang_map = preproc["lang_map"]
    num_cols = preproc["num_cols"]

    X_num = df[num_cols].astype(float).values
    mean = np.array(preproc["scaler_mean"])
    scale = np.array(preproc["scaler_scale"])
    X_num = (X_num - mean) / scale

    def map_value(val, mapping):
        return mapping.get(val, 0)

    X_cat = []
    if "Director" in df.columns:
        X_cat.append(df["Director"].map(lambda x: map_value(x, director_map)).values.reshape(-1, 1))
    if "Writer" in df.columns:
        X_cat.append(df["Writer"].map(lambda x: map_value(x, writer_map)).values.reshape(-1, 1))
    if "Genre" in df.columns:
        X_cat.append(df["Genre"].map(lambda x: map_value(x, genre_map)).values.reshape(-1, 1))
    if "Language" in df.columns:
        X_cat.append(df["Language"].map(lambda x: map_value(x, lang_map)).values.reshape(-1, 1))

    if len(X_cat) > 0:
        X_cat = np.hstack(X_cat)
        X_final = np.hstack([X_num, X_cat])
    else:
        X_final = X_num

    return X_final

class PredictRequest(BaseModel):
    data: dict

@fastapi_app.post("/predict")
def predict_api(request: PredictRequest):
    try:
        mdl = load_model()
        preproc = load_preproc()

        df = pd.DataFrame([request.data])
        X_final = preprocess_input(df, preproc)

        preds = mdl.predict(X_final)
        prediction = preds.tolist()

        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@fastapi_app.get("/healthz")
def healthz():
    return {"status": "ok"}
