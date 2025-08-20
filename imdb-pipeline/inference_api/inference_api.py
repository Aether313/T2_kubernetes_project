# done by edric
# # # app.py
# #
# from flask import Flask, request, jsonify, render_template
# import joblib
# import numpy as np
#
# # Create the Flask application instance
# app = Flask(__name__)
#
# # Load the pre-trained model
# # The model should be saved in the same directory as this file
# model = joblib.load("best_model.h5")
#
# # Define the home route
# @app.route('/')
# def home():
#     # This renders an HTML page if you have a templates folder
#     # For a simple API, you might not need this.
#     return "<h1>Flask API for Model Inference</h1><p>Send a POST request to /predict to get a prediction.</p>"
#
# # Define the prediction endpoint
# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Get the JSON data from the request
#         data = request.get_json(force=True)
#
#         # Assuming the incoming data is a list of features, e.g., {'features': [1.2, 3.4, 5.6]}
#         features = np.array(data['features']).reshape(1, -1)
#
#         # Make a prediction using the loaded model
#         prediction = model.predict(features)
#
#         # Convert the prediction to a list or a single value for JSON serialization
#         prediction_list = prediction.tolist()
#
#         # Return the prediction as a JSON response
#         return jsonify({
#             'prediction': prediction_list
#         })
#
#     except Exception as e:
#         return jsonify({
#             'error': str(e)
#         })
#
# if __name__ == '__main__':
#     # Run the application
#     app.run(debug=True)


#Done by Feiyang and Stephanie
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import tensorflow as tf
import os
import pickle
import numpy as np

app = FastAPI()

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
    director_map = preproc.get("director_map", {})
    writer_map = preproc.get("writer_map", {})
    prod_map = preproc.get("prod_map", {})
    genre_map = preproc.get("genre_map", {})
    lang_map = preproc.get("lang_map", {})
    tags_map = preproc.get("tags_map", {})
    country_map = preproc.get("country_map", {})
    num_cols = preproc.get("num_cols", [])

    # numeric normalization
    X_num = df[num_cols].astype(float).values
    mean = np.array(preproc.get("scaler_mean", np.zeros(len(num_cols))))
    scale = np.array(preproc.get("scaler_scale", np.ones(len(num_cols))))
    X_num = (X_num - mean) / scale

    def map_value(val, mapping):
        return mapping.get(val, mapping.get("[UNK]", 1))

    def process_list_column(val, mapping, max_len):
        if pd.isna(val) or val == "":
            tokens = []
        else:
            tokens = [t.strip() for t in str(val).split(",") if t.strip()]
        unk = mapping.get("[UNK]", 1)
        pad = mapping.get("[PAD]", 0)
        seq = [mapping.get(t, unk) for t in tokens][:max_len]
        if len(seq) < max_len:
            seq += [pad] * (max_len - len(seq))
        return np.array(seq, dtype=np.int32)

    # safe access with get
    X_dir = df.get("Director", pd.Series([""])).map(lambda x: map_value(x, director_map)).values.reshape(-1, 1)
    X_wri = df.get("Writer", pd.Series([""])).map(lambda x: map_value(x, writer_map)).values.reshape(-1, 1)
    X_prod = df.get("Production House", pd.Series([""])).map(lambda x: map_value(x, prod_map)).values.reshape(-1, 1)

    X_gen = np.vstack(df.get("Genre", pd.Series([""])).apply(lambda x: process_list_column(x, genre_map, 6)).values)
    X_lang = np.vstack(df.get("Language", pd.Series([""])).apply(lambda x: process_list_column(x, lang_map, 4)).values)
    X_tags = np.vstack(df.get("Tags", pd.Series([""])).apply(lambda x: process_list_column(x, tags_map, 12)).values)
    X_ctry = np.vstack(df.get("Country", pd.Series([""])).apply(lambda x: process_list_column(x, country_map, 10)).values)

    return [X_num, X_dir, X_wri, X_prod, X_gen, X_lang, X_tags, X_ctry]

class PredictRequest(BaseModel):
    data: dict

@app.post("/predict")
def predict(request: PredictRequest):
    try:
        mdl = load_model()
        preproc = load_preproc()

        df = pd.DataFrame([request.data])
        X_list = preprocess_input(df, preproc)  # list of 8 inputs

        preds = mdl.predict(X_list)
        prediction = preds.tolist()

        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/healthz")
def healthz():
    return {"status": "ok"}
