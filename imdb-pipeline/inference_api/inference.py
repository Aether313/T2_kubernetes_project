# done by edric
# # app.py

# from flask import Flask, request, jsonify, render_template
# import joblib
# import numpy as np

# # Create the Flask application instance
# app = Flask(__name__)

# # Load the pre-trained model
# # The model should be saved in the same directory as this file
# model = joblib.load("model.pkl")

# # Define the home route
# @app.route('/')
# def home():
#     # This renders an HTML page if you have a templates folder
#     # For a simple API, you might not need this.
#     return "<h1>Flask API for Model Inference</h1><p>Send a POST request to /predict to get a prediction.</p>"

# # Define the prediction endpoint
# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Get the JSON data from the request
#         data = request.get_json(force=True)

#         # Assuming the incoming data is a list of features, e.g., {'features': [1.2, 3.4, 5.6]}
#         features = np.array(data['features']).reshape(1, -1)

#         # Make a prediction using the loaded model
#         prediction = model.predict(features)

#         # Convert the prediction to a list or a single value for JSON serialization
#         prediction_list = prediction.tolist()

#         # Return the prediction as a JSON response
#         return jsonify({
#             'prediction': prediction_list
#         })

#     except Exception as e:
#         return jsonify({
#             'error': str(e)
#         })

# if __name__ == '__main__':
#     # Run the application
#     app.run(debug=True)


# edited by stephanie and feiyang:
# inference.py
from tensorflow.keras.models import load_model
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the trained Keras model
model = load_model("/data/models/best_model.h5")

# Load preprocessing
preproc = joblib.load("/data/models/preproc.pkl")

def preprocess_input(df):
    X_num = df[['Runtime']].fillna(0).values.astype(np.float32)
    mean = np.array(preproc['scaler_mean'])[:1]
    scale = np.array(preproc['scaler_scale'])[:1]
    X_num = (X_num - mean) / scale

    def map_single(series, mapping):
        unk = mapping.get('[UNK]', 1)
        return series.map(lambda x: mapping.get(x, unk)).values.astype(np.int32)

    def map_multi(series, mapping, max_len=6):
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        unk = mapping.get('[UNK]', 1)
        pad = mapping.get('[PAD]', 0)
        seqs = [[mapping.get(t.strip(), unk) for t in str(val).split(',') if t.strip()][:max_len]
                for val in series]
        return pad_sequences(seqs, padding='post', truncating='post', value=pad, maxlen=max_len)

    X_director = map_single(df['Director'], preproc['director_map'])
    X_writer = map_single(df['Writer'], preproc['writer_map'])
    X_prod = map_single(df['Production House'], preproc['prod_map'])
    X_genre_seq = map_multi(df['Genre'], preproc['genre_map'], max_len=6)
    X_lang_seq = map_multi(df['Language'], preproc['lang_map'], max_len=4)
    X_tags_seq = np.zeros((len(df), 12), dtype=np.int32)
    X_country_seq = np.zeros((len(df), 10), dtype=np.int32)

    return {
        'num_input': X_num,
        'director_input': X_director,
        'writer_input': X_writer,
        'prod_input': X_prod,
        'genre_seq': X_genre_seq,
        'lang_seq': X_lang_seq,
        'tags_seq': X_tags_seq,
        'country_seq': X_country_seq
    }

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        # If form fields are sent flat (no "row"), wrap them in a DataFrame directly
        if not isinstance(data, dict):
            return jsonify({'status': 'error', 'msg': 'Invalid input format'})

        df = pd.DataFrame([{
            "Genre": data.get("Genre", ""),
            "Runtime": data.get("Runtime", 0),
            "Language": data.get("Language", ""),
            "Director": data.get("Director", ""),
            "Writer": data.get("Writer", ""),
            "Production House": data.get("Production House", "")
        }])

        X = preprocess_input(df)
        pred = model.predict(X)
        return jsonify({'prediction': pred.flatten().tolist()})
    except Exception as e:
        return jsonify({'status': 'error', 'msg': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
