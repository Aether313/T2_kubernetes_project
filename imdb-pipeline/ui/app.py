from flask import Flask, render_template, request
import requests
import os

app = Flask(__name__)

# Use environment variable or default to service name in Kubernetes
INFERENCE_URL = os.getenv("INFERENCE_URL", "http://inference-svc:5000/predict")

# Page 1: Predict Me
@app.route('/')
def predict_page():
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    form_data = request.form.to_dict()

    # Build payload matching inference.py expectations ("row" key with all features)
    payload = {
        "row": {
            "Genre": form_data.get('Genre', ''),
            "Runtime": int(form_data.get('Runtime (minutes)', 0) or 0),
            "Language": form_data.get('Language', ''),
            "Director": form_data.get('Director', 'Unknown'),
            "Writer": form_data.get('Writer', 'Unknown'),
            "Production House": form_data.get('Production House', 'Unknown')
        }
    }

    # Debug log
    print("Payload to inference API:", payload)

    try:
        response = requests.post(INFERENCE_URL, json=payload)
        print("Response status:", response.status_code, response.text)

        if response.status_code == 200:
            prediction_list = response.json().get('prediction', [])
            if prediction_list:
                prediction_value = prediction_list[0]
                prediction = f"Predicted IMDB Score: {prediction_value}"
            else:
                prediction = "No prediction returned"
        else:
            prediction = f"Error: {response.text}"
    except Exception as e:
        prediction = f"Error: {str(e)}"

    return render_template('predict.html', prediction=prediction)

# Page 2: Upload CSV & Train (optional)
@app.route('/train_page')
def train_page():
    return render_template('train.html')

@app.route('/train', methods=['POST'])
def train():
    file = request.files['file']
    accuracy = "Fake Accuracy: 0.85"
    return render_template('train.html', accuracy=accuracy)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)






