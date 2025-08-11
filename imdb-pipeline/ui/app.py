from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    predicted_score = None
    if request.method == 'POST':
        # Get form data
        genre = request.form.get('genre')
        runtime = request.form.get('runtime')
        language = request.form.get('language')
        director = request.form.get('director')
        writer = request.form.get('writer')
        production = request.form.get('production')

        # Here, add your real model inference logic
        # For now, we'll just mock a prediction
        predicted_score = 7.8

    return render_template('index.html', score=predicted_score)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
