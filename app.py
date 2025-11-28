from flask import Flask, request, render_template
import numpy as np
import pickle
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

template_path = os.path.join(BASE_DIR, "pages")
static_path = os.path.join(BASE_DIR, "images")

print("Templates:", template_path)
print("Static:", static_path)

app = Flask(__name__, template_folder=template_path, static_folder=static_path)

model = pickle.load(open(os.path.join(BASE_DIR, 'model.pkl'), 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['feature']

    try:
        features = [float(x) for x in user_input.split(',')]
        final_features = np.array([features])
        prediction = model.predict(final_features)[0]

        result = "Malignant (Cancer Detected)" if prediction == 1 else "Benign (No Cancer)"
        return render_template('index.html', prediction=result)

    except:
        return render_template('index.html', prediction="❌ Invalid input format.")

if __name__ == "__main__":
    app.run(debug=True)
