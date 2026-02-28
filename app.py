from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load saved model
model = joblib.load("random_forest_model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form values
        features = [
            float(request.form['age_first_funding']),
            float(request.form['age_last_funding']),
            float(request.form['age_first_milestone']),
            float(request.form['age_last_milestone']),
            float(request.form['relationships']),
            float(request.form['funding_rounds']),
            float(request.form['total_funding']),
            float(request.form['milestones']),
            float(request.form['avg_participants'])
        ]

        final_features = np.array([features])
        prediction = model.predict(final_features)

        if prediction[0] == 1:
            result = "Startup Will Succeed 🚀"
        else:
            result = "Startup May Not Succeed ❌"

        return render_template('result.html', prediction_text=result)

    except:
        return render_template('result.html', prediction_text="Error in Input Values")

if __name__ == "__main__":
    app.run(debug=True)
