from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("Loan_Model.pkl")
scalar=joblib.load("Scalar_model.pkl")

@app.route('/')
def form():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # Extract features in correct order
    features = [
        data['dependents'], data['education'], data['self_employed'],
        data['income'], data['loan_amount'], data['tenure'],
        data['cibil'], data['movable_assets'], data['immovable_assets']
    ]
    input_data = np.array([features])
    Scaled_data=scalar.transform(input_data)
    prediction = model.predict(Scaled_data)[0]
    result = "Approved ✅" if prediction == 1 else "Rejected ❌"
    return jsonify({"result": result})

if __name__ == '__main__':
    app.run(debug=True)
