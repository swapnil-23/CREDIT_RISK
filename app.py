import string
from flask import Flask, jsonify, url_for, render_template, request
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import OrdinalEncoder

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))
ordinal_encoder = pickle.load(open("ordinal_encoder.pkl", "rb"))

@app.route('/')
def Main():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the form
    checking_status = request.form['checking_status']
    credit_history = request.form['credit_history']
    purpose = request.form['purpose']
    savings_status = request.form['savings_status']
    employment = request.form['employment']
    personal_status = request.form['personal_status']
    other_parties = request.form['other_parties']
    property_magnitude = request.form['property_magnitude']
    other_payment_plans = request.form['other_payment_plans']
    housing = request.form['housing']
    job = request.form['job']
    own_telephone = request.form['own_telephone']
    foreign_worker = request.form['foreign_worker']

    # Encode the input data
    input_data = pd.DataFrame({
        'checking_status': [checking_status],
        'credit_history': [credit_history],
        'purpose': [purpose],
        'savings_status': [savings_status],
        'employment': [employment],
        'personal_status': [personal_status],
        'other_parties': [other_parties],
        'property_magnitude': [property_magnitude],
        'other_payment_plans': [other_payment_plans],
        'housing': [housing],
        'job': [job],
        'own_telephone': [own_telephone],
        'foreign_worker': [foreign_worker],
        'age': [0],
        'credit_amount': [0],
        'duration': [0],
        'installment_commitment': [0],
        'residence_since': [0],
        'existing_credits': [0],
        'class': [0]
    })

    # Ensure feature names are in the same order as during fitting
    input_data_encoded = ordinal_encoder.transform(input_data)

    # Make the prediction
    prediction = model.predict(input_data_encoded)[0]

    # Render the result
    if prediction == 0:
        prediction_text = 'Credit risk is LOW'
    else:
        prediction_text = 'Credit risk is HIGH'

    return render_template('credit_risk.html', prediction_res=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)
