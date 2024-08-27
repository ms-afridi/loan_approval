from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the scaler and model from pickle files
scaler = pickle.load(open("Model/scalerr.pkl", "rb"))
model = pickle.load(open("Model/log_reg.pkl", "rb"))

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predictdata', methods=['POST'])
def predict_datapoint():
    if request.method == 'POST':
        # Retrieve and process form data
        no_of_dependents = int(request.form.get("no_of_dependents"))
        education = int(request.form.get("education"))
        self_employed = int(request.form.get("self_employed"))
        income_annum = float(request.form.get("income_annum"))
        loan_amount = float(request.form.get("loan_amount"))
        loan_term = float(request.form.get("loan_term"))
        cibil_score = float(request.form.get("cibil_score"))
        residential_assets_value = float(request.form.get("residential_assets_value"))
        commercial_assets_value = float(request.form.get("commercial_assets_value"))
        luxury_assets_value = float(request.form.get("luxury_assets_value"))
        bank_asset_value = float(request.form.get("bank_asset_value"))

        # Prepare data for prediction
        new_data = pd.DataFrame([[no_of_dependents, education, self_employed, income_annum,
                                  loan_amount, loan_term, cibil_score,
                                  residential_assets_value, commercial_assets_value,
                                  luxury_assets_value, bank_asset_value]],
                                columns=['no_of_dependents', 'education', 'self_employed', 'income_annum',
                                         'loan_amount', 'loan_term', 'cibil_score',
                                         'residential_assets_value', 'commercial_assets_value',
                                         'luxury_assets_value', 'bank_asset_value'])

        # Scale and predict
        new_data_scaled = scaler.transform(new_data)
        prediction = model.predict(new_data_scaled)

        result = 'Approved' if prediction[0] == 1 else 'Rejected'
        return render_template('single_prediction.html', result=result)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
