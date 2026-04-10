from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Load model
model_path = os.path.join('model', 'c_card_approval_pred.pickle')
with open(model_path, 'rb') as handle:
    model = pickle.load(handle)

# 🔑 Encoding dictionaries
gender_map = {"Female": 0, "Male": 1}
yes_no_map = {"No": 0, "Yes": 1}

income_type_map = {
    "Working": 0,
    "Commercial associate": 1,
    "Pensioner": 2,
    "State servant": 3,
    "Student": 4
}

education_map = {
    "Secondary": 0,
    "Higher": 1,
    "Incomplete Higher": 2,
    "Lower Secondary": 3,
    "Academic Degree": 4
}

family_status_map = {
    "Single": 0,
    "Married": 1,
    "Civil marriage": 2,
    "Separated": 3,
    "Widow": 4
}

housing_map = {
    "House / apartment": 0,
    "With parents": 1,
    "Municipal apartment": 2,
    "Rented apartment": 3,
    "Office apartment": 4
}

# Routes
@app.route('/')
def home():
    return render_template('ccaindex.html')

@app.route('/Prediction')
def prediction():
    return render_template('ccaindex1.html')

@app.route('/predict', methods=["POST"])
def predict():
    try:
        form = request.form

        features = [
            gender_map[form['gender']],
            yes_no_map[form['car']],
            yes_no_map[form['realty']],
            float(form['income']),
            income_type_map[form['income_type']],
            education_map[form['education']],
            family_status_map[form['family_status']],
            housing_map[form['housing']],
            float(form['days_birth']),
            float(form['days_employed']),
            float(form['family_members']),
            yes_no_map[form['paid_off']],
            float(form['past_dues']),
            float(form['loans'])
        ]

        feature_names = [
            'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
            'AMT_INCOME_TOTAL', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
            'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'DAYS_BIRTH',
            'DAYS_EMPLOYED', 'CNT_FAM_MEMBERS', 'paid_off',
            '#_of_pastdues', 'no_loan'
        ]

        x = pd.DataFrame([features], columns=feature_names)

        pred = model.predict(x)

        prediction = "Eligible ✅" if pred[0] == 1 else "Not Eligible ❌"

        return render_template("Results.html", prediction=prediction)

    except Exception as e:
        return f"Error: {e}"


# 🔥 THIS GOES AT THE VERY END (NOT INSIDE ANY FUNCTION)
if __name__ == "__main__":
    app.run(debug=True)