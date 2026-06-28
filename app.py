from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# ---------------------------------------------------------
# Load the artifacts saved at the end of the notebook
# ---------------------------------------------------------
with open("data/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("data/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("data/columns.pkl", "rb") as f:
    training_columns = pickle.load(f)  # X.columns.tolist() from training

# ---------------------------------------------------------
# Same column groups used in the notebook
# ---------------------------------------------------------
binary_col = [
    'gender',
    'SeniorCitizen',
    'Partner',
    'Dependents',
    'PhoneService',
    'PaperlessBilling'
]

nominal_col = [
    'MultipleLines',
    'InternetService',
    'OnlineSecurity',
    'OnlineBackup',
    'DeviceProtection',
    'TechSupport',
    'StreamingTV',
    'StreamingMovies',
    'Contract',
    'PaymentMethod',
]

num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

# LabelEncoder encodes alphabetically, so for every Yes/No and
# Female/Male column in this dataset it lands on this exact mapping.
binary_map = {
    "gender": {"Female": 0, "Male": 1},
    "SeniorCitizen": {"0": 0, "1": 1},
    "Partner": {"No": 0, "Yes": 1},
    "Dependents": {"No": 0, "Yes": 1},
    "PhoneService": {"No": 0, "Yes": 1},
    "PaperlessBilling": {"No": 0, "Yes": 1},
}


def preprocess(form_data):
    """
    Turn raw form input into the exact feature row the model expects:
    1) label-encode binary columns
    2) one-hot encode nominal columns (drop_first=True, same as training)
    3) reindex to training_columns (adds any missing dummy cols as 0,
       drops anything extra, fixes column order)
    4) scale the numeric columns with the saved scaler
    """
    row = {
        "gender": form_data["gender"],
        "SeniorCitizen": str(form_data["SeniorCitizen"]),
        "Partner": form_data["Partner"],
        "Dependents": form_data["Dependents"],
        "tenure": float(form_data["tenure"]),
        "PhoneService": form_data["PhoneService"],
        "MultipleLines": form_data["MultipleLines"],
        "InternetService": form_data["InternetService"],
        "OnlineSecurity": form_data["OnlineSecurity"],
        "OnlineBackup": form_data["OnlineBackup"],
        "DeviceProtection": form_data["DeviceProtection"],
        "TechSupport": form_data["TechSupport"],
        "StreamingTV": form_data["StreamingTV"],
        "StreamingMovies": form_data["StreamingMovies"],
        "Contract": form_data["Contract"],
        "PaperlessBilling": form_data["PaperlessBilling"],
        "PaymentMethod": form_data["PaymentMethod"],
        "MonthlyCharges": float(form_data["MonthlyCharges"]),
        "TotalCharges": float(form_data["TotalCharges"]),
    }

    df = pd.DataFrame([row])

    # 1) Binary columns -> same 0/1 mapping LabelEncoder produced in training
    for col in binary_col:
        df[col] = df[col].map(binary_map[col]).astype(int)

    # 2) Nominal columns -> one-hot, dropping first category just like training
    df = pd.get_dummies(df, columns=nominal_col, drop_first=True)

    # 3) Align columns with what the model was trained on
    df = df.reindex(columns=training_columns, fill_value=0)

    # 4) Scale numeric columns with the fitted scaler
    df[num_cols] = scaler.transform(df[num_cols])

    return df


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    form_data = request.form

    X_new = preprocess(form_data)

    prediction = model.predict(X_new)[0]
    probability = model.predict_proba(X_new)[0][1]

    result = "Customer is likely to CHURN" if prediction == 1 else "Customer is likely to STAY"

    return render_template(
        "index.html",
        prediction_text=result,
        probability_text=f"Churn probability: {probability:.2%}"
    )


if __name__ == "__main__":
    app.run(debug=True)