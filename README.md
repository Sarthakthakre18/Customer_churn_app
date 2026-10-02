# 📞 Customer Churn Prediction using Machine Learning

An end-to-end Machine Learning project that predicts whether a telecom customer is likely to churn based on their demographic and service information.

The project covers the complete ML workflow — from data preprocessing and model training to deployment as a Flask web application.

---

## 🚀 Live Demo

**Try the deployed application:**

[Customer Churn Prediction — Live Demo](https://customer-churn-app-z1qv.onrender.com/)

The application accepts customer details and predicts whether the customer is likely to stay or churn, along with the estimated churn probability.

---

## 📂 Project Structure

```text
Customer_churn_app/

├── app.py                  # Flask application
├── requirements.txt        # Python dependencies
├── README.md

├── data/
│   ├── model.pkl
│   ├── scaler.pkl
│   └── columns.pkl

├── notebook/
│   └── customer_churn.ipynb

├── templates/
│   └── index.html

└── dataset/
    └── WA_Fn-UseC_-Telco-Customer-Churn.csv
```

---

# 📊 Dataset

**Dataset:** Telco Customer Churn Dataset

The dataset contains customer information such as:

* Gender
* Senior Citizen
* Partner
* Dependents
* Tenure
* Phone Service
* Internet Service
* Online Security
* Online Backup
* Device Protection
* Tech Support
* Streaming TV
* Streaming Movies
* Contract Type
* Paperless Billing
* Payment Method
* Monthly Charges
* Total Charges

### Target Variable

**Churn**

* Yes
* No

---

# ⚙️ Machine Learning Pipeline

## 1. Data Cleaning

* Removed Customer ID
* Converted `TotalCharges` to numeric
* Removed missing values

## 2. Feature Engineering

### Label Encoding

Applied to binary features such as:

* Gender
* Senior Citizen
* Partner
* Dependents
* Phone Service
* Paperless Billing

The target variable `Churn` was also encoded for model training.

### One-Hot Encoding

Categorical features were converted using:

```python
pd.get_dummies(drop_first=True)
```

This was applied to features such as:

* Internet Service
* Contract
* Payment Method
* Streaming Services
* Online Security
* Other categorical variables

---

## 3. Feature Scaling

`StandardScaler` was used to standardize numerical features.

During training:

```python
fit_transform()
```

During deployment:

```python
transform()
```

The trained scaler is saved and reused during prediction to maintain consistency between training and deployment.

---

## 4. Train-Test Split

```text
80% Training
20% Testing
```

Random state:

```text
42
```

---

## 5. Model Training

The following models were experimented with:

* Logistic Regression
* Decision Tree
* Random Forest

Logistic Regression was selected as the final model for deployment based on the evaluation performed during model development.

---

# 📈 Model Evaluation

The model was evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

For churn prediction, recall is an important metric because missing customers who are actually likely to churn can be costly from a business perspective.

---

# 💾 Model Persistence

The trained model artifacts were saved using Pickle.

```text
data/
├── model.pkl
├── scaler.pkl
└── columns.pkl
```

These artifacts are loaded by the Flask application during prediction.

This ensures that the same trained model, scaler, and feature structure are used during deployment.

---

# 🌐 Application Workflow

The application is built using Flask.

```text
User
  ↓
HTML Form
  ↓
Flask
  ↓
request.form
  ↓
Preprocessing
  ↓
Encoding
  ↓
Column Alignment
  ↓
Feature Scaling
  ↓
Logistic Regression
  ↓
Prediction
  ↓
Churn Probability
  ↓
Display Result
```

---

# 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit
