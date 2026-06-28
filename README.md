# 📞 Customer Churn Prediction using Machine Learning

An end-to-end Machine Learning project that predicts whether a telecom customer is likely to churn based on their demographic and service information.

The project covers the complete ML workflow—from data preprocessing and model training to deployment using Flask.

---

## 🚀 Project Demo

Predict whether a customer will:

- ✅ Stay with the company
- ❌ Churn from the company

The web application accepts customer details through a user-friendly interface and instantly predicts the churn probability.

---

## 📂 Project Structure

```
Customer_Churn_Prediction/
│
├── app.py                  # Flask application
├── requirements.txt
├── README.md
│
├── data/
│   ├── model.pkl
│   ├── scaler.pkl
│   └── columns.pkl
│
├── notebook/
│   └── customer_churn.ipynb
│
├── templates/
│   └── index.html
│
└── dataset/
    └── WA_Fn-UseC_-Telco-Customer-Churn.csv
```

---

# 📊 Dataset

Dataset: **Telco Customer Churn Dataset**

The dataset contains customer information such as:

- Gender
- Senior Citizen
- Partner
- Dependents
- Tenure
- Phone Service
- Internet Service
- Online Security
- Online Backup
- Device Protection
- Tech Support
- Streaming TV
- Streaming Movies
- Contract Type
- Paperless Billing
- Payment Method
- Monthly Charges
- Total Charges

Target Variable:

- **Churn**
    - Yes
    - No

---

# ⚙️ Machine Learning Pipeline

## 1. Data Cleaning

- Removed Customer ID
- Converted TotalCharges to numeric
- Removed missing values

---

## 2. Feature Engineering

### Label Encoding

Applied on binary columns:

- Gender
- Senior Citizen
- Partner
- Dependents
- Phone Service
- Paperless Billing
- Churn

---

### One Hot Encoding

Applied using:

```python
pd.get_dummies(drop_first=True)
```

for categorical features like:

- Internet Service
- Contract
- Payment Method
- Streaming Services
- Online Security
- etc.

---

## 3. Feature Scaling

StandardScaler was applied on:

- Tenure
- Monthly Charges
- Total Charges

Training:

```python
fit_transform()
```

Deployment:

```python
transform()
```

---

## 4. Train-Test Split

```
80% Training
20% Testing
```

Random State:

```
42
```

---

## 5. Model Training

The following models were experimented with:

- Logistic Regression
- Decision Tree
- Random Forest

After comparing performance metrics, **Logistic Regression** was selected as the final model.

---

# 📈 Model Performance

Evaluation Metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

Business Goal:

Customer churn prediction is a **recall-focused problem**, where identifying potential churn customers is more important than minimizing false positives.

---

# 💾 Model Persistence

The trained artifacts were saved using Pickle.

Saved files:

```
model.pkl
scaler.pkl
columns.pkl
```

These files are loaded during deployment to ensure the preprocessing pipeline remains identical to training.

---

# 🌐 Deployment

The application is built using **Flask**.

Workflow:

```
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
Label Encoding
    ↓
One Hot Encoding
    ↓
Column Alignment
    ↓
Feature Scaling
    ↓
Logistic Regression
    ↓
Prediction
    ↓
Display Result
```

---

# 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Flask
- HTML
- Pickle

---

# ▶️ Installation

Clone the repository

```bash
git clone https://github.com/Sarthakthakre18/Customer_churn_app.git
```

Move inside the project

```bash
cd Customer-Churn-Prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the Flask application

```bash
python app.py
```

Open your browser

```
http://127.0.0.1:5000/
```

# 🎯 Key Learnings

Through this project I learned:

- Data preprocessing
- Label Encoding
- One Hot Encoding
- Feature Scaling
- Logistic Regression
- Model Evaluation
- Model Serialization using Pickle
- Flask Deployment
- Building an end-to-end Machine Learning application
- Handling preprocessing during deployment
- Feature alignment using `reindex()`

---

# 🚀 Future Improvements

- Deploy on Render
- Dockerize the application
- Improve UI using Bootstrap or React
- Add probability visualization
- Add SHAP explanations for predictions
- Store prediction history in a database

---

# 👨‍💻 Author

**Sarthak Thakre**

3rd Year AIML Engineering Student

Aspiring Machine Learning Engineer

Focused on building end-to-end ML projects, mastering Python, Machine Learning
