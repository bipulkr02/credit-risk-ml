# 🏦 Credit Risk Prediction System

## 📌 Overview
The **Credit Risk Prediction System** is a Machine Learning based application designed to evaluate the financial risk associated with loan applicants. The system analyzes applicant information, credit behaviour, and loan characteristics to estimate the probability of loan default and generate a credit score.

This project demonstrates an end-to-end Machine Learning pipeline, including data preprocessing, exploratory data analysis, feature engineering, training and comparing multiple models, and deployment using Streamlit.

## 🚀 Live Application
🔗 **Live Demo:** _add your Streamlit Cloud link here after deploying_

## 🎯 Problem Statement
Financial institutions must assess whether a borrower is capable of repaying a loan. Manual evaluation is time-consuming and prone to bias.

This project automates credit evaluation using Machine Learning to support data-driven lending decisions.

## ⚙️ Features
- Credit Risk Prediction
- Default Probability Estimation
- Credit Score Generation (300–900)
- Risk Rating Classification
- Interactive Streamlit Dashboard
- Multi-Model Comparison Dashboard (Logistic Regression / Decision Tree / Random Forest / XGBoost)

## 🧠 Machine Learning Workflow
1. Data Collection
2. Data Cleaning & Preprocessing
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. Feature Encoding & Scaling using StandardScaler
6. Model Training (Logistic Regression, Decision Tree, Random Forest, XGBoost)
7. Model Evaluation (Accuracy, Precision, Recall, F1-Score, ROC-AUC)
8. Best Model Selection & Serialization using Joblib
9. Deployment using Streamlit Cloud

## 🛠 Tech Stack
- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib / Seaborn
- Streamlit
- Joblib

## 📊 Input Parameters
The model evaluates applicants using:
- Age
- Job Level
- Housing Type
- Savings Account Status
- Checking Account Status
- Monthly Income
- Loan / Credit Amount
- Loan Duration (Tenure)
- Loan Purpose
- Years of Employment
- Existing Credit Lines
- Number of Dependents

## 📈 Model Output
The system provides:
- Default Probability
- Credit Score (300–900)
- Risk Rating (Poor / Average / Good / Excellent)

## 📂 Project Structure
```
credit-risk-ml
│
├── data/
│   └── credit_risk_dataset.csv
├── models/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   └── encoders.pkl
├── notebooks/
│   └── Credit_Risk_Analysis.ipynb
├── reports/
│   └── (EDA plots, model_comparison.csv)
├── src/
│   ├── generate_data.py
│   ├── data_preprocessing.py
│   ├── eda.py
│   └── train_models.py
├── app.py
├── main.py
├── requirements.txt
├── runtime.txt
└── README.md
```

## ▶️ How to Run Locally

Clone repository:
```bash
git clone https://github.com/bipulkr02/credit-risk-ml.git
```

Move into project folder:
```bash
cd credit-risk-ml
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Run the ML pipeline (generates data, EDA, trains & compares models):
```bash
python main.py
```

Run the web application:
```bash
streamlit run app.py
```

## 📷 Application Preview
_(add a screenshot of your running app here)_

## 🌐 Deployment
The application is deployed using **Streamlit Cloud** for real-time access.

## 👨‍💻 Author
**Bipul Kumar**

## ⭐ Future Improvements
- Hyperparameter tuning using Optuna / GridSearchCV
- Explainable AI (SHAP values)
- API integration using FastAPI
- Real-world banking dataset expansion
- Handling class imbalance with SMOTE
