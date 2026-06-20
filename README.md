# Credit Risk Analysis using Machine Learning

**Duration:** Jan 2026 – Feb 2026

An end-to-end machine learning pipeline that analyzes and predicts customer credit risk (loan default) from financial/applicant data. The project covers data preprocessing, feature engineering, encoding, normalization, exploratory data analysis, and comparison of multiple classification models.

## Project Overview

- Built an end-to-end ML pipeline to predict customer credit risk using financial datasets.
- Performed data preprocessing, feature engineering, encoding, and normalization to improve prediction accuracy.
- Conducted exploratory data analysis (EDA) to identify customer risk patterns, correlations, and loan default trends using statistical analysis and visualization techniques.
- Trained and compared multiple machine learning models: **Logistic Regression, Decision Tree, Random Forest, and XGBoost**.
- Evaluated models using **Accuracy, Precision, Recall, and F1-Score**.

## Project Structure

```
credit-risk-ml/
├── data/
│   └── credit_risk_dataset.csv      # generated/loaded dataset
├── notebooks/
│   └── Credit_Risk_Analysis.ipynb   # full walkthrough notebook
├── src/
│   ├── generate_data.py             # synthetic data generator
│   ├── data_preprocessing.py        # cleaning, feature engineering, encoding, scaling
│   ├── eda.py                       # exploratory data analysis + plots
│   └── train_models.py              # model training, evaluation, model selection
├── models/                          # saved best model, scaler, encoders (.pkl)
├── reports/                         # EDA plots, model comparison CSV, metrics
├── main.py                          # runs the full pipeline end-to-end
├── requirements.txt
└── README.md
```

## Dataset

This repo ships with a **synthetic credit-risk dataset generator** (`src/generate_data.py`) so the project runs out of the box with no external downloads. It produces realistic applicant features (age, income, credit amount, duration, savings/checking status, employment history, etc.) with a default label driven by a configurable risk function (~28% default rate).

If you have a real dataset (e.g. the UCI **German Credit Data**, Kaggle's **"Give Me Some Credit"**, or a LendingClub export), just drop the CSV into `data/` and update `DATA_PATH` in `src/data_preprocessing.py` — the rest of the pipeline works unchanged.

## Features Engineered

- `credit_to_income_ratio` — credit amount relative to monthly income
- `est_monthly_installment` / `installment_to_income_ratio` — affordability proxies
- `age_group` — binned age buckets to capture non-linear risk
- `employment_stability` — employment years relative to working-age years

## Models Compared

| Model | Notes |
|---|---|
| Logistic Regression | Baseline linear model, interpretable coefficients |
| Decision Tree | Captures non-linear splits, prone to overfitting |
| Random Forest | Ensemble of trees, reduces variance |
| XGBoost | Gradient boosting, typically strongest performer |

## Sample Results

*(from a run on the synthetic dataset — your numbers will vary slightly by random seed / dataset)*

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Logistic Regression | 0.812 | 0.711 | 0.554 | 0.623 |
| Random Forest | 0.810 | 0.750 | 0.482 | 0.587 |
| Decision Tree | 0.763 | 0.588 | 0.514 | 0.549 |
| XGBoost | *(install xgboost to populate)* | | | |

Full metrics are saved to `reports/model_comparison.csv` after each run.

## How to Run

1. **Clone the repo and install dependencies**
   ```bash
   git clone https://github.com/<your-username>/credit-risk-ml.git
   cd credit-risk-ml
   pip install -r requirements.txt
   ```

2. **Run the full pipeline**
   ```bash
   python main.py
   ```
   This will:
   - Generate the dataset (if not already present in `data/`)
   - Run EDA and save plots to `reports/`
   - Train and compare all 4 models
   - Save the best model to `models/best_model.pkl`

3. **Or explore step by step in the notebook**
   ```bash
   jupyter notebook notebooks/Credit_Risk_Analysis.ipynb
   ```

## Tech Stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `XGBoost` · `Matplotlib` · `Seaborn` · `Jupyter`

## Future Improvements

- Hyperparameter tuning (GridSearchCV / Optuna)
- SHAP-based model explainability
- Handling class imbalance with SMOTE
- Deploy best model behind a simple Flask/FastAPI scoring endpoint
