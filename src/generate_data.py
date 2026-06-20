"""
generate_data.py
----------------
Generates a synthetic credit risk dataset that mimics real-world
loan/credit applicant data (similar in spirit to the UCI German Credit
dataset). Use this if you don't have your own dataset yet.

If you have a real dataset (e.g. UCI German Credit, Kaggle "Give Me Some
Credit", LendingClub, etc.), just drop the CSV into the `data/` folder
and update DATA_PATH in src/data_preprocessing.py instead of running this.
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

N = 5000  # number of synthetic loan applicants

def generate_dataset(n=N):
    age = np.random.randint(18, 70, n)
    job_levels = np.random.choice([0, 1, 2, 3], n, p=[0.1, 0.3, 0.4, 0.2])  # unskilled->highly skilled
    housing = np.random.choice(['own', 'rent', 'free'], n, p=[0.6, 0.3, 0.1])
    saving_accounts = np.random.choice(['little', 'moderate', 'rich', 'none'], n, p=[0.4, 0.3, 0.1, 0.2])
    checking_account = np.random.choice(['little', 'moderate', 'rich', 'none'], n, p=[0.35, 0.35, 0.1, 0.2])
    credit_amount = np.round(np.random.lognormal(mean=8.2, sigma=0.8, size=n), 0)
    duration = np.random.choice([6, 12, 18, 24, 36, 48, 60], n, p=[0.1, 0.2, 0.15, 0.2, 0.2, 0.1, 0.05])
    purpose = np.random.choice(
        ['car', 'furniture', 'education', 'business', 'appliances', 'repairs'],
        n, p=[0.3, 0.2, 0.15, 0.15, 0.15, 0.05]
    )
    employment_years = np.random.randint(0, 40, n)
    existing_credits = np.random.randint(1, 5, n)
    dependents = np.random.randint(0, 4, n)
    monthly_income = np.round(np.random.normal(45000, 18000, n).clip(8000, 200000), 0)

    # ---- Build a latent "risk score" that drives default probability ----
    risk_score = (
        -0.015 * age
        - 0.35 * job_levels
        + 0.000012 * credit_amount
        + 0.04 * duration
        - 0.00003 * monthly_income
        - 0.25 * employment_years.clip(0, 10) / 10
        + 0.3 * existing_credits
        + 0.15 * dependents
    )

    # categorical adjustments
    saving_map = {'little': 0.5, 'none': 0.8, 'moderate': 0.0, 'rich': -0.6}
    checking_map = {'little': 0.4, 'none': 0.7, 'moderate': 0.0, 'rich': -0.5}
    housing_map = {'own': -0.3, 'rent': 0.2, 'free': 0.1}
    purpose_map = {'car': 0.0, 'furniture': -0.1, 'education': -0.2,
                   'business': 0.3, 'appliances': -0.05, 'repairs': 0.1}

    risk_score += pd.Series(saving_accounts).map(saving_map).values
    risk_score += pd.Series(checking_account).map(checking_map).values
    risk_score += pd.Series(housing).map(housing_map).values
    risk_score += pd.Series(purpose).map(purpose_map).values

    # add noise
    risk_score += np.random.normal(0, 0.6, n)

    prob_default = 1 / (1 + np.exp(-risk_score))
    default = (prob_default > np.percentile(prob_default, 72)).astype(int)  # ~28% default rate

    df = pd.DataFrame({
        'age': age,
        'job_level': job_levels,
        'housing': housing,
        'saving_accounts': saving_accounts,
        'checking_account': checking_account,
        'credit_amount': credit_amount,
        'duration_months': duration,
        'purpose': purpose,
        'employment_years': employment_years,
        'existing_credits': existing_credits,
        'dependents': dependents,
        'monthly_income': monthly_income,
        'default': default
    })

    return df


if __name__ == "__main__":
    df = generate_dataset()
    out_path = Path(__file__).resolve().parent.parent / "data" / "credit_risk_dataset.csv"
    out_path.parent.mkdir(exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Generated dataset with {len(df)} rows -> {out_path}")
    print(df['default'].value_counts(normalize=True))
