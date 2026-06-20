"""
data_preprocessing.py
----------------------
Loads the raw dataset and performs cleaning, feature engineering,
encoding, and normalization. Returns train/test splits ready for
model training.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "credit_risk_dataset.csv"
TARGET_COL = "default"


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create a few derived features that help separate risk classes."""
    df = df.copy()

    # Debt-to-income style ratio
    df["credit_to_income_ratio"] = df["credit_amount"] / (df["monthly_income"] + 1)

    # Monthly installment estimate and burden ratio
    df["est_monthly_installment"] = df["credit_amount"] / df["duration_months"]
    df["installment_to_income_ratio"] = df["est_monthly_installment"] / (df["monthly_income"] + 1)

    # Age bucket (captures non-linear age risk effect)
    df["age_group"] = pd.cut(
        df["age"], bins=[17, 25, 35, 45, 60, 100],
        labels=["18-25", "26-35", "36-45", "46-60", "60+"]
    )

    # Stability proxy
    df["employment_stability"] = df["employment_years"] / (df["age"] - 17)

    return df


def encode_and_scale(df: pd.DataFrame):
    """Label-encode categoricals, then standard-scale numeric features."""
    df = df.copy()

    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    feature_cols = [c for c in df.columns if c != TARGET_COL]
    X = df[feature_cols]
    y = df[TARGET_COL]

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols, index=X.index)

    return X_scaled, y, scaler, encoders, feature_cols


def get_train_test_split(test_size: float = 0.2, random_state: int = 42):
    df = load_data()
    df = engineer_features(df)
    X, y, scaler, encoders, feature_cols = encode_and_scale(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test, scaler, encoders, feature_cols


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler, encoders, feature_cols = get_train_test_split()
    print("Train shape:", X_train.shape, " Test shape:", X_test.shape)
    print("Features:", feature_cols)
    print("Train default rate:", y_train.mean().round(3), " Test default rate:", y_test.mean().round(3))
