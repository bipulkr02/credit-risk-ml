"""
eda.py
------
Exploratory Data Analysis: summary statistics, correlations, and
visualizations of risk patterns / default trends. Saves all plots
to the reports/ folder.
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from data_preprocessing import load_data, engineer_features

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")


def run_eda():
    df = load_data()
    df = engineer_features(df)

    print("=== Shape ===")
    print(df.shape)
    print("\n=== Missing values ===")
    print(df.isnull().sum())
    print("\n=== Default rate ===")
    print(df["default"].value_counts(normalize=True))
    print("\n=== Numeric summary ===")
    print(df.describe())

    # 1. Default distribution
    plt.figure(figsize=(5, 4))
    sns.countplot(data=df, x="default")
    plt.title("Loan Default Distribution (0=Good, 1=Default)")
    plt.savefig(REPORTS_DIR / "default_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 2. Correlation heatmap (numeric features only)
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Feature Correlation Heatmap")
    plt.savefig(REPORTS_DIR / "correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 3. Default rate by age group
    plt.figure(figsize=(7, 4))
    sns.barplot(data=df, x="age_group", y="default", errorbar=None)
    plt.title("Default Rate by Age Group")
    plt.ylabel("Default Rate")
    plt.savefig(REPORTS_DIR / "default_by_age_group.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 4. Default rate by checking account status
    plt.figure(figsize=(7, 4))
    sns.barplot(data=df, x="checking_account", y="default", errorbar=None)
    plt.title("Default Rate by Checking Account Status")
    plt.ylabel("Default Rate")
    plt.savefig(REPORTS_DIR / "default_by_checking_account.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 5. Credit amount distribution by default
    plt.figure(figsize=(7, 4))
    sns.kdeplot(data=df, x="credit_amount", hue="default", fill=True, common_norm=False)
    plt.title("Credit Amount Distribution by Default Status")
    plt.savefig(REPORTS_DIR / "credit_amount_by_default.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 6. Duration vs default
    plt.figure(figsize=(7, 4))
    sns.boxplot(data=df, x="default", y="duration_months")
    plt.title("Loan Duration vs Default")
    plt.savefig(REPORTS_DIR / "duration_vs_default.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nAll EDA plots saved to: {REPORTS_DIR}")


if __name__ == "__main__":
    run_eda()
