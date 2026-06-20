"""
main.py
-------
Runs the full credit risk ML pipeline end-to-end:
1. Generate (or load) dataset
2. Run EDA and save visualizations
3. Preprocess / engineer features
4. Train & compare models
5. Save best model + evaluation report
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent / "src"))

from generate_data import generate_dataset
from eda import run_eda
from train_models import train_and_compare

DATA_PATH = Path(__file__).resolve().parent / "data" / "credit_risk_dataset.csv"


def main():
    if not DATA_PATH.exists():
        print("No dataset found — generating synthetic credit risk dataset...")
        df = generate_dataset()
        DATA_PATH.parent.mkdir(exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
        print(f"Dataset saved to {DATA_PATH}\n")

    print("Running EDA...\n")
    run_eda()

    print("\nTraining and comparing models...\n")
    train_and_compare()

    print("\nPipeline complete. Check the 'reports/' folder for plots, metrics, "
          "and 'models/' for the saved best model.")


if __name__ == "__main__":
    main()
