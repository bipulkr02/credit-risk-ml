"""
train_models.py
----------------
Trains and compares Logistic Regression, Decision Tree, Random Forest,
and XGBoost classifiers for credit risk prediction. Evaluates each
model using Accuracy, Precision, Recall, and F1-Score, and saves a
comparison report + the best model.
"""

import json
import joblib
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

from data_preprocessing import get_train_test_split

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[WARNING] xgboost is not installed in this environment. "
          "Run `pip install xgboost` to include it. Skipping for now.")

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)


def get_models():
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            use_label_encoder=False, eval_metric="logloss", random_state=42
        )
    return models


def evaluate_model(name, model, X_test, y_test, y_pred):
    return {
        "model": name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_pred), 4),
    }


def train_and_compare():
    X_train, X_test, y_train, y_test, scaler, encoders, feature_cols = get_train_test_split()
    models = get_models()

    results = []
    best_model, best_name, best_f1 = None, None, -1

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = evaluate_model(name, model, X_test, y_test, y_pred)
        results.append(metrics)

        print(f"\n--- {name} ---")
        print(classification_report(y_test, y_pred, target_names=["No Default", "Default"]))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

        if metrics["f1_score"] > best_f1:
            best_f1 = metrics["f1_score"]
            best_model = model
            best_name = name

    results_df = pd.DataFrame(results).sort_values("f1_score", ascending=False)
    print("\n=== Model Comparison ===")
    print(results_df.to_string(index=False))

    results_df.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)

    # Feature importance for the best tree-based model (if applicable)
    if hasattr(best_model, "feature_importances_"):
        importances = pd.Series(best_model.feature_importances_, index=feature_cols)
        importances.sort_values(ascending=False).to_csv(REPORTS_DIR / "feature_importance.csv")

    # Save best model + scaler + encoders
    joblib.dump(best_model, MODELS_DIR / "best_model.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    joblib.dump(encoders, MODELS_DIR / "encoders.pkl")

    with open(REPORTS_DIR / "best_model_info.json", "w") as f:
        json.dump({"best_model": best_name, "f1_score": best_f1}, f, indent=2)

    print(f"\nBest model: {best_name} (F1-Score: {best_f1}) saved to {MODELS_DIR/'best_model.pkl'}")
    return results_df


if __name__ == "__main__":
    train_and_compare()
