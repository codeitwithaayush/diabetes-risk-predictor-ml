"""Predict diabetes risk for one patient using the trained model."""
import argparse
from pathlib import Path

import joblib
import pandas as pd

MODEL = Path(__file__).resolve().parents[1] / "models" / "svm_model.joblib"
FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]


def main():
    parser = argparse.ArgumentParser(description="Diabetes prediction (educational use only)")
    for feature in FEATURES:
        parser.add_argument(f"--{feature.lower()}", type=float, required=True, help=feature)
    args = parser.parse_args()

    if not MODEL.exists():
        raise SystemExit("Model not found. Run: python src/train.py")

    model = joblib.load(MODEL)
    row = pd.DataFrame([[getattr(args, f.lower()) for f in FEATURES]], columns=FEATURES)
    result = int(model.predict(row)[0])
    print("Prediction:", "Diabetic (1)" if result == 1 else "Non-diabetic (0)")


if __name__ == "__main__":
    main()