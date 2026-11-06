"""Train and evaluate a linear SVM on the Pima diabetes dataset."""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "diabetesdata.csv"
MODEL = ROOT / "models" / "svm_model.joblib"


def main():
    df = pd.read_csv(DATA)
    X = df.drop(columns="Outcome")
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=2
    )

    # Scaler lives inside the pipeline so it is fitted on training data only
    model = make_pipeline(StandardScaler(), SVC(kernel="linear"))
    model.fit(X_train, y_train)

    print(f"Train accuracy: {accuracy_score(y_train, model.predict(X_train)):.4f}")
    print(f"Test accuracy : {accuracy_score(y_test, model.predict(X_test)):.4f}")
    print(classification_report(
        y_test, model.predict(X_test), target_names=["Non-diabetic", "Diabetic"]
    ))

    MODEL.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL)
    print(f"Model saved to {MODEL}")


if __name__ == "__main__":
    main()