# Diabetes Risk Predictor (SVM)

Machine-learning project that predicts whether a patient is likely to have diabetes from basic diagnostic measurements, using a **Support Vector Machine** classifier built with scikit-learn.

## Overview

- Binary classification: **diabetic (1)** vs **non-diabetic (0)**
- Model: linear-kernel SVM with standardized features
- Includes an exploratory notebook, a reproducible training script and a small command-line predictor
## Dataset

`data/diabetesdata.csv` is the Pima Indians Diabetes dataset: 768 patients, 8 numeric features and a binary `Outcome` label.

| Feature | Meaning |
|---|---|
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skin-fold thickness (mm) |
| Insulin | 2-hour serum insulin |
| BMI | Body mass index |
| DiabetesPedigreeFunction | Diabetes family-history score |
| Age | Age in years |
| Outcome | 1 = diabetic, 0 = not diabetic |

Note: zeros in Glucose, BloodPressure, SkinThickness, Insulin and BMI usually stand for missing readings in this dataset.
## Methodology

1. Load the CSV with pandas and explore it (matplotlib, seaborn)
2. Split into train and test sets (`train_test_split`)
3. Standardize features (`StandardScaler`) so no feature dominates the margin
4. Train `SVC(kernel='linear')`
5. Evaluate with accuracy on both train and test sets to check for overfitting