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
## Project structure
.
data/diabetesdata.csv
notebooks/diabetes_svm_analysis.ipynb
src/train.py # train + evaluate + save model
src/predict.py # predict for one patient from the command line
requirements.txt
LICENSE
## Quick start

```bash
git clone https://github.com/codeitwithaayush/diabetes-risk-predictor-ml.git
cd diabetes-risk-predictor-ml
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python src/train.py
python src/predict.py --pregnancies 6 --glucose 148 --bloodpressure 72 --skinthickness 35 --insulin 0 --bmi 33.6 --diabetespedigreefunction 0.627 --age 50
```
## Results

- `python src/train.py` prints train accuracy, test accuracy and a per-class report (precision, recall, F1)
- The notebook shows the same train/test accuracy comparison step by step
- The script uses its own split settings (80/20, stratified, fixed seed), so its numbers can differ slightly from the notebook
## Limitations

- Educational project only, **not a medical tool** and not a substitute for professional diagnosis
- Small dataset (768 rows) from one population, so it may not generalize
- Missing values are stored as zeros and are not imputed in the baseline
- Accuracy alone can hide weak recall on the diabetic class