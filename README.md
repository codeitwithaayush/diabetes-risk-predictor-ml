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

```text
.
|-- data/diabetesdata.csv
|-- notebooks/diabetes_svm_analysis.ipynb
|-- src/train.py        <- train, evaluate, save model
|-- src/predict.py      <- predict one patient from the CLI
|-- requirements.txt
|-- LICENSE
```
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

Linear-kernel SVM, 80/20 stratified split (614 train / 154 test, `random_state=2`), features standardized inside a scikit-learn pipeline. Reproduce with `python src/train.py`.

| Metric | Value |
|---|---|
| Train accuracy | 78.66% |
| Test accuracy | 77.27% |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Non-diabetic (0) | 0.78 | 0.91 | 0.84 | 100 |
| Diabetic (1) | 0.76 | 0.52 | 0.62 | 54 |

**Takeaways**
- Train and test accuracy are close, so the model is not overfitting
- Recall on the diabetic class is only 0.52: about half of diabetic patients are missed. For a screening use case this matters more than overall accuracy
- Next steps (see Roadmap): impute zero-as-missing values, try non-linear kernels and tree ensembles, and optimize for recall or ROC-AUC
- The notebook uses its own split settings, so its numbers can differ slightly
## Limitations

- Educational project only, **not a medical tool** and not a substitute for professional diagnosis
- Small dataset (768 rows) from one population, so it may not generalize
- Missing values are stored as zeros and are not imputed in the baseline
- Accuracy alone can hide weak recall on the diabetic class
## Roadmap

- [ ] Impute zero-as-missing values (median or KNN)
- [ ] Compare against logistic regression, random forest and gradient boosting
- [ ] Tune hyperparameters with cross-validated grid search (RBF kernel, C, gamma)
- [ ] Add ROC-AUC, confusion matrix and recall-focused evaluation
- [ ] Build a Streamlit demo app
## Acknowledgements

This project started from [KalyanM45/Diabetes-Prediction-using-Machine-Learning](https://github.com/KalyanM45/Diabetes-Prediction-using-Machine-Learning). I restructured it, added the training and prediction scripts, and rewrote the documentation.

## License

Released under the GNU GPL v3.0, the same license as the original project. See [LICENSE](LICENSE).