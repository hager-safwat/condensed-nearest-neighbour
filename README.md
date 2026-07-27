# KNN with PCA and Complexity Reduction

A machine-learning coursework project studying how dimensionality
reduction and feature selection affect a K-Nearest-Neighbours classifier —
specifically the trade-off between model simplicity and accuracy.

## What it does
- Tunes KNN with `GridSearchCV` to find the best k.
- Applies **PCA** to reduce dimensionality and measures the effect on accuracy.
- Uses **RFECV** (Random-Forest estimator) for feature selection.
- Handles class imbalance with **down-sampling** and **SMOTE**.
- Compares performance (accuracy, precision/recall, F1, ROC) before and
  after each reduction step.

## Finding
PCA and feature selection simplify the model and cut computation, but on
this dataset they don't consistently beat the full-feature baseline — a
concrete illustration of the accuracy/complexity trade-off.

## Files
- `ML_Project-2.ipynb` — full analysis and experiments.
- `ML_project_Report.pdf` — written report with figures.

