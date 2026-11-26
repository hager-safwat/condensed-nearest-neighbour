# KNN with PCA, Feature Selection and Condensed Nearest Neighbour

A machine-learning coursework project on how dimensionality reduction,
feature selection and data reduction affect a K-Nearest-Neighbours
classifier: the trade-off between model simplicity and accuracy.

## What it does

- Compares baselines (logistic regression, KNN, SVM, decision tree, gradient
  boosting, MLP, naive Bayes) with classification reports, ROC and
  precision-recall curves.
- Tunes KNN with `GridSearchCV` (k from 1 to 29, 5-fold).
- Applies **PCA** (95 % of variance) and measures the effect on accuracy.
- Uses **RFECV** with a Random Forest estimator for feature selection.
- Handles class imbalance with **down-sampling** and **SMOTE**.
- Reduces the training set with **Condensed Nearest Neighbour** (CNN), an
  iterative 1-NN data-reduction algorithm, and re-evaluates.

## Results

From the notebook's stored outputs (test set of 1000 samples unless noted):

| Model | Accuracy |
|---|---|
| Logistic regression | 0.87 |
| SVM (RBF, default) | 0.88 |
| KNN, k = 5 | 0.81 |
| KNN, best k = 29 (5-fold CV on training data) | 0.848 (CV) |
| 1-NN, before / after PCA | 0.768 / 0.772 |
| KNN, k = 5, after CNN data reduction (181 test samples) | 0.67 |
| KNN, best k = 24 after CNN reduction | 0.794 (CV) |

**Finding:** PCA and feature selection simplify the model and cut
computation, but on this dataset they do not consistently beat the
full-feature baseline, and CNN data reduction costs a clear drop in accuracy
in exchange for a much smaller training set.

## Dataset

[Waveform Database Generator (Version 1)](https://archive.ics.uci.edu/dataset/107/waveform+database+generator+version+1)
(UCI Machine Learning Repository, from Breiman et al., *Classification and
Regression Trees*, 1984): 5000 instances, 21 noisy continuous attributes,
3 balanced wave classes. Download the zip from the UCI page and place it next
to the notebook; it is not included in this repository.

## Run it

```bash
pip install -r requirements.txt
jupyter notebook ML_Project-2.ipynb
```

The notebook was written for Google Colab: set `zip_file_path` in the first
cell to the downloaded zip. The data files are in the old Unix `.Z` format and
are decompressed with the `uncompress` command, available by default on
macOS and Colab (on Debian or Ubuntu: `sudo apt install ncompress`).

## Files

- `ML_Project-2.ipynb`: analysis and experiments, with outputs.
- `ML_project_Report.pdf`: written report with figures.
- `requirements.txt`: Python packages.
