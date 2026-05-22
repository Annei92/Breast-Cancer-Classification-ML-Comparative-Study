# Breast Cancer Classification – Machine Learning Comparative Study

## Overview
A university machine learning coursework (University of London) comparing three classification approaches to predict whether a breast tumour is **malignant or benign**, using the UCI Breast Cancer Wisconsin (Diagnostic) dataset.

## Problem Statement
Missing a malignant tumour in clinical diagnosis can have life-threatening consequences. This project evaluates which ML model best classifies tumours, with a specific focus on maximising **recall for the malignant class** to minimise false negatives.

## Dataset
- **Source:** UCI Breast Cancer Wisconsin (Diagnostic) – loaded via `sklearn.datasets`
- **Size:** 569 samples, 30 numerical features (cell nucleus measurements from FNA images)
- **Class distribution:** 357 benign, 212 malignant
- **Task:** Binary classification (malignant vs benign)

## Models Compared

| Model | Accuracy | Precision (Malignant) | Recall (Malignant) | F1 (Malignant) |
|---|---|---|---|---|
| Logistic Regression | **98.25%** | 97.62% | 97.62% | 97.62% |
| PCA + Logistic Regression | 97.37% | 95.35% | **97.62%** | 96.47% |
| Random Forest | 95.61% | 95.12% | 92.86% | 93.98% |

## Key Techniques
- Train/test split (80/20) with stratification
- StandardScaler for feature normalisation
- PCA retaining 95% variance for dimensionality reduction
- Stratified 5-fold cross-validation (PCA + LR emerged most consistent: 97.72% mean accuracy)
- Custom evaluation function prioritising malignant class as positive label
- **Manual implementation of Logistic Regression from scratch using NumPy** with batch gradient descent, L2 regularisation, and gradient verification — matched sklearn's performance exactly (98.25% accuracy)

## Results Summary
- Logistic Regression achieved the highest single-split accuracy (98.25%)
- PCA + Logistic Regression was the most **consistent and generalisable** model under cross-validation (lowest std deviation across folds)
- Random Forest underperformed — the dataset's near-linear separability favoured simpler models
- Manual NumPy implementation matched sklearn identically, confirming correct gradient descent implementation

## Tech Stack
Python, Scikit-learn, NumPy, Pandas, Matplotlib, Jupyter Notebook

## How to Run
```bash
git clone https://github.com/Annei92/breast-cancer-classification
cd breast-cancer-classification
pip install -r requirements.txt
jupyter notebook
```

## References
- Breiman (2001) – Random Forests
- Jolliffe (2002) – Principal Component Analysis
- Pedregosa et al. (2011) – Scikit-learn
