# Breast Cancer Classification – ML Comparative Study

## Overview
A machine learning project comparing three classification models to predict whether a breast tumour is malignant or benign using the UCI Breast Cancer Wisconsin dataset.

## Problem Statement
Early and accurate detection of malignant tumours is critical in clinical settings. This project evaluates which ML model best classifies tumours while minimising false negatives (missing a malignant case).

## Dataset
- **Source:** UCI Machine Learning Repository – Breast Cancer Wisconsin (Diagnostic)
- **Size:** 569 samples, 30 features
- **Target:** Malignant (1) or Benign (0)

## Models Compared
| Model | Accuracy |
|---|---|
| Logistic Regression | 98.25% |
| Random Forest | XX% |
| PCA + Logistic Regression | XX% |

## Key Steps
- Exploratory Data Analysis (EDA)
- Feature scaling and preprocessing
- Dimensionality reduction using PCA (95% variance retained)
- Model training and hyperparameter tuning
- Evaluation using Precision, Recall, F1-Score (focus on malignant class recall)

## Results
Logistic Regression achieved the highest accuracy of **98.25%**, demonstrating the dataset is near-linearly separable. PCA reduced dimensionality with minimal performance loss.

## Tech Stack
Python, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn, Jupyter Notebook

## How to Run
```bash
git clone https://github.com/Annei92/breast-cancer-classification
cd breast-cancer-classification
pip install -r requirements.txt
jupyter notebook
```
