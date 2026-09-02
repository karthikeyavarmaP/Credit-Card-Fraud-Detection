# Credit Card Fraud Detection

A machine learning project that detects fraudulent credit card transactions using Python and scikit-learn.

## Project Overview

Credit card fraud detection is a highly imbalanced classification problem because fraudulent transactions represent only a very small portion of total transactions.

This project compares **Logistic Regression** and **Random Forest** models to identify fraudulent transactions and evaluates them using classification metrics that are more meaningful than accuracy alone.

## Dataset

The dataset contains:

- **284,807 total transactions**
- **284,315 legitimate transactions**
- **492 fraudulent transactions**
- **31 columns**
- Target column: `Class`

`Class = 0` represents a legitimate transaction.

`Class = 1` represents a fraudulent transaction.

The dataset is highly imbalanced, with fraud representing only a very small fraction of all transactions.

## Technologies Used

- Python
- Pandas
- Matplotlib
- Scikit-learn

## Machine Learning Workflow

1. Loaded and explored the transaction dataset using Pandas.
2. Examined the class distribution to identify class imbalance.
3. Separated input features from the fraud target variable.
4. Split the dataset into training and testing sets using stratified sampling.
5. Standardized the `Time` and `Amount` features.
6. Trained a Logistic Regression model with balanced class weights.
7. Trained a Random Forest classifier with balanced class weights.
8. Evaluated both models using precision, recall, F1-score, accuracy, and confusion matrices.
9. Compared the models based on their ability to identify fraudulent transactions.

## Class Distribution

The dataset contains a large imbalance between legitimate and fraudulent transactions:

```text
Legitimate Transactions: 284,315
Fraudulent Transactions:     492
```

Because of this imbalance, accuracy alone is not a reliable measure of fraud-detection performance.

## Model Results

### Logistic Regression

| Metric | Result |
|---|---:|
| Accuracy | 97.55% |
| Precision | 6.07% |
| Recall | 91.84% |
| F1 Score | 11.44% |

Confusion Matrix:

```text
[[55478  1386]
 [    8    90]]
```

The Logistic Regression model detected **90 of 98 fraudulent transactions** in the test set, resulting in high recall.

However, it also incorrectly classified **1,386 legitimate transactions as fraud**, leading to low precision.

### Random Forest

| Metric | Result |
|---|---:|
| Accuracy | 99.95% |
| Precision | 96.05% |
| Recall | 74.49% |
| F1 Score | 83.91% |

Confusion Matrix:

```text
[[56861     3]
 [   25    73]]
```

The Random Forest model detected **73 of 98 fraudulent transactions** while incorrectly flagging only **3 legitimate transactions**.

This resulted in a substantially higher precision and F1-score compared with Logistic Regression.

## Model Comparison

| Model | Precision | Recall | F1 Score |
|---|---:|---:|---:|
| Logistic Regression | 6.07% | 91.84% | 11.44% |
| Random Forest | 96.05% | 74.49% | 83.91% |

Logistic Regression achieved higher recall, meaning it identified a larger proportion of fraudulent transactions.

Random Forest provided a much better overall balance between detecting fraud and avoiding false fraud alerts.

## Why Accuracy Is Not Enough

The dataset is extremely imbalanced.

A model could classify almost every transaction as legitimate and still achieve very high accuracy.

For fraud detection, the following metrics are therefore especially important:

- **Precision:** Of all transactions predicted as fraud, how many were actually fraudulent?
- **Recall:** Of all actual fraudulent transactions, how many were detected?
- **F1 Score:** The balance between precision and recall.

## Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

## Running the Project

Place `creditcard.csv` in the same directory as the Python script.

Run:

```bash
python fraud_detection.py
```

The program will:

- Display the transaction class distribution
- Train Logistic Regression
- Train Random Forest
- Print model evaluation metrics
- Display confusion matrices and classification reports

## Project Structure

```text
Credit-Card-Fraud-Detection/
├── fraud_detection.py
├── requirements.txt
├── README.md
└── .gitignore
```

The dataset is not included in the repository because of its large file size.

## Key Learning Outcomes

This project demonstrates:

- Binary classification
- Imbalanced datasets
- Train/test splitting
- Feature standardization
- Logistic Regression
- Random Forest
- Class weighting
- Precision and recall
- F1-score
- Confusion matrices
- Model comparison

## Future Improvements

- Tune classification thresholds
- Perform hyperparameter optimization
- Analyze precision-recall curves
- Add ROC-AUC and PR-AUC evaluation
- Explore sampling techniques such as SMOTE
- Evaluate additional anomaly-detection models

## Author

**Karthikeyavarma**

IIT Madras
