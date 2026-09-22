# Fraud Detection Using Machine Learning

## Project Overview

This project focuses on detecting fraudulent financial transactions using machine learning. The project uses the PaySim financial transaction dataset and follows an end-to-end machine learning workflow, including data exploration, preprocessing, model training, evaluation, and deployment through a Streamlit web application.

The main objective is to build a classification model that can identify potentially fraudulent transactions from transaction details.

## Dataset

The project uses the PaySim financial transaction dataset.

The dataset contains **6,362,620 transactions** and **11 original columns**.

The main features include:

* `step` — time step of the transaction
* `type` — type of transaction
* `amount` — transaction amount
* `nameOrig` — sender account identifier
* `oldbalanceOrg` — sender's balance before the transaction
* `newbalanceOrig` — sender's balance after the transaction
* `nameDest` — receiver account identifier
* `oldbalanceDest` — receiver's balance before the transaction
* `newbalanceDest` — receiver's balance after the transaction
* `isFraud` — target variable indicating whether the transaction is fraudulent
* `isFlaggedFraud` — existing fraud flag in the dataset

The target variable is highly imbalanced:

* Non-fraudulent transactions: **6,354,407**
* Fraudulent transactions: **8,213**

## Exploratory Data Analysis

Several exploratory analyses were performed to understand the dataset and identify patterns related to fraudulent transactions.

### Transaction Types

The dataset contains five transaction types:

* CASH_IN
* CASH_OUT
* DEBIT
* PAYMENT
* TRANSFER

The distribution of these transaction types was examined using visualization.

### Fraud Rate by Transaction Type

The fraud rate for each transaction type was calculated using the mean of the binary `isFraud` variable.

### Transaction Amount

The transaction amount was examined using descriptive statistics and visualizations.

Because the transaction amount is highly right-skewed, a log transformation using `log1p()` was used for visualization:

```python
np.log1p(df["amount"])
```

This transformation was used only for visualization and did not modify the original amount values.

### Fraud Over Time

Fraudulent transactions were examined across the `step` variable to observe how fraud cases were distributed over time.

After examining the distribution, the `step` column was removed before model development.

## Feature Engineering

Two balance-difference features were created during the analysis:

```python
df["balanceDiffOrig"] = (
    df["oldbalanceOrg"] - df["newbalanceOrig"]
)

df["balanceDiffDest"] = (
    df["newbalanceDest"] - df["oldbalanceDest"]
)
```

These features were examined as part of the exploratory analysis.

However, **they were not included in the final machine learning model**.

The final model was trained using the following numerical features:

* `amount`
* `oldbalanceOrg`
* `newbalanceOrig`
* `oldbalanceDest`
* `newbalanceDest`

and the categorical feature:

* `type`

## Data Preprocessing

The sender and receiver account identifiers were removed from the model:

```python
nameOrig
nameDest
```

The existing `isFlaggedFraud` column was also excluded from the model.

The categorical feature `type` was encoded using One-Hot Encoding.

The numerical features were standardized using `StandardScaler`.

A `ColumnTransformer` was used to apply the appropriate preprocessing to numerical and categorical features.

## Model Development

A **Logistic Regression** classifier was used for fraud detection.

A complete Scikit-learn pipeline was created containing:

1. Data preprocessing
2. Numerical feature scaling
3. One-hot encoding of transaction type
4. Logistic Regression classifier

The model used:

```python
LogisticRegression(
    class_weight="balanced",
    max_iter=1000
)
```

The `class_weight="balanced"` option was used to account for the severe imbalance between fraudulent and non-fraudulent transactions.

The dataset was divided into training and testing sets using a **70:30 split** with stratification.

## Model Evaluation

The model was evaluated using:

* Precision
* Recall
* F1-score
* Accuracy
* Confusion matrix

### Classification Report

The model achieved approximately:

| Class     | Precision | Recall | F1-score |
| --------- | --------: | -----: | -------: |
| Non-Fraud |      1.00 |   0.95 |     0.97 |
| Fraud     |      0.02 |   0.94 |     0.04 |

Overall accuracy:

**94.62%**

### Confusion Matrix

The confusion matrix was:

```text
[[1803851  102471]
 [    148    2316]]
```

This corresponds to:

* True Negatives: 1,803,851
* False Positives: 102,471
* False Negatives: 148
* True Positives: 2,316

The results show that the model achieved high recall for fraudulent transactions but produced a large number of false positives. The low fraud precision is particularly important because the dataset contains a very small proportion of fraudulent transactions.

Therefore, accuracy alone does not fully describe the model's performance on this highly imbalanced dataset.

## Streamlit Application

A Streamlit application was developed to demonstrate the trained model.

The application allows users to enter:

* Transaction type
* Transaction amount
* Sender's old balance
* Sender's new balance
* Receiver's old balance
* Receiver's new balance

The application then uses the saved machine learning pipeline to classify the transaction as either:

**Fraudulent Transaction**

or

**Non-Fraudulent Transaction**

The trained model is saved as:

```text
fraud_detection_pipeline.pkl
```

and loaded by the Streamlit application using:

```python
model = joblib.load("fraud_detection_pipeline.pkl")
```

## Project Structure

```text
fraud-detection-ml/
│
├── fraud_detection.ipynb
├── app.py
├── fraud_detection_pipeline.pkl
├── README.md
└── requirements.txt
```

### File Descriptions

**`fraud_detection.ipynb`**

Contains the complete data analysis, feature engineering, preprocessing, model training, and evaluation workflow.

**`app.py`**

Contains the Streamlit application used to interact with the trained model.

**`fraud_detection_pipeline.pkl`**

Contains the trained Scikit-learn machine learning pipeline saved using Joblib.

**`requirements.txt`**

Contains the Python packages required to run the project.

**`README.md`**

Contains the project documentation.

## Installation

Clone the repository and install the required packages:

```bash
pip install -r requirements.txt
```

## Running the Streamlit Application

Make sure `app.py` and `fraud_detection_pipeline.pkl` are in the same directory.

Run:

```bash
streamlit run app.py
```

The Streamlit application will then open in a browser.

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib
* Streamlit
* Google Colab

## Project Workflow

```text
Dataset
   ↓
Data Inspection
   ↓
Exploratory Data Analysis
   ↓
Feature Engineering
   ↓
Data Preprocessing
   ↓
Train-Test Split
   ↓
Logistic Regression
   ↓
Model Evaluation
   ↓
Save Trained Pipeline
   ↓
Streamlit Application
```

## Notes

The original dataset is not included in this repository because of its large file size.

The trained model and application are included so that the project structure and deployment workflow can be demonstrated without uploading the full dataset.

