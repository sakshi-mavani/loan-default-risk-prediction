# Loan Default Risk Prediction

## Project Overview
This project focuses on predicting **loan default risk** using structured financial data.  
The objective is to design an **end-to-end machine learning pipeline** that reflects how credit risk problems are handled in real-world banking and fintech environments.

The system helps identify customers who are more likely to default on a loan, enabling financial institutions to make informed lending decisions.

---

## Business Context
Loan defaults represent a significant financial risk for banks and lending institutions.  
Accurately identifying high-risk applicants allows organizations to:
- Minimize financial losses  
- Improve credit approval strategies  
- Apply risk-based pricing or additional verification  

This project is designed from a **risk analytics and business decision-making perspective**, not just a modeling exercise.

---

## Dataset Description
- **Dataset Type:** Loan Default (CSV)
- **Target Variable:** `Status`
  - `1` → Default  
  - `0` → Non-default  

The dataset contains a mix of numerical and categorical features related to:
- Loan amount and interest rate  
- Credit history and borrower profile  
- Loan purpose and type  

---

## Methodology

### 1. Data Understanding
Initial exploration was performed to understand:
- Dataset structure and feature types  
- Class imbalance in the target variable  

**Script:** `01_data_understanding.py`

---

### 2. Data Cleaning
Data preprocessing included:
- Handling missing values using appropriate statistical methods  
- Removing non-informative identifier columns  
- Creating a clean and consistent dataset  

**Script:** `02_data_cleaning.py`

---

### 3. Feature Encoding
Categorical variables were converted into numerical format using **One-Hot Encoding**, ensuring compatibility with machine learning models.

**Script:** `03_encoding.py`

---

### 4. Baseline Model: Logistic Regression
A logistic regression model was implemented as a baseline to establish initial performance.  
Stratified train-test split was used to preserve class distribution.

**Script:** `04_model_training.py`

---

### 5. Random Forest Model
A Random Forest classifier was trained to capture non-linear relationships in the data.  
This model demonstrated improved performance compared to the baseline.

**Script:** `05_random_forest.py`

---

### 6. Feature Importance Analysis
Feature importance was extracted to identify the most influential factors contributing to loan default risk.  
This step supports **model interpretability and business understanding**.

**Script:** `06_feature_importance.py`

---

### 7. XGBoost Model
An XGBoost classifier was implemented as the final model due to its strong performance on structured and imbalanced datasets.  
The model was evaluated using business-relevant metrics and saved for future deployment.

**Script:** `07_xgboost_model.py`

---

## Model Evaluation
Models were evaluated using multiple metrics to ensure business relevance:
- Accuracy  
- Precision  
- Recall  
- F1-score  
- ROC-AUC  
- Confusion Matrix  

Special emphasis was placed on **recall for default cases**, as missing a defaulter has higher business cost than flagging a low-risk customer.

---

## Project Structure


---

## Tools and Technologies
- Python  
- Pandas, NumPy  
- Scikit-learn  
- XGBoost  
- Joblib  

---

## Key Takeaways
- Built a complete machine learning pipeline from raw data to deployment-ready model  
- Addressed class imbalance using appropriate evaluation strategies  
- Applied advanced ensemble models for improved predictive performance  
- Focused on interpretability and business impact rather than accuracy alone  

---

## Conclusion
This project demonstrates a **practical and industry-aligned approach** to loan default prediction.  
It reflects how data scientists work with financial risk problems by combining technical modeling skills with business-oriented evaluation and decision-making.

---

## Author
**Sakshi Mavani**  
