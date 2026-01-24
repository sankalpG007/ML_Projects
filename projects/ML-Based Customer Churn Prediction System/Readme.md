# 📉 Customer Churn Analysis & Prediction

A Machine Learning project designed to identify customers at risk of leaving a service provider. This project uses the **Telco Customer Churn dataset** to build a predictive model and extract actionable business insights.

## 🎯 Project Objective
The goal is to predict customer behavior (Churn vs. No Churn) by analyzing demographic data, account information, and service usage patterns. This allows businesses to implement proactive retention strategies.

## 🛠️ Tech Stack
* **Language:** Python
* **Environment:** VS Code (Jupyter Notebooks)
* **Libraries:** * `Scikit-Learn` (Machine Learning)
    * `Pandas` & `NumPy` (Data Manipulation)
    * `Seaborn` & `Matplotlib` (Advanced Visualization)

## 📊 Key Insights & Visualizations
This project goes beyond simple prediction by implementing:
* **Feature Importance:** Identifying that 'Contract Type', 'Tenure', and 'Total Charges' are the primary drivers of churn.
* **Correlation Heatmap:** Visualizing the relationship between service features (like Fiber Optic internet) and churn rates.
* **Churn Distribution:** Highlighting that Month-to-Month customers represent the highest risk group.



## 🧠 Machine Learning Pipeline
1.  **Data Cleaning:** Handled missing values in `TotalCharges` and removed non-predictive features like `customerID`.
2.  **Encoding:** Converted categorical data into numerical format using `LabelEncoder`.
3.  **Scaling:** Standardized features using `StandardScaler` to ensure the model isn't biased by large numerical values.
4.  **Modeling:** Employed a **Random Forest Classifier**, an ensemble method that provides high stability and accuracy.
5.  **Evaluation:** Used Confusion Matrices and Precision-Recall curves to measure performance.

**Current Model Accuracy:** ~80%

## 📂 Project Structure
```text
├── Churn_Analysis.ipynb    # Full analysis and model code
├── Telco_Churn_Data.csv    # Raw dataset
├── README.md               # Project documentation
└── requirements.txt        # List of dependencies
