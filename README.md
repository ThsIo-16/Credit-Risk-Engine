# 🏦 Credit Risk Assessment & Prediction Engine

## 📌 Business Overview
In the financial sector, approving a loan for a customer who eventually defaults costs significantly more than rejecting a good customer. This project is an **End-to-End Machine Learning Pipeline** designed to predict the **Probability of Default (PD)** for loan applicants. 

The goal was to build a highly accurate, explainable risk model that balances the trade-off between identifying risky profiles (Recall) and maintaining profitable loan approvals (Precision).

## 🚀 Key Features & Methodology

* **Regulatory Data Preprocessing:** Cleaned and engineered features based on actual business logic (e.g., capping age limits to 75 based on standard banking regulations, removing employment length anomalies).
* **Machine Learning Pipeline:** Trained a `RandomForestClassifier` utilizing `RandomizedSearchCV` for hyperparameter tuning. Handled the Dummy Variable Trap (Multicollinearity) ensuring econometric soundness.
* **Imbalanced Data Handling:** Shifted focus from standard Accuracy to **F1-Score** and **Recall** for the minority class (Defaults), ensuring the model correctly identifies actual risks rather than simply predicting the majority class.
* **Explainable AI (XAI):** Extracted Feature Importances to understand the drivers behind defaults (e.g., Loan Interest Rate and Income).
* **Production-Ready Web App:** Deployed the model locally via a **Streamlit Dashboard**, allowing non-technical stakeholders to input applicant data and view the dynamic Probability of Default.

## 📊 Model Performance
The optimized Random Forest model achieved robust results on unseen test data:
* **Overall Accuracy:** `93.44%`
* **Precision (Defaults):** `96%` *(Extremely confident when flagging a risky loan)*
* **Recall (Defaults):** `73%` *(Successfully captures the majority of actual defaults)*

## 📂 Repository Structure
```text
Credit_Risk_Analysis/
│
├── data/
│   ├── raw/                 # Dataset
│   └── processed/           # Cleaned dataset
│
├── notebooks/
│   └── 01_Exploratory_Data_Analysis.ipynb  # EDA, Data Cleaning & Feature Importance
│
├── scripts/
│   ├── model_training.py    # Training script 
│   └── dashboard.py         # Streamlit Web Application 
│
├── src/
│   ├── risk_model.pkl       # Saved RF Model
│   └── model_features.pkl   # Saved feature columns for dummy alignment
│
└── README.md
```

## ⚙️ How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/ThsIo-16/Credit-Risk-Engine.git](https://github.com/ThsIo-16/Credit-Risk-Engine.git)
   cd Credit-Risk-Engine
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Dashboard:**
   ```bash
   streamlit run scripts/dashboard.py
   ```