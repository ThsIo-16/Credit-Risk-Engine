"""
Streamlit Web Dashboard for Credit Risk Prediction
"""
import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Credit Risk Predictor", page_icon="🏦", layout="centered")

script_dir = os.path.dirname(os.path.abspath(__file__))

# where to find dashboard
path_a = os.path.join(script_dir, 'risk_model.pkl')
path_b = os.path.join(script_dir, 'src', 'risk_model.pkl')

if os.path.exists(path_a):
    model_path = path_a
    features_path = os.path.join(script_dir, 'model_features.pkl')
elif os.path.exists(path_b):
    model_path = path_b
    features_path = os.path.join(script_dir, 'src', 'model_features.pkl')
else:
    current_working_dir = os.getcwd()
    model_path = os.path.join(current_working_dir, 'src', 'risk_model.pkl')
    features_path = os.path.join(current_working_dir, 'src', 'model_features.pkl')

# model loading
@st.cache_resource
def load_model_and_features():
    try:
        model = joblib.load(model_path)
        features = joblib.load(features_path)
        return model, features
    except FileNotFoundError:
        st.error(f"🚨 Unsuccessful Loading: Model not Found!")
        st.warning(f"Looked at: {model_path}")
        st.info("💡 Tip: Look for where risk_model.pkl is saved and make sure you ran model_training.py")
        st.stop()

rf_model, expected_features = load_model_and_features()


# Dashboard
st.title("🏦 Credit Risk Prediction Engine")
st.markdown("""
Welcome to the interactive Risk Assessment Dashboard. 
Adjust the applicant's parameters below to see the **Probability of Default** in real-time.
""")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Applicant Profile")
    age = st.slider("Age", 18, 75, 30)
    income = st.number_input("Annual Income (€)", min_value=10000, max_value=200000, value=50000, step=5000)
    emp_length = st.slider("Employment Length (Years)", 0, 40, 5)
    home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])

with col2:
    st.subheader("💳 Loan Details")
    loan_intent = st.selectbox("Loan Intent", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
    loan_amount = st.number_input("Loan Amount (€)", min_value=500, max_value=50000, value=10000, step=500)
    interest_rate = st.slider("Interest Rate (%)", 1.0, 25.0, 10.5, 0.1)

st.divider()

#Prediction Logic
if st.button("Calculate Risk Score 🎯", type="primary", use_container_width=True):
    
    # Put the inputs into a DataFrame
    input_data = pd.DataFrame({
        'person_age': [age],
        'person_income': [income],
        'person_emp_length': [emp_length],
        'loan_amnt': [loan_amount],
        'loan_int_rate': [interest_rate],
        'person_home_ownership': [home_ownership],
        'loan_intent': [loan_intent]
    })

    input_encoded = pd.get_dummies(input_data)
    input_aligned = input_encoded.reindex(columns=expected_features, fill_value=0)

    # Make the prediction 
    default_probability = rf_model.predict_proba(input_aligned)[0][1]

    # Results Display
    st.subheader("📊 Assessment Result")
    
    prob_percent = default_probability * 100
    #Results interpretation
    if prob_percent < 20:
        st.success(f"**Low Risk:** Probability of Default is {prob_percent:.1f}%")
    elif 20 <= prob_percent < 50:
        st.warning(f"**Medium Risk:** Probability of Default is {prob_percent:.1f}%. Further review recommended.")
    else:
        st.error(f"**HIGH RISK:** Probability of Default is {prob_percent:.1f}%. Recommended to reject.")
        
    st.progress(default_probability)