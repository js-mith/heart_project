import streamlit as st
import numpy as np
import joblib

# Load the trained model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# Health status checker
def health_status(bp, chol, fbs, age):
    issues = []
    if bp > 130:
        issues.append("High BP")
    if chol > 200:
        issues.append("High Cholesterol")
    if fbs == 1:
        issues.append("High Blood Sugar")
    if age > 45:
        issues.append("Age-Related Risk")
    return "✅ You seem healthy!" if not issues else "⚠️ " + ", ".join(issues)

# Title
st.title("💓 Heart Attack Risk & Health Status Evaluator")

# Input fields
age = st.slider("Age", 20, 80)
sex = st.radio("Sex", ["Male", "Female"])
sex = 1 if sex == "Male" else 0

cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
trtbps = st.number_input("Resting Blood Pressure (trtbps)", min_value=90, max_value=200, value=120)
chol = st.number_input("Cholesterol (chol)", min_value=100, max_value=400, value=200)
fbs = st.radio("Fasting Blood Sugar > 120 mg/dl (fbs)", [1, 0])
restecg = st.selectbox("Resting ECG Result (restecg)", [0, 1, 2])
thalachh = st.slider("Maximum Heart Rate Achieved (thalachh)", 70, 210, 150)
exng = st.radio("Exercise-Induced Angina (exng)", [1, 0])
oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=6.0, step=0.1)
slp = st.selectbox("Slope of ST Segment (slp)", [0, 1, 2])
caa = st.selectbox("Number of Major Vessels (caa)", [0, 1, 2, 3, 4])
thall = st.selectbox("Thalassemia Type (thall)", [0, 1, 2, 3])

# Predict button
if st.button("🔍 Evaluate My Risk"):
    # Prepare input
    user_input = np.array([[age, sex, cp, trtbps, chol, fbs, restecg,
                            thalachh, exng, oldpeak, slp, caa, thall]])
    
    # Scale and predict
    scaled_input = scaler.transform(user_input)
    prediction = model.predict(scaled_input)[0]

    # Display prediction
    if prediction == 1:
        st.error("🚨 You are at HIGH RISK of heart attack.")
    else:
        st.success("✅ You are at LOW RISK of heart attack.")
    
    # Show health warning (rule-based)
    health = health_status(trtbps, chol, fbs, age)
    st.info(f"💡 Health Status: {health}")
