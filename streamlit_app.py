import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("heart.csv")

# Split features and target
X = df.drop('target', axis=1)
y = df['target']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model (directly on Streamlit Cloud)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Health checker function
def health_status(bp, chol, fbs, age):
    issues = []
    if bp > 130:
        issues.append("High BP")
    if chol > 200:
        issues.append("High Cholesterol")
    if fbs == 1:
        issues.append("High Blood Sugar")
    if age > 45:
        issues.append("Age Risk")
    return "✅ You seem healthy!" if not issues else "⚠️ " + ", ".join(issues)

# Streamlit UI
st.title("💓 Heart Attack Risk & Health Status Evaluator")

# User Inputs
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

# Predict Button
if st.button("🔍 Evaluate My Risk"):
    user_input = np.array([[age, sex, cp, trtbps, chol, fbs, restecg,
                            thalachh, exng, oldpeak, slp, caa, thall]])
    
    user_scaled = scaler.transform(user_input)
    prediction = model.predict(user_scaled)[0]
    
    if prediction == 1:
        st.error("🚨 You are at HIGH RISK of heart attack.")
    else:
        st.success("✅ You are at LOW RISK of heart attack.")
    
    st.info("💡 Health Status: " + health_status(trtbps, chol, fbs, age))
