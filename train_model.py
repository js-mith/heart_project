import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# 1. Load dataset
df = pd.read_csv("heart.csv")

# 2. Split into features and target
X = df.drop('target', axis=1)  # Features
y = df['target']               # Labels (0 or 1)

# 3. Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5. Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 6. Save model and scaler
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Model and scaler saved successfully.")
