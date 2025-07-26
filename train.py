# train.py
import pandas as pd
from xgboost import XGBClassifier
import joblib

# Load data
df = pd.read_csv("data/sample_logs.csv")

# Prepare features and target
X = df[['latency', 'errors']]
y = df['dropout']

# Train model
model = XGBClassifier()
model.fit(X, y)

# Save model
joblib.dump(model, "model/model.pkl")