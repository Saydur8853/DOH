import pandas as pd
import numpy as np
import joblib

# Load model and scaler
scaler = joblib.load("../models/ssl_scaler.joblib")
model = joblib.load("../models/ssl_oneclass_model.joblib")

# Load malicious test data
mal_df = pd.read_csv("../data/MaliciousDoH-CSVs/iodine.csv")
X_test = mal_df.select_dtypes(include=[np.number])

# Preprocess and predict
X_test_scaled = scaler.transform(X_test)
preds = model.predict(X_test_scaled)

# Analyze predictions
anomaly_count = (preds == -1).sum()
total = len(preds)
print(f"🚨 Detected {anomaly_count} anomalies out of {total} samples.")
