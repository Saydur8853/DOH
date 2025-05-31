import pandas as pd
import numpy as np
import joblib
import os

# Load model and scaler
scaler_path = os.path.join(os.path.dirname(__file__), "../models/ssl_scaler.joblib")
model_path = os.path.join(os.path.dirname(__file__), "../models/ssl_oneclass_model.joblib")
scaler = joblib.load(scaler_path)
model = joblib.load(model_path)

# List of malicious CSV files
csv_files = [
    "iodine.csv",
    "dnscat2.csv",
    "dns2tcp.csv"
]

# Directory paths
data_dir = os.path.join(os.path.dirname(__file__), "../data/MaliciousDoH-CSVs")
output_dir = os.path.join(os.path.dirname(__file__), "../data")

# Process each file
for file_name in csv_files:
    csv_path = os.path.join(data_dir, file_name)
    mal_df = pd.read_csv(csv_path)
    print(f"[INFO] Loaded test data from {file_name}: {mal_df.shape}")

    # Drop rows with missing values
    mal_df.dropna(inplace=True)

    # Drop same non-numeric columns as in training
    non_numeric_cols = ['SourceIP', 'DestinationIP', 'SourcePort', 'DestinationPort', 'TimeStamp', 'label']
    mal_numeric = mal_df.drop(columns=non_numeric_cols, errors='ignore')

    # Ensure same feature columns as training
    expected_features = scaler.feature_names_in_  # sklearn >= 1.0
    mal_numeric = mal_numeric[expected_features]  # match order and names

    # Scale features
    X_test_scaled = scaler.transform(mal_numeric)

    # Predict
    preds = model.predict(X_test_scaled)

    # Analyze predictions
    anomaly_count = (preds == -1).sum()
    total = len(preds)
    print(f"🚨 {file_name}: Detected {anomaly_count} anomalies out of {total} samples.")

    # Save results
    mal_df["Prediction"] = preds
    mal_df["Anomaly"] = mal_df["Prediction"].apply(lambda x: "Malicious" if x == -1 else "Benign")

    output_path = os.path.join(output_dir, file_name.replace(".csv", "_predictions.csv"))
    mal_df.to_csv(output_path, index=False)
    print(f"[INFO] Prediction results saved to {output_path}")
