import pandas as pd
import numpy as np
import joblib
import os
import tensorflow as tf
from tqdm import tqdm  # For progress bars

# Load models and scaler
print("📦 Loading models and scaler...")
model_dir = os.path.join(os.path.dirname(__file__), "../models")

# Load VAE encoder for feature extraction
encoder_path = os.path.join(model_dir, "vae_encoder.h5")
encoder = tf.keras.models.load_model(encoder_path)

# Load scaler and OneClass SVM model
scaler_path = os.path.join(model_dir, "ssl_scaler.joblib")
# model_path = os.path.join(model_dir, "ssl_oneclass_model.joblib")
model_path = os.path.join(model_dir, "iso_forest_model.joblib")
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

# Columns to drop (same as training)
non_numeric_cols = ['SourceIP', 'DestinationIP', 'SourcePort', 
                   'DestinationPort', 'TimeStamp', 'label']

# Process each malicious file
for file_name in csv_files:
    print(f"\n🔍 Processing {file_name}...")
    csv_path = os.path.join(data_dir, file_name)
    
    # Load and preprocess data
    mal_df = pd.read_csv(csv_path)
    mal_df.dropna(inplace=True)
    
    # Ensure same features as training
    mal_numeric = mal_df.drop(columns=non_numeric_cols, errors='ignore')
    mal_numeric = mal_numeric[scaler.feature_names_in_]  # Match training feature order
    
    # Scale features
    X_test_scaled = scaler.transform(mal_numeric)
    
    # Extract latent features using VAE encoder
    print("🔮 Extracting latent features with VAE...")
    X_test_latent = encoder.predict(X_test_scaled, batch_size=1024, verbose=1)
    
    # Predict anomalies using OneClass SVM
    print("🛡️ Running anomaly detection...")
    preds = model.predict(X_test_latent)
    
    # Analyze results
    anomaly_count = (preds == -1).sum()
    total = len(preds)
    detection_rate = anomaly_count / total * 100
    
    print(f"🚨 {file_name}:")
    print(f"   - Detected {anomaly_count}/{total} anomalies ({detection_rate:.2f}%)")
    
    # Save results with original data + predictions
    mal_df["Prediction"] = preds
    mal_df["Anomaly"] = mal_df["Prediction"].apply(lambda x: "Malicious" if x == -1 else "Benign")
    
    output_path = os.path.join(output_dir, file_name.replace(".csv", "_predictions.csv"))
    mal_df.to_csv(output_path, index=False)
    print(f"💾 Saved predictions to {output_path}")

print("\n✅ All malicious files processed!")



# This script performs anomaly detection on a set of malicious DNS traffic CSV files using a pre-trained Variational Autoencoder (VAE) encoder and an Isolation Forest model.

# For each malicious file (iodine.csv, dnscat2.csv, and dns2tcp.csv), the script:

# Loads the data and removes missing values.

# Drops non-numeric and irrelevant columns to match the training features.

# Scales the data using a previously saved StandardScaler.

# Uses the VAE encoder to extract latent features from the scaled data.

# Applies the Isolation Forest model to detect anomalies.

# Labels each record as "Malicious" (if prediction is -1) or "Benign" (if prediction is 1).

# Saves the results (with predictions and anomaly labels) into a new CSV file for each input.

# The script helps identify potentially malicious patterns in DNS traffic and outputs labeled data for further analysis or reporting.