from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import pandas as pd
import time
import threading
import os
import joblib
import numpy as np  # ✅ Required for saving .npy files

# Heartbeat spinner
def heartbeat():
    while not stop_flag:
        print(".", end="", flush=True)
        time.sleep(1)

# Load data
csv_path = os.path.join(os.path.dirname(__file__), "../data/ssl_zero_day_benign.csv")
df_full = pd.read_csv(csv_path)
print("[INFO] Total rows in CSV file:", df_full.shape[0])

# Slice for testing
df = df_full
print("[INFO] Loaded data:", df.shape)

# Drop non-numeric
non_numeric_cols = ['SourceIP', 'DestinationIP', 'SourcePort', 'DestinationPort', 'TimeStamp', 'label']
df_numeric = df.drop(columns=non_numeric_cols, errors='ignore')
df_numeric.dropna(inplace=True)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_numeric)

# Train One-Class SVM with simulated progress
model = OneClassSVM(kernel='rbf', gamma='auto')

print("[INFO] Training One-Class SVM...")
start = time.time()
stop_flag = False
heartbeat_thread = threading.Thread(target=heartbeat)
heartbeat_thread.start()

# Simulated training progress
batch_size = 10000
n_batches = X_scaled.shape[0] // batch_size

for i in range(n_batches):
    batch_data = X_scaled[i * batch_size:(i + 1) * batch_size]
    model.fit(batch_data)  # Not actual incremental training, for simulation only
    print(f"\n[INFO] Progress: {(i + 1) * batch_size}/{X_scaled.shape[0]} rows processed")

stop_flag = True
heartbeat_thread.join()
end = time.time()

print(f"\n[INFO] Model training simulation completed in {end - start:.2f} seconds.")

# Create models directory if it doesn't exist
model_dir = os.path.join(os.path.dirname(__file__), "../models")
os.makedirs(model_dir, exist_ok=True)

# Save scaler and model
joblib.dump(scaler, os.path.join(model_dir, "ssl_scaler.joblib"))
joblib.dump(model, os.path.join(model_dir, "ssl_oneclass_model.joblib"))
print("[INFO] Model and scaler saved.")

# ✅ Save features and dummy labels for visualization
np.save(os.path.join(model_dir, 'features.npy'), X_scaled)

# Create dummy label array (all benign = 1)
y_dummy = np.ones(X_scaled.shape[0])
np.save(os.path.join(model_dir, 'labels.npy'), y_dummy)
print("[INFO] features.npy and labels.npy saved in models/ for PCA visualization.")
