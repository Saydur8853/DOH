import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
import joblib
import os
import time
import threading

# Define heartbeat spinner function
def heartbeat():
    while not stop_flag:
        print(".", end="", flush=True)
        time.sleep(2)

# Load the dataset
csv_path = os.path.join(os.path.dirname(__file__), "../data/ssl_zero_day_benign.csv")  # Adjust path if needed
df = pd.read_csv(csv_path)

print("[INFO] Loaded data:", df.shape, "rows,", list(df.columns))

# Check for missing values
missing_total = df.isnull().sum().sum()
print("[INFO] Total missing values:", missing_total)

# Drop rows with any missing values
df.dropna(inplace=True)
print("[INFO] After dropna:", df.shape)

# Drop non-numeric columns
non_numeric_cols = ['SourceIP', 'DestinationIP', 'SourcePort', 'DestinationPort', 'TimeStamp', 'label']
df_numeric = df.drop(columns=non_numeric_cols, errors='ignore')
print("[INFO] Drop non-numeric columns done")

# Scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_numeric)
print("[INFO] Feature scaling done")

# Train One-Class SVM with time and heartbeat
print("[INFO] Training One-Class SVM...")
start = time.time()
stop_flag = False
heartbeat_thread = threading.Thread(target=heartbeat)
heartbeat_thread.start()

model = OneClassSVM(kernel='rbf', gamma='auto')
model.fit(X_scaled)

stop_flag = True
heartbeat_thread.join()
end = time.time()
print(f"\n[INFO] Model training completed in {end - start:.2f} seconds.")

# Save the model and scaler
joblib.dump(model, "oneclass_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("[INFO] Model and scaler saved successfully.")
