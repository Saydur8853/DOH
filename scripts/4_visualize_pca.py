import pandas as pd
import numpy as np
import joblib
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

# Load models
scaler = joblib.load("../models/ssl_scaler.joblib")
model = joblib.load("../models/ssl_oneclass_model.joblib")

# Load datasets
benign = pd.read_csv("../data/ssl_zero_day_benign.csv")
malicious = pd.read_csv("../data/MaliciousDoH-CSVs/iodine.csv")

# Ensure both datasets have the same features (numerical and used during training)
expected_features = scaler.feature_names_in_  # This comes from the training scaler

# Drop non-numeric or irrelevant columns
benign = benign[expected_features]
malicious = malicious[expected_features]

# Handle missing values (drop or fill)
benign.dropna(inplace=True)
malicious.dropna(inplace=True)

# Combine both datasets
X_all = pd.concat([benign, malicious], ignore_index=True)

# Scale the data
X_scaled = scaler.transform(X_all)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Predict using One-Class SVM
preds = model.predict(X_scaled)

# Assign labels for plotting
labels = ["Benign"] * len(benign) + ["Malicious"] * len(malicious)

# Adjust labels in case rows dropped due to NaNs
labels = labels[:len(X_pca)]

# Plotting
plt.figure(figsize=(10, 7))
for label in set(labels):
    idx = [i for i, l in enumerate(labels) if l == label]
    plt.scatter(X_pca[idx, 0], X_pca[idx, 1], label=label, alpha=0.6)

# Highlight anomalies
anomaly_idx = np.where(preds == -1)[0]
plt.scatter(X_pca[anomaly_idx, 0], X_pca[anomaly_idx, 1],
            facecolors='none', edgecolors='r', label="Anomalies", linewidths=1.5)

plt.title("PCA of Benign vs Malicious DoH Traffic")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
