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

# Select numerical features only
X_benign = benign.select_dtypes(include=[np.number]).drop(columns=["label"], errors='ignore')
X_malicious = malicious.select_dtypes(include=[np.number])

# Combine and scale
X_all = pd.concat([X_benign, X_malicious], ignore_index=True)
X_scaled = scaler.transform(X_all)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Predict using One-Class SVM
preds = model.predict(X_scaled)

# Assign labels for plotting
labels = ["Benign"] * len(X_benign) + ["Malicious"] * len(X_malicious)

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
