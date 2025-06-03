import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# -------------------------------------------
# 🔹 Utility: Save Plot Function
# -------------------------------------------
def save_plot(path, title=None):
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"[INFO] Saved: {os.path.basename(path)}")

# -------------------------------------------
# 🔹 Output Directory
# -------------------------------------------
output_dir = "visualize"
os.makedirs(output_dir, exist_ok=True)
print("[INFO] Output directory ensured.")

# -------------------------------------------
# 🔹 Load Features and Labels
# -------------------------------------------
print("[INFO] Loading data...")
X = np.load('../models/features.npy')
y = np.load('../models/labels.npy')  # 1 = benign, -1 = anomaly
y_true = np.where(y == -1, 1, 0)     # 1 = anomaly, 0 = benign
print(f"[INFO] X shape: {X.shape}, y shape: {y.shape}")

# -------------------------------------------
# 🔹 Preprocessing
# -------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
print(f"[INFO] Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# -------------------------------------------
# 🔹 Train One-Class SVM Ensemble
# -------------------------------------------
chunk_size = 10000
models = []
print("[INFO] Training One-Class SVM in chunks...")
for i in range(0, X_train.shape[0], chunk_size):
    chunk = X_train[i:i+chunk_size]
    model = OneClassSVM(kernel='rbf', nu=0.05, gamma='scale')
    model.fit(chunk)
    models.append(model)
    print(f"  Trained chunk {i}–{i+len(chunk)}")

# -------------------------------------------
# 🔹 Ensemble Prediction Function
# -------------------------------------------
def ensemble_predict(X, models):
    scores = np.zeros(X.shape[0])
    for m in tqdm(models, desc="[INFO] Ensemble prediction"):
        scores += m.decision_function(X)
    scores /= len(models)
    return np.where(scores < 0, -1, 1), scores

# Predict and get decision scores
y_pred, decision_scores = ensemble_predict(X_test, models)
y_pred_bin = np.where(y_pred == -1, 1, 0)

# -------------------------------------------
# 🔹 PCA Visualization
# -------------------------------------------
print("[INFO] Running PCA...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_test)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[y_test == 1, 0], X_pca[y_test == 1, 1], label='Benign', alpha=0.6, c='green')
plt.scatter(X_pca[y_test == -1, 0], X_pca[y_test == -1, 1], label='Anomaly', alpha=0.6, c='red')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()
plt.grid(True)
save_plot(os.path.join(output_dir, "pca_scatter.png"), "PCA Scatter Plot of DoH Traffic")

# -------------------------------------------
# 🔹 Decision Boundary in PCA Space
# -------------------------------------------
print("[INFO] Plotting decision boundary in PCA space...")
xx, yy = np.meshgrid(
    np.linspace(X_pca[:, 0].min(), X_pca[:, 0].max(), 500),
    np.linspace(X_pca[:, 1].min(), X_pca[:, 1].max(), 500)
)
grid_points = np.c_[xx.ravel(), yy.ravel()]

svm_pca = OneClassSVM(kernel='rbf', nu=0.05, gamma='scale')
svm_pca.fit(X_pca)

Z_vals = []
for i in tqdm(range(0, grid_points.shape[0], 5000), desc="Computing decision surface"):
    Z_vals.extend(svm_pca.decision_function(grid_points[i:i+5000]))
Z = np.array(Z_vals).reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.Blues_r)
plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='red')
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_test, cmap='coolwarm', edgecolors='k', s=10)
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
save_plot(os.path.join(output_dir, "decision_boundary.png"), "Decision Boundary (PCA Space)")

print("[✅] All tasks completed successfully.")
