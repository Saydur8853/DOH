# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
# from sklearn.svm import OneClassSVM
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split

# # -------------------------------------------
# # 🔹 Create 'visualize' folder if not exists
# # -------------------------------------------
# output_dir = "visualize"
# os.makedirs(output_dir, exist_ok=True)
# print("[INFO] Output directory created.")

# # -------------------------------------------
# # 🔹 Step 1: Load preprocessed features and labels
# # -------------------------------------------
# print("[INFO] Loading features and labels...")
# X = np.load('../models/features.npy')
# y = np.load('../models/labels.npy')  # 1 for benign, -1 for anomaly

# y_true = np.where(y == -1, 1, 0)  # 1 = anomaly, 0 = benign
# print(f"[INFO] Features shape: {X.shape}, Labels shape: {y.shape}")

# # -------------------------------------------
# # 🔹 Step 2: Preprocessing
# # -------------------------------------------
# print("[INFO] Preprocessing features...")
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
# print(f"[INFO] Train/Test split: {X_train.shape}, {X_test.shape}")

# # -------------------------------------------
# # 🔹 Step 3: Train One-Class SVM
# # -------------------------------------------
# print("[INFO] Training One-Class SVM...")
# model = OneClassSVM(kernel='rbf', nu=0.05, gamma='scale')
# model.fit(X_train)

# print("[INFO] Predicting with One-Class SVM...")
# y_pred = model.predict(X_test)
# y_pred_bin = np.where(y_pred == -1, 1, 0)
# decision_scores = model.decision_function(X_test)
# print("[INFO] Prediction complete.")

# # -------------------------------------------
# # 🔹 Step 4: PCA for 2D projection
# # -------------------------------------------
# print("[INFO] Performing PCA projection...")
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X_test)

# # ✅ 1. PCA Scatter Plot
# print("[INFO] Saving PCA scatter plot...")
# plt.figure(figsize=(8, 6))
# plt.scatter(X_pca[y_test == 1, 0], X_pca[y_test == 1, 1], label='Benign', alpha=0.6, c='green')
# plt.scatter(X_pca[y_test == -1, 0], X_pca[y_test == -1, 1], label='Anomaly', alpha=0.6, c='red')
# plt.title('PCA Scatter Plot of DoH Traffic')
# plt.xlabel('PCA Component 1')
# plt.ylabel('PCA Component 2')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig(os.path.join(output_dir, "pca_scatter.png"))
# plt.close()
# print("[INFO] PCA scatter plot saved.")

# # ✅ 2. Decision Boundary (PCA space)
# print("[INFO] Calculating decision boundary in PCA space...")
# xx, yy = np.meshgrid(np.linspace(X_pca[:, 0].min(), X_pca[:, 0].max(), 500),
#                      np.linspace(X_pca[:, 1].min(), X_pca[:, 1].max(), 500))
# grid_points = np.c_[xx.ravel(), yy.ravel()]

# svm_pca = OneClassSVM(kernel='rbf', nu=0.05, gamma='scale')
# svm_pca.fit(X_pca)
# Z = svm_pca.decision_function(grid_points).reshape(xx.shape)

# plt.figure(figsize=(8, 6))
# plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.Blues_r)
# plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='red')
# plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_test, cmap='coolwarm', edgecolors='k')
# plt.title("One-Class SVM Decision Boundary in PCA Space")
# plt.xlabel('PCA Component 1')
# plt.ylabel('PCA Component 2')
# plt.tight_layout()
# plt.savefig(os.path.join(output_dir, "decision_boundary.png"))
# plt.close()
# print("[INFO] Decision boundary saved.")

# # ✅ 3. t-SNE Scatter Plot
# print("[INFO] Performing t-SNE projection...")
# tsne = TSNE(n_components=2, perplexity=30, random_state=42)
# X_tsne = tsne.fit_transform(X_test)

# plt.figure(figsize=(8, 6))
# plt.scatter(X_tsne[y_test == 1, 0], X_tsne[y_test == 1, 1], label="Benign", alpha=0.6, c='blue')
# plt.scatter(X_tsne[y_test == -1, 0], X_tsne[y_test == -1, 1], label="Anomaly", alpha=0.6, c='orange')
# plt.title('t-SNE Visualization of DoH Traffic')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig(os.path.join(output_dir, "tsne_plot.png"))
# plt.close()
# print("[INFO] t-SNE plot saved.")

# # ✅ 4. Confusion Matrix
# print("[INFO] Generating confusion matrix...")
# cm = confusion_matrix(y_true, y_pred_bin)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign", "Anomaly"])
# disp.plot(cmap=plt.cm.Blues)
# plt.title("Confusion Matrix")
# plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
# plt.close()
# print("[INFO] Confusion matrix saved.")

# # ✅ 5. ROC Curve
# print("[INFO] Generating ROC curve...")
# fpr, tpr, _ = roc_curve(y_true, decision_scores)
# roc_auc = auc(fpr, tpr)

# plt.figure(figsize=(8, 6))
# plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}', color='darkorange')
# plt.plot([0, 1], [0, 1], 'k--')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve for One-Class SVM')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig(os.path.join(output_dir, "roc_curve.png"))
# plt.close()
# print("[INFO] ROC curve saved.")

# # ✅ 6. PCA Explained Variance
# print("[INFO] Plotting PCA explained variance...")
# explained_variance = PCA().fit(X_test).explained_variance_ratio_

# plt.figure(figsize=(8, 6))
# plt.bar(range(1, len(explained_variance) + 1), explained_variance, color='teal')
# plt.xlabel('Principal Component')
# plt.ylabel('Explained Variance Ratio')
# plt.title('PCA Explained Variance (Feature Importance)')
# plt.tight_layout()
# plt.savefig(os.path.join(output_dir, "pca_variance.png"))
# plt.close()
# print("[INFO] PCA explained variance plot saved.")

# print("[INFO] All visualizations completed successfully.")
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import threading

from openTSNE import TSNE


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

# # -------------------------------------------
# # 🔹 PCA Visualization
# # -------------------------------------------
# print("[INFO] Running PCA...")
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X_test)

# plt.figure(figsize=(8, 6))
# plt.scatter(X_pca[y_test == 1, 0], X_pca[y_test == 1, 1], label='Benign', alpha=0.6, c='green')
# plt.scatter(X_pca[y_test == -1, 0], X_pca[y_test == -1, 1], label='Anomaly', alpha=0.6, c='red')
# plt.xlabel('PCA 1')
# plt.ylabel('PCA 2')
# plt.legend()
# plt.grid(True)
# save_plot(os.path.join(output_dir, "pca_scatter.png"), "PCA Scatter Plot of DoH Traffic")

# # -------------------------------------------
# # 🔹 Decision Boundary in PCA Space
# # -------------------------------------------
# print("[INFO] Plotting decision boundary in PCA space...")
# xx, yy = np.meshgrid(
#     np.linspace(X_pca[:, 0].min(), X_pca[:, 0].max(), 500),
#     np.linspace(X_pca[:, 1].min(), X_pca[:, 1].max(), 500)
# )
# grid_points = np.c_[xx.ravel(), yy.ravel()]

# svm_pca = OneClassSVM(kernel='rbf', nu=0.05, gamma='scale')
# svm_pca.fit(X_pca)

# Z_vals = []
# for i in tqdm(range(0, grid_points.shape[0], 5000), desc="Computing decision surface"):
#     Z_vals.extend(svm_pca.decision_function(grid_points[i:i+5000]))
# Z = np.array(Z_vals).reshape(xx.shape)

# plt.figure(figsize=(8, 6))
# plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.Blues_r)
# plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='red')
# plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_test, cmap='coolwarm', edgecolors='k', s=10)
# plt.xlabel('PCA 1')
# plt.ylabel('PCA 2')
# save_plot(os.path.join(output_dir, "decision_boundary.png"), "Decision Boundary (PCA Space)")

# -------------------------------------------
# 🔹 t-SNE Visualization
# -------------------------------------------
# Function to show progress
def progress_callback(iteration, error, n_iter):
    percent = (iteration / n_iter) * 100
    print(f"\r[INFO] t-SNE progress: {percent:.2f}% completed (error: {error:.4f})", end='')

# ✅ t-SNE with progress
print("[INFO] Running t-SNE with progress tracking...")

tsne = TSNE(
    n_components=2,
    perplexity=30,
    n_iter=1000,              # default
    callbacks=progress_callback,
    callbacks_every_iters=50,  # show progress every 50 iterations
    random_state=42
)

X_tsne = tsne.fit(X_test)

print("\n[INFO] t-SNE completed.")

# Plotting
plt.figure(figsize=(8, 6))
plt.scatter(X_tsne[y_test == 1, 0], X_tsne[y_test == 1, 1], label="Benign", alpha=0.6, c='blue')
plt.scatter(X_tsne[y_test == -1, 0], X_tsne[y_test == -1, 1], label="Anomaly", alpha=0.6, c='orange')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "tsne_plot.png"))
plt.close()
print("[INFO] t-SNE plot saved.")


# Simulated progress bar function
def show_progress_bar(task_name="Working", steps=20, delay=0.05):
    print(f"[INFO] {task_name}...", end='', flush=True)
    for i in range(steps):
        time.sleep(delay)  # simulate work
        print('█', end='', flush=True)
    print(" done.")

# -------------------------------------------
# 🔹 Confusion Matrix with simulated progress
# -------------------------------------------
show_progress_bar("Plotting confusion matrix")

cm = confusion_matrix(y_true, y_pred_bin)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign", "Anomaly"])
disp.plot(cmap=plt.cm.Blues)

save_plot(os.path.join(output_dir, "confusion_matrix.png"), "Confusion Matrix")
print("[INFO] Confusion matrix saved.")

# -------------------------------------------
# 🔹 ROC Curve
# -------------------------------------------
show_progress_bar("Plotting ROC curve")

fpr, tpr, _ = roc_curve(y_true, decision_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid(True)
save_plot(os.path.join(output_dir, "roc_curve.png"), "ROC Curve for One-Class SVM")
print("[INFO] ROC curve saved.")

# -------------------------------------------
# 🔹 PCA Explained Variance
# -------------------------------------------
show_progress_bar("Plotting PCA explained variance")

explained_variance = PCA().fit(X_test).explained_variance_ratio_

plt.figure(figsize=(8, 6))
plt.bar(range(1, len(explained_variance) + 1), explained_variance, color='teal')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
save_plot(os.path.join(output_dir, "pca_variance.png"), "PCA Explained Variance")

print("[✅] All tasks completed successfully.")
