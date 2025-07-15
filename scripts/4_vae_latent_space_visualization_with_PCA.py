import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tensorflow.keras.models import load_model

# === CONFIGURATION ===
LATENT_FEATURES_PATH = '../models/latent_features.npy'
LABELS_PATH = '../models/labels.npy'
ENCODER_MODEL_PATH = '../models/vae_encoder.h5'

# === ENSURE OUTPUT FOLDER EXISTS ===
os.makedirs('output', exist_ok=True)
print("[INFO] Output directory ensured.")

# === LOAD LABELS ===
labels = None
if os.path.exists(LABELS_PATH):
    labels = np.load(LABELS_PATH)
    print("[INFO] Labels loaded.")
else:
    print("[WARNING] Label file not found. Proceeding without labels.")

# === LOAD ENCODER MODEL ===
if not os.path.exists(ENCODER_MODEL_PATH):
    raise FileNotFoundError(f"Encoder model not found at {ENCODER_MODEL_PATH}")
encoder = load_model(ENCODER_MODEL_PATH, compile=False)
print("[INFO] Encoder model loaded.")

# === LOAD LATENT FEATURES ===
if not os.path.exists(LATENT_FEATURES_PATH):
    raise FileNotFoundError(f"Latent features not found at {LATENT_FEATURES_PATH}")
X_latent = np.load(LATENT_FEATURES_PATH)
print(f"[INFO] Latent features loaded. Shape: {X_latent.shape}")

# === APPLY PCA ===
print("[INFO] Reducing dimensions using PCA...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_latent)

# === ALIGN LABELS TO LATENT SPACE ===
if labels is not None:
    if len(labels) != len(X_pca):
        print(f"[WARNING] Label size ({len(labels)}) and PCA data size ({len(X_pca)}) do not match. Truncating to minimum size.")
        min_len = min(len(labels), len(X_pca))
        labels = labels[:min_len]
        X_pca = X_pca[:min_len]

# === PLOT ===
print("[INFO] Plotting latent space...")
plt.figure(figsize=(10, 8))
if labels is not None:
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='coolwarm', s=10, alpha=0.7)
    plt.colorbar(label='Label')
else:
    plt.scatter(X_pca[:, 0], X_pca[:, 1], s=10, alpha=0.7)

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("VAE Latent Space (PCA Reduced)")
plt.grid(True)
plt.tight_layout()
plt.savefig("output/vae_latent_pca_plot.png")

