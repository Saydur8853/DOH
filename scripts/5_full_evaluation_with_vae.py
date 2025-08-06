import pandas as pd
import numpy as np
import joblib
import os
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------
# 🔹 Custom VAE Model Definition (must match training)
# -------------------------------------------
class VAE(Model):
    def __init__(self, input_dim, latent_dim=8, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        self.encoder_h = Dense(64, activation='relu')
        self.z_mean = Dense(latent_dim)
        self.z_log_var = Dense(latent_dim)

        # Decoder
        self.decoder_h = Dense(64, activation='relu')
        self.decoder_out = Dense(input_dim, activation='sigmoid')

    def get_config(self):
        config = super(VAE, self).get_config()
        config.update({
            'input_dim': self.input_dim,
            'latent_dim': self.latent_dim
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def encode(self, x):
        h = self.encoder_h(x)
        return self.z_mean(h), self.z_log_var(h)

    def reparameterize(self, z_mean, z_log_var):
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def decode(self, z):
        h = self.decoder_h(z)
        return self.decoder_out(h)

    def call(self, inputs):
        z_mean, z_log_var = self.encode(inputs)
        z = self.reparameterize(z_mean, z_log_var)
        reconstructed = self.decode(z)

        # KL Divergence
        kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
        kl_loss = tf.reduce_mean(kl_loss)

        # Reconstruction Loss
        reconstruction_loss = tf.reduce_mean(tf.keras.losses.mse(inputs, reconstructed)) * self.input_dim

        self.add_loss(kl_loss)
        self.add_loss(reconstruction_loss)

        return reconstructed

# -------------------------------------------
# 🔹 Initialize and Load Models
# -------------------------------------------
print("Initializing evaluation pipeline...")

# Create output directory
vis_dir = os.path.join(os.path.dirname(__file__), "../result")
os.makedirs(vis_dir, exist_ok=True)

# Load all required models
model_dir = os.path.join(os.path.dirname(__file__), "../models")
scaler = joblib.load(os.path.join(model_dir, "ssl_scaler.joblib"))
iso_forest = joblib.load(os.path.join(model_dir, "iso_forest_model.joblib"))
encoder = tf.keras.models.load_model(os.path.join(model_dir, "vae_encoder.h5"))

# Load VAE by instantiating, building, and loading weights
input_dim = len(scaler.feature_names_in_)
vae = VAE(input_dim=input_dim)
# Build the model by calling it on a dummy input to create the weights
vae(tf.zeros((1, input_dim)))
vae.load_weights(os.path.join(model_dir, "vae_model.h5"))
vae.compile(optimizer='adam')

# -------------------------------------------
# 🔹 Data Preparation (Benign + Malicious)
# -------------------------------------------
print("\nPreparing datasets...")

non_numeric_cols = ['SourceIP', 'DestinationIP', 'SourcePort', 
                   'DestinationPort', 'TimeStamp', 'label']

# Load benign data
benign_path = os.path.join(os.path.dirname(__file__), "../data/ssl_zero_day_benign.csv")
df_benign = pd.read_csv(benign_path).dropna()
X_benign = df_benign.drop(columns=non_numeric_cols, errors='ignore')
X_benign_scaled = scaler.transform(X_benign)
y_benign_true = np.ones(len(X_benign))  # 1 = benign

# Load malicious data
malicious_files = ["iodine.csv", "dnscat2.csv", "dns2tcp.csv"]
malicious_dir = os.path.join(os.path.dirname(__file__), "../data/MaliciousDoH-CSVs")

X_mal_list = []
for file in tqdm(malicious_files, desc="Loading malicious data"):
    df = pd.read_csv(os.path.join(malicious_dir, file)).dropna()
    X_mal = df.drop(columns=non_numeric_cols, errors='ignore')
    X_mal = X_mal[scaler.feature_names_in_]  # Ensure feature alignment
    X_mal_list.append(scaler.transform(X_mal))

X_mal_scaled = np.vstack(X_mal_list)
y_mal_true = np.zeros(len(X_mal_scaled))  # 0 = malicious

# Combine datasets
X_full = np.vstack([X_benign_scaled, X_mal_scaled])
y_true = np.concatenate([y_benign_true, y_mal_true])

# -------------------------------------------
# 🔹 Feature Extraction (VAE Latent Space)
# -------------------------------------------
print("\nExtracting latent features...")
chunk_size = 10000
X_latent = []

for i in tqdm(range(0, len(X_full), chunk_size), desc="VAE Encoding"):
    chunk = X_full[i:i+chunk_size]
    X_latent.append(encoder.predict(chunk, verbose=0))

X_latent = np.vstack(X_latent)

# -------------------------------------------
# 🔹 Anomaly Detection (Two Methods)
# -------------------------------------------
print("\nRunning detection algorithms...")

# Method 1: Isolation Forest on latent features
iso_pred = iso_forest.predict(X_latent)
y_pred_iso = np.where(iso_pred == -1, 0, 1)  # Convert to 0/1 (malicious/benign)

# Method 2: VAE Reconstruction Error
recon_errors = []
for i in tqdm(range(0, len(X_full), chunk_size), desc="VAE Reconstruction"):
    chunk = X_full[i:i+chunk_size]
    recon = vae.predict(chunk, verbose=0)
    recon_errors.append(np.mean(np.square(chunk - recon), axis=1))

recon_errors = np.concatenate(recon_errors)
threshold = np.percentile(recon_errors[:len(X_benign)], 95)  # Use benign 95th %ile
y_pred_vae = np.where(recon_errors > threshold, 0, 1)

# -------------------------------------------
# 🔹 Evaluation Metrics
# -------------------------------------------
def evaluate(y_true, y_pred, method_name):
    print(f"{method_name} Performance:")
    print(classification_report(y_true, y_pred, digits=4))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=['Malicious', 'Benign'], 
               yticklabels=['Malicious', 'Benign'])
    plt.title(f"{method_name} Confusion Matrix")
    plt.savefig(os.path.join(vis_dir, f"confusion_matrix_{method_name.lower().replace(' ', '_')}.png"))
    plt.close()

# Evaluate both methods
evaluate(y_true, y_pred_iso, "Isolation Forest")
evaluate(y_true, y_pred_vae, "VAE Reconstruction")

# -------------------------------------------
# 🔹 Advanced Visualizations
# -------------------------------------------
print("\nGenerating visualizations...")

# ROC Curve Comparison
plt.figure(figsize=(8,6))
for name, scores in [('Isolation Forest', -iso_forest.decision_function(X_latent)),
                    ('VAE', recon_errors)]:
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc="lower right")
plt.savefig(os.path.join(vis_dir, "roc_comparison.png"))
plt.close()

# Latent Space Visualization (First 2 dimensions)
plt.figure(figsize=(8,6))
plt.scatter(X_latent[y_true==1, 0], X_latent[y_true==1, 1], 
            alpha=0.3, label='Benign', c='green')
plt.scatter(X_latent[y_true==0, 0], X_latent[y_true==0, 1], 
            alpha=0.3, label='Malicious', c='red')
plt.title('VAE Latent Space (First 2 Dimensions)')
plt.xlabel('Latent Dimension 1')
plt.ylabel('Latent Dimension 2')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(vis_dir, "latent_space.png"))
plt.close()

print("\nEvaluation complete! Results saved to /result directory")


# This Python script is designed to visualize the latent space learned by a trained Variational Autoencoder (VAE) model. The goal is to understand how the model has internally represented the input data in a compressed form (latent features).

# 🔹 1. Configuration and Setup
# The script begins by defining paths to required files:

# latent_features.npy: A file containing the compressed (latent) features generated by the encoder part of the VAE.

# labels.npy: Optional labels for the data points (e.g., malicious or benign).

# vae_encoder.h5: The saved VAE encoder model used to generate latent features.

# It also ensures that an output folder named output exists, where the final visualization will be saved.

# 🔹 2. Loading Data
# If the labels.npy file is present, it loads the labels. Otherwise, it warns that it will proceed without labels.

# It then loads the trained encoder model and the latent features from disk.

# 🔹 3. Dimensionality Reduction using PCA
# Since latent features may have high dimensionality (e.g., 16 or 32), the script applies PCA (Principal Component Analysis) to reduce these to 2 dimensions. This makes it possible to visualize the data as a 2D scatter plot.

# If labels are available, it ensures the number of labels matches the number of data points. If not, it trims both to the minimum length to align properly.

# 🔹 4. Visualization
# The script then plots the 2D PCA-transformed features:

# If labels are present, the plot uses color to distinguish different categories.

# If labels are not present, all points are plotted in the same color.

# Axes are labeled as "PCA Component 1" and "PCA Component 2", and a title is added for clarity. Finally, the plot is saved as vae_latent_pca_plot.png inside the output folder.

# ✅ Purpose of the Script
# This tool helps in analyzing how well the VAE model has captured the data’s structure in its latent space. Clusters or separations in the plot can indicate the model's effectiveness in encoding meaningful patterns—such as differentiating between benign and malicious data in cybersecurity applications.