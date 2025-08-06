import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import os
import time
import threading

# -------------------------------------------
# 🔹 Custom VAE Model
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

# -------------------------------------------
# 🔹 Data Loading
# -------------------------------------------
csv_path = os.path.join(os.path.dirname(__file__), "../data/ssl_zero_day_benign.csv")
df = pd.read_csv(csv_path)
print(f"Data loaded. Shape: {df.shape}")

# ✅ Save labels before dropping them
if 'label' in df.columns:
    labels = df['label'].values
    os.makedirs("../models", exist_ok=True)
    np.save('../models/labels.npy', labels)
    print("Labels saved to ../models/labels.npy")
else:
    print("'label' column not found in the dataset.")

# Drop non-numeric and irrelevant columns
non_numeric_cols = ['SourceIP', 'DestinationIP', 'SourcePort', 
                    'DestinationPort', 'TimeStamp', 'label']
df_numeric = df.drop(columns=non_numeric_cols, errors='ignore')

# Drop rows with NaNs
df_numeric = df_numeric.dropna()

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_numeric)

# -------------------------------------------
# 🔹 Model Training
# -------------------------------------------
vae = VAE(input_dim=X_scaled.shape[1])
vae.compile(optimizer='adam')

# Create encoder model
inputs = Input(shape=(X_scaled.shape[1],))
z_mean, _ = vae.encode(inputs)
encoder = Model(inputs, z_mean)

print("\n Training VAE (5 epochs)...")
vae.fit(X_scaled, X_scaled, epochs=5, batch_size=256)

# -------------------------------------------
# 🔹 Encode latent space
# -------------------------------------------
print("\n Encoding latent features...")
X_latent = encoder.predict(X_scaled)

# Ensure no NaNs
if np.isnan(X_latent).any():
    raise ValueError("Latent features contain NaNs. Check VAE training.")

# -------------------------------------------
# 🔹 Train OneClass SVM with Spinner
# -------------------------------------------
# stop_flag = False

# def heartbeat():
#     spinner = ['|', '/', '-', '\\']
#     idx = 0
#     while not stop_flag:
#         print(f"\r[INFO] Training OneClass SVM... {spinner[idx % len(spinner)]}", end="")
#         idx += 1
#         time.sleep(0.5)

# print("\n🤖 Training OneClass SVM...")

# heartbeat_thread = threading.Thread(target=heartbeat)
# heartbeat_thread.start()

# start_time = time.time()
# ocsvm = OneClassSVM(kernel='rbf', gamma='auto', verbose=False)
# ocsvm.fit(X_latent)
# end_time = time.time()

# stop_flag = True
# heartbeat_thread.join()
# print(f"\n✅ OneClass SVM training completed in {end_time - start_time:.2f} seconds.")


# -------------------------------------------
# 🔹 Train Isolation Forest with Spinner
# -------------------------------------------
stop_flag = False

def heartbeat():
    spinner = ['|', '/', '-', '\\']
    idx = 0
    while not stop_flag:
        print(f"\r[INFO] Training Isolation Forest... {spinner[idx % len(spinner)]}", end="")
        idx += 1
        time.sleep(0.5)

print("\n Training Isolation Forest...")

heartbeat_thread = threading.Thread(target=heartbeat)
heartbeat_thread.start()

start_time = time.time()
iso_forest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42, n_jobs=-1)
iso_forest.fit(X_latent)
end_time = time.time()

stop_flag = True
heartbeat_thread.join()
print(f"\n Isolation Forest training completed in {end_time - start_time:.2f} seconds.")

# -------------------------------------------
# 🔹 Save Models
# -------------------------------------------
# os.makedirs("../models", exist_ok=True)
vae.save("../models/vae_model.h5")
encoder.save("../models/vae_encoder.h5")
joblib.dump(scaler, "../models/ssl_scaler.joblib")
# joblib.dump(ocsvm, "../models/ocsvm_model.joblib")
joblib.dump(iso_forest, "../models/iso_forest_model.joblib")
np.save("../models/latent_features.npy", X_latent)

print("\n All models saved successfully!")



# This Python script builds an anomaly detection system using a Variational Autoencoder (VAE) and Isolation Forest to identify malicious patterns in DNS traffic. Here's a breakdown of what the code does:

# 🔹 1. Library Imports
# Essential libraries are imported, including TensorFlow for building deep learning models, scikit-learn for scaling and anomaly detection, and pandas for data manipulation.

# 🔹 2. Custom VAE Class
# A custom VAE (Variational Autoencoder) class is defined with:

# Encoder to compress the input into latent features.

# Decoder to reconstruct the original input from the latent space.

# Reparameterization trick to sample from the latent distribution.

# KL Divergence and Reconstruction Loss for training.

# 🔹 3. Data Loading
# Reads benign DNS traffic data from ssl_zero_day_benign.csv.
# If the label column exists, it's saved separately in a .npy file.

# 🔹 4. Data Preprocessing
# Removes non-numeric columns like IP addresses and ports.

# Drops rows with missing values.

# Scales the numerical data using StandardScaler.

# 🔹 5. VAE Model Training
# A VAE model is compiled and trained for 5 epochs on the scaled data.

# A separate encoder model is built to extract latent features.

# 🔹 6. Latent Feature Extraction
# The encoder is used to generate latent features from the input data.
# If any NaN values appear, the program raises an error.

# 🔹 7. Isolation Forest Training
# The latent features are used to train an Isolation Forest, an unsupervised anomaly detection algorithm.

# A spinner animation is shown in the terminal while training for visual feedback.

# (Note: OneClassSVM section is commented out and not used.)

# 🔹 8. Model Saving
# The following components are saved into the ../models directory:

# vae_model.h5 – full VAE model

# vae_encoder.h5 – encoder model for latent features

# ssl_scaler.joblib – scaler used for standardization

# iso_forest_model.joblib – trained Isolation Forest model

# latent_features.npy – extracted latent features

# ✅ Final Outcome:
# This script creates an end-to-end anomaly detection pipeline:

# Loads and processes benign DNS traffic data,

# Extracts compressed representations via a VAE,

# Detects outliers using Isolation Forest, and

# Saves all components for future use.

# This system is especially useful for detecting zero-day attacks or unusual behavior in DNS traffic.