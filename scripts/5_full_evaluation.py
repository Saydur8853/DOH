import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm  # for progress bar
import matplotlib.pyplot as plt
import seaborn as sns

# Create visualization directory
vis_dir = os.path.join(os.path.dirname(__file__), "../result")
os.makedirs(vis_dir, exist_ok=True)

# Load model and scaler
print("📦 Loading model and scaler...")
scaler_path = os.path.join(os.path.dirname(__file__), "../models/ssl_scaler.joblib")
model_path = os.path.join(os.path.dirname(__file__), "../models/ssl_oneclass_model.joblib")
scaler = joblib.load(scaler_path)
model = joblib.load(model_path)

# Non-numeric columns to drop
non_numeric_cols = ['SourceIP', 'DestinationIP', 'SourcePort', 'DestinationPort', 'TimeStamp', 'label']

# 1️⃣ Load benign data
print("📁 Loading benign data...")
benign_path = os.path.join(os.path.dirname(__file__), "../data/ssl_zero_day_benign.csv")
df_benign = pd.read_csv(benign_path)
df_benign.dropna(inplace=True)

# OPTIONAL: For testing speed
# df_benign = df_benign.sample(n=50000, random_state=42)

print(f"🔍 Predicting on benign data ({len(df_benign)} rows)...")
X_benign = df_benign.drop(columns=non_numeric_cols, errors='ignore')
X_benign = scaler.transform(X_benign)
y_benign_true = np.ones(X_benign.shape[0])  # 1 = benign

# Predict in chunks
chunk_size = 10000
y_benign_pred = []
for i in tqdm(range(0, len(X_benign), chunk_size), desc="📊 Benign Prediction Progress"):
    chunk = X_benign[i:i+chunk_size]
    preds = model.predict(chunk)
    y_benign_pred.extend(preds)

y_benign_pred = np.array(y_benign_pred)
y_benign_pred = np.where(y_benign_pred == -1, 0, 1)

# 2️⃣ Load malicious data
print("📁 Loading malicious data...")
malicious_files = ["iodine.csv", "dnscat2.csv", "dns2tcp.csv"]
data_dir = os.path.join(os.path.dirname(__file__), "../data/MaliciousDoH-CSVs")

X_mal_all = []
for file_name in malicious_files:
    print(f"🔄 Processing {file_name}...")
    path = os.path.join(data_dir, file_name)
    df_mal = pd.read_csv(path)
    df_mal.dropna(inplace=True)
    X_mal = df_mal.drop(columns=non_numeric_cols, errors='ignore')
    X_mal = X_mal[scaler.feature_names_in_]  # match column order
    X_mal = scaler.transform(X_mal)
    X_mal_all.append(X_mal)

X_mal_combined = np.vstack(X_mal_all)
y_mal_true = np.zeros(X_mal_combined.shape[0])  # 0 = malicious

print(f"🔍 Predicting on malicious data ({len(X_mal_combined)} rows)...")
y_mal_pred = []
for i in tqdm(range(0, len(X_mal_combined), chunk_size), desc="📊 Malicious Prediction Progress"):
    chunk = X_mal_combined[i:i+chunk_size]
    preds = model.predict(chunk)
    y_mal_pred.extend(preds)

y_mal_pred = np.array(y_mal_pred)
y_mal_pred = np.where(y_mal_pred == -1, 0, 1)

# 3️⃣ Combine all
print("🧩 Combining predictions...")
y_true = np.concatenate([y_benign_true, y_mal_true])
y_pred = np.concatenate([y_benign_pred, y_mal_pred])

# 4️⃣ Print Evaluation
print("\n🔍 Classification Report (1 = Benign, 0 = Malicious):\n")
print(classification_report(y_true, y_pred, digits=4))

# Confusion matrix
print("📊 Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# 5️⃣ 📊 Plot confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Malicious (0)', 'Benign (1)'], yticklabels=['Malicious (0)', 'Benign (1)'])
plt.title("🔍 Confusion Matrix Heatmap")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig(os.path.join(vis_dir, "confusion_matrix.png"))
plt.close()

# 6️⃣ 📈 Bar chart for precision, recall, f1-score
report = classification_report(y_true, y_pred, output_dict=True)
labels = ['Malicious (0)', 'Benign (1)']
metrics = ['precision', 'recall', 'f1-score']
data = [[report['0.0'][m], report['1.0'][m]] for m in metrics]

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots()
bar1 = ax.bar(x - width/2, [d[0] for d in data], width, label='Malicious (0)', color='crimson')
bar2 = ax.bar(x + width/2, [d[1] for d in data], width, label='Benign (1)', color='green')

ax.set_ylabel('Score')
ax.set_title('📊 Classification Metrics by Class')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
plt.ylim(0, 1.1)
plt.tight_layout()
plt.savefig(os.path.join(vis_dir, "classification_metrics_bar.png"))
plt.close()

# 7️⃣ 🥧 Pie chart of prediction distribution
pred_labels, pred_counts = np.unique(y_pred, return_counts=True)
plt.figure(figsize=(5, 5))
plt.pie(pred_counts, labels=[f'Pred: {int(label)}' for label in pred_labels], autopct='%1.1f%%', colors=['red', 'lightgreen'], startangle=90)
plt.title("🧠 Model Predictions Distribution")
plt.savefig(os.path.join(vis_dir, "prediction_distribution_pie.png"))
plt.close()