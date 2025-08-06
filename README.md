# DNS over HTTPS (DoH) Malware Detection System

## 🎯 Project Overview

This project implements an advanced anomaly detection system for identifying malicious DNS over HTTPS (DoH) traffic using a hybrid approach combining **Variational Autoencoders (VAE)** and **Isolation Forest** algorithms. The system is designed to detect zero-day attacks and malicious DNS tunneling activities by learning patterns from benign traffic data.

## 🏗️ System Architecture

The system employs a sophisticated multi-stage pipeline:

1. **Data Preprocessing**: Normalization and feature engineering of DNS traffic data
2. **Variational Autoencoder**: Learns compressed latent representations of benign traffic patterns
3. **Isolation Forest**: Performs anomaly detection in the learned latent space
4. **Dual Detection Methods**: Both reconstruction error and isolation-based anomaly detection
5. **Advanced Visualization**: PCA-based latent space analysis and performance metrics

## 📊 Technical Specifications

### Core Technologies
- **Deep Learning Framework**: TensorFlow/Keras 3.10.0
- **Machine Learning**: Scikit-learn 1.6.1
- **Data Processing**: Pandas 2.2.3, NumPy 2.1.3
- **Visualization**: Matplotlib 3.10.3, Seaborn 0.13.2
- **Model Persistence**: Joblib, HDF5

### VAE Architecture
- **Input Layer**: Variable dimensions based on feature count
- **Encoder**: 64-unit hidden layer with ReLU activation
- **Latent Space**: 8-dimensional compressed representation
- **Decoder**: 64-unit hidden layer with sigmoid output
- **Loss Function**: Combined KL divergence + MSE reconstruction loss

### Detection Algorithms
- **Isolation Forest**: 100 estimators, 1% contamination rate
- **VAE Reconstruction**: 95th percentile threshold on benign data
- **Feature Scaling**: StandardScaler normalization

## 📁 Project Structure

```
DOH/
├── scripts/                     # Main execution scripts (5 core modules)
├── data/                        # Dataset storage
│   ├── BenignDoH-NonDoH-CSVs/   # Chrome & Firefox benign traffic
│   ├── MaliciousDoH-CSVs/       # Malicious traffic samples
│   └── *_predictions.csv        # Generated prediction results
├── models/                      # Saved ML models and components
├── result/                      # Evaluation reports and visualizations
└── requirements.txt             # Python dependencies
```

## 🚀 Core Scripts & Functionality

### 1. `1_prepare_benign_data.py`
**Purpose**: Data preparation and consolidation
- Merges Chrome and Firefox benign DNS traffic datasets
- Creates unified labeled dataset (`ssl_zero_day_benign.csv`)
- Assigns benign labels (value: 1) to all training samples
- **Output**: Consolidated benign training dataset

### 2. `2_train_vae_oneclass_model.py`
**Purpose**: Advanced model training pipeline
- **VAE Training**: 5-epoch unsupervised learning on benign traffic
- **Feature Extraction**: Generates 8-dimensional latent representations
- **Isolation Forest**: Trains anomaly detector on latent features
- **Data Processing**: StandardScaler normalization, NaN handling
- **Model Persistence**: Saves VAE, encoder, scaler, and Isolation Forest
- **Output**: Complete trained model ecosystem

### 3. `3_test_on_malicious_data_vae_latent.py`
**Purpose**: Malware detection and evaluation
- **Batch Processing**: Evaluates 3 malicious datasets (iodine, dnscat2, dns2tcp)
- **Feature Pipeline**: VAE latent encoding → Isolation Forest prediction
- **Detection Analysis**: Calculates detection rates and anomaly statistics
- **Result Generation**: Creates labeled prediction files with confidence scores
- **Output**: Malware detection results with performance metrics

### 4. `4_vae_latent_space_visualization_with_PCA.py`
**Purpose**: Latent space analysis and visualization
- **PCA Reduction**: Compresses latent features to 2D for visualization
- **Data Alignment**: Synchronizes labels with feature dimensions
- **Cluster Analysis**: Visual representation of learned data distributions
- **Quality Assessment**: Evaluates VAE's ability to separate traffic types
- **Output**: `vae_latent_pca_plot.png` visualization

### 5. `5_full_evaluation_with_vae.py`
**Purpose**: Comprehensive system evaluation and benchmarking
- **Dual Detection**: Both Isolation Forest and VAE reconstruction methods
- **Performance Metrics**: Classification reports, confusion matrices, ROC curves
- **Advanced Visualization**: Latent space plots, ROC comparisons
- **Threshold Optimization**: 95th percentile adaptive thresholding
- **Output**: Complete evaluation report with multiple visualizations

## 🎯 System Performance & Results

### Detection Capabilities
- **Malware Types**: DNS tunneling tools (iodine, dnscat2, dns2tcp)
- **Detection Methods**: Dual-approach for enhanced accuracy
- **Real-time Processing**: Batch processing with progress tracking
- **Scalability**: Chunked processing for large datasets

### Output Artifacts
- **Model Files**: Trained VAE, encoder, Isolation Forest, and scaler
- **Prediction Results**: Labeled CSV files with anomaly classifications
- **Visualizations**: ROC curves, confusion matrices, latent space plots
- **Performance Reports**: Detailed classification metrics and statistics

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Installation Steps

1. **Clone and Navigate**
   ```bash
   cd DOH
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux  
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Execution Workflow

Run scripts sequentially for complete pipeline:

```bash
# Step 1: Prepare training data
python scripts/1_prepare_benign_data.py

# Step 2: Train VAE and anomaly detection models
python scripts/2_train_vae_oneclass_model.py

# Step 3: Test on malicious samples
python scripts/3_test_on_malicious_data_vae_latent.py

# Step 4: Generate latent space visualization
python scripts/4_vae_latent_space_visualization_with_PCA.py

# Step 5: Comprehensive evaluation and reporting
python scripts/5_full_evaluation_with_vae.py
```

## 🔬 Technical Innovation

### Key Features
- **Hybrid Architecture**: Combines generative modeling with isolation-based detection
- **Latent Space Learning**: VAE extracts meaningful traffic patterns
- **Zero-day Detection**: Learns from benign data to detect unknown malware
- **Dual Validation**: Multiple detection algorithms for robust results
- **Comprehensive Evaluation**: Advanced metrics and visualization tools

### Research Applications
- **Cybersecurity**: DNS tunneling malware detection
- **Network Security**: Anomaly detection in encrypted traffic
- **Machine Learning**: Unsupervised anomaly detection techniques
- **Deep Learning**: Variational autoencoder applications in security

## 📈 Expected Outcomes

The system generates comprehensive outputs including:
- **Detection Results**: High-accuracy identification of DNS-based malware
- **Performance Metrics**: ROC curves, precision/recall, F1-scores
- **Visual Analysis**: Latent space representations and cluster analysis
- **Research Data**: Detailed logs and metrics for further analysis

---

**Note**: This system is designed for research and educational purposes in cybersecurity and machine learning. Ensure compliance with applicable laws and regulations when analyzing network traffic data.
