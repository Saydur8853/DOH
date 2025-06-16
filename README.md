# One-Class Classification for Malware Detection
This project implements a One-Class Classification system to detect malicious behavior based on benign training data. It includes data preparation, model training, evaluation on malicious samples, and visualization using PCA.

📁 Project Structure

/scripts <br>
1_prepare_benign_data.py <br>
2_train_oneclass_model.py <br>
3_test_on_malicious_data.py <br>
4_visualize_pca.py <br> 

🚀 Getting Started
Follow the steps below to set up and run the project.

### 1. Create and Activate a Virtual Environment

`python -m venv venv` <br>
`source venv/Scripts/activate` <br>

### 2. Install Dependencies
If requirements.txt is already available:


`pip install -r requirements.txt` <br>
If not, after installing required packages:

`pip freeze > requirements.txt` <br>
### 3. Run Scripts Sequentially
Execute the following Python scripts in order:


`python scripts/1_prepare_benign_data.py` <br>
`python scripts/2_train_oneclass_model.py` <br>
`python scripts/3_test_on_malicious_data.py` <br>
`python scripts/4_visualize_pca.py` <br>

Each script performs the following:

#### Prepare Benign Data – Preprocess benign samples and save features.

#### Train Model – Train a one-class classifier (e.g., OneClassSVM).

#### Test on Malicious Data – Evaluate the trained model on malicious samples.

#### Visualize with PCA – Visualize benign vs. malicious data using PCA.
