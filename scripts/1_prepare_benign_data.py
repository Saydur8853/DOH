import pandas as pd
import os

BENIGN_PATH = "../data/BenignDoH-NonDoH-CSVs"
OUTPUT_PATH = "../data/ssl_zero_day_benign.csv"

def prepare_benign_only_dataset():
    chrome = pd.read_csv(os.path.join(BENIGN_PATH, "chrome.csv"))
    firefox = pd.read_csv(os.path.join(BENIGN_PATH, "firefox.csv"))
    benign = pd.concat([chrome, firefox], ignore_index=True)
    benign['label'] = 1  # All benign labeled as 1
    benign.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Benign-only dataset saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    prepare_benign_only_dataset()



# This Python script prepares a benign-only dataset for DNS traffic analysis:

# It reads two CSV files: chrome.csv and firefox.csv, both containing benign (non-malicious) DNS traffic data.

# It combines the data from both files into a single dataset using pandas.

# A new column called label is added, where all entries are assigned the value 1 to indicate that they are benign.

# The combined and labeled dataset is saved as a new CSV file named ssl_zero_day_benign.csv.

# A confirmation message is printed once the process is complete.

# This dataset can be used for training machine learning models to differentiate between benign and malicious traffic.