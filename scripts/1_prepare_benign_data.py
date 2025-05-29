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
