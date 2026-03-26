import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# -------------------------------
# Paths
# -------------------------------
RAW_DATA_PATH = "../data/raw"
PROCESSED_DATA_PATH = "../data/processed/dataset.csv"

# -------------------------------
# Load all CSVs
# -------------------------------
def load_raw_data(raw_path):
    print("[*] Loading CSV files from raw folder...")
    all_dataframes = []

    for file in os.listdir(raw_path):
        if file.endswith(".csv"):
            file_path = os.path.join(raw_path, file)
            print(f"    [+] Loading: {file_path}")
            df = pd.read_csv(file_path, low_memory=False)
            all_dataframes.append(df)

    combined = pd.concat(all_dataframes, ignore_index=True)
    print(f"[*] Loaded data shape: {combined.shape}")
    return combined


# -------------------------------
# Preprocessing (Optimized)
# -------------------------------
def preprocess(df):
    print("[*] Preprocessing data...")

    # Drop duplicates and missing values
    df = df.drop_duplicates()
    df = df.dropna()

    # Encode all categorical (string/object) columns using LabelEncoder
    label_encoders = {}
    for col in df.select_dtypes(include=["object"]).columns:
        print(f"    [>] Encoding: {col}")
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    print(f"[*] Preprocessed data shape: {df.shape}")
    return df


# -------------------------------
# Save processed file
# -------------------------------
def save_processed_data(df, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"[+] Saved processed dataset to: {save_path}")


# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    data = load_raw_data(RAW_DATA_PATH)
    processed = preprocess(data)
    save_processed_data(processed, PROCESSED_DATA_PATH)
    print("[✔] Data preparation complete.")
