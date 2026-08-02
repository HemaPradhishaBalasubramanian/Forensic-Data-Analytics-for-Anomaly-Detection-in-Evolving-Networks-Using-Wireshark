import os
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------
# Paths
# ---------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(
    BASE_DIR, "../data/raw/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
)

MODEL_PATH = os.path.join(BASE_DIR, "../models/autoencoder_best.h5")
SCALER_PATH = os.path.join(BASE_DIR, "../models/scaler.joblib")

# ---------------------------------------------------
# Load Data
# ---------------------------------------------------
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ Test CSV not found at {DATA_PATH}")

print(f"📂 Loading training data from {DATA_PATH}...")
df = pd.read_csv(DATA_PATH)

# 🔹 IMPORTANT: clean column names (removes hidden spaces)
df.columns = df.columns.str.strip()

LABEL_COLUMN = "Label"
if LABEL_COLUMN not in df.columns:
    raise KeyError(f"❌ Label column '{LABEL_COLUMN}' not found!")

# Separate labels
labels = df[LABEL_COLUMN]
df_features = df.drop(columns=[LABEL_COLUMN])
# add dummy column if only 78 features
if df_features.shape[1] == 78:
    df_features["dummy_feature"] = 0
print("Number of features:", df_features.shape[1])
print(df_features.columns)

# Handle missing/infinite values
df_features.replace([np.inf, -np.inf], np.nan, inplace=True)
df_features.fillna(0, inplace=True)

# ---------------------------------------------------
# Load scaler and model
# ---------------------------------------------------
scaler = joblib.load(SCALER_PATH)
autoencoder = load_model(MODEL_PATH, compile=False)

# Scale features
# make test columns match training columns

scaler.fit(df_features)      # refit scaler on test data
X_scaled = scaler.transform(df_features)

# ---------------------------------------------------
# Reconstruction
# ---------------------------------------------------
X_recon = autoencoder.predict(X_scaled, verbose=1)

# Compute reconstruction error
mse = np.mean(np.power(X_scaled - X_recon, 2), axis=1)

# ---------------------------------------------------
# Threshold (simple)
# ---------------------------------------------------
threshold = np.percentile(mse, 95)
print(f"🔴 Anomaly threshold: {threshold}")

# Predict anomalies
predictions = np.where(mse > threshold, "ATTACK", "BENIGN")

# ---------------------------------------------------
# Results
# ---------------------------------------------------
df_results = pd.DataFrame({
    "TrueLabel": labels,
    "Prediction": predictions,
    "ReconstructionError": mse
})

print(df_results.head(20))

# Save results
OUTPUT_PATH = os.path.join(BASE_DIR, "../data/processed/detection_results.csv")
df_results.to_csv(OUTPUT_PATH, index=False)
print(f"\n✅ Detection results saved at: {OUTPUT_PATH}")