# train.py – trains the Autoencoder model
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import ModelCheckpoint
from model.autoencoder_model import build_autoencoder
import joblib

# -------------------------------------
# Load preprocessed data
# -------------------------------------
data_path = os.path.join("data", "processed", "dataset.csv")
print(f"[*] Loading data from {data_path} ...")

# Try to read the CSV safely
try:
    df = pd.read_csv(data_path)
except Exception as e:
    print(f"[!] Default CSV load failed: {e}")
    print("[*] Trying with semicolon (;) delimiter ...")
    df = pd.read_csv(data_path, sep=';')

print(f"[+] Dataset loaded successfully with shape: {df.shape}")

# -------------------------------------
# Clean the dataset
# -------------------------------------
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()
df = df.reset_index(drop=True)

# Drop label column if it exists — only train on benign data
if 'Label' in df.columns:
    df = df[df['Label'] == 'BENIGN']
    df = df.drop(columns=['Label'], errors='ignore')

# -------------------------------------
# Scale the data
# -------------------------------------
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df.values)

# Save the scaler for testing later
joblib.dump(scaler, "models/scaler.joblib")
print("[+] Scaler saved at models/scaler.joblib")

# -------------------------------------
# Split dataset
# -------------------------------------
X_train, X_val = train_test_split(X_scaled, test_size=0.2, random_state=42)
print(f"[*] Training samples: {X_train.shape}, Validation samples: {X_val.shape}")

# -------------------------------------
# Build and compile model
# -------------------------------------
input_dim = X_train.shape[1]
model = build_autoencoder(input_dim)

# Ensure model folder exists
os.makedirs("models", exist_ok=True)

checkpoint = ModelCheckpoint(
    "models/autoencoder_best.h5",
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

# -------------------------------------
# Train the model
# -------------------------------------
print("[*] Training Autoencoder...")
history = model.fit(
    X_train, X_train,
    epochs=50,
    batch_size=256,
    shuffle=True,
    validation_data=(X_val, X_val),
    callbacks=[checkpoint]
)

print("[+] Training completed! Model saved at models/autoencoder_best.h5")
