# training/train.py
import pandas as pd
import glob
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from preprocess import preprocess_for_training
from utils.config_loader import load_config

# DATA_DIR = "data/"
# ARTIFACTS_DIR = "artifacts/"
# MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.joblib")
# OHE_PATH = os.path.join(ARTIFACTS_DIR, "ohe_encoder.joblib")

config = load_config()

DATA_DIR = config["data_dir"]
ARTIFACTS_DIR = config["artifacts_dir"]
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.joblib")
OHE_PATH = os.path.join(ARTIFACTS_DIR, "ohe_encoder.joblib")

def load_raw_data():
    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    return df

def train():
    df = load_raw_data()
    y = df['pitch_name']
    X = df.drop(columns=['pitch_name'])

    # Preprocessing + fit OHE
    X_processed, ohe = preprocess_for_training(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

    # Entraîner le modèle
    model = RandomForestClassifier(n_estimators=200)
    model.fit(X_train, y_train)

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(ohe, OHE_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")
    print(f"✅ OHE saved to {OHE_PATH}")

if __name__ == "__main__":
    train()
