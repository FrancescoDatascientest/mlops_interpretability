# training/train.py
import pandas as pd
import glob
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from preprocess import preprocess_train
from utils.config_loader import load_config

# DATA_DIR = "data/"
# ARTIFACTS_DIR = "artifacts/"
# MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.joblib")
# OHE_PATH = os.path.join(ARTIFACTS_DIR, "ohe_encoder.joblib")

config = load_config()

DATA_DIR = config["data_dir"]
ARTIFACTS_DIR = config["artifacts_dir"]
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "rf_pitch_model.joblib")
OHE_PATH = os.path.join(ARTIFACTS_DIR, "ohe_encoder.joblib")
LE_PATH = os.path.join(ARTIFACTS_DIR, "label_encoder.joblib")

def load_raw_data():
    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    dfs = []
    for f in files:
        df_csv = pd.read_csv(f)
        if not df_csv.empty:
            dfs.append(df_csv)
    if len(dfs) == 0:
        raise ValueError("Aucun CSV non vide trouvé dans data/")
    df = pd.concat(dfs, ignore_index=True)
    return df

def train():
    df = load_raw_data()
    # Preprocessing + fit OHE
    X_processed, ohe, le, y = preprocess_train(df)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

    # Entraîner le modèle
    model = RandomForestClassifier(n_estimators=200)
    model.fit(X_train, y_train)

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(ohe, OHE_PATH)
    joblib.dump(le, LE_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")
    print(f"✅ OHE saved to {OHE_PATH}")
    print(f"✅ LE saved to {LE_PATH}")

if __name__ == "__main__":
    train()
