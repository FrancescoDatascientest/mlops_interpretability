# training/train.py
import pandas as pd
import glob
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from .preprocess import preprocess_train
from .evaluation import save_confusion_matrix
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


def train_and_select_best_model(artifacts_dir: str):

    df_raw = load_raw_data()
    X_processed, ohe_encoder, label_encoder, y = preprocess_train(df_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42
    )

    models = {
        "rf": RandomForestClassifier(n_estimators=100, random_state=42),
        "xgb": XGBClassifier(n_estimators=100, eval_metric="logloss", random_state=42)
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        results[name] = {
            "accuracy": accuracy_score(y_test, preds),
            "confusion_matrix": confusion_matrix(y_test, preds).tolist()
        }
 
        labels_enc = list(range(len(label_encoder.classes_)))

        save_confusion_matrix(
        y_true=y_test,
        y_pred=preds,
        labels=labels_enc,
        model_name=name,
        artifacts_dir=artifacts_dir,
        display_labels=label_encoder.classes_)

        joblib.dump(model, os.path.join(artifacts_dir, f"{name}_pitch_model.joblib"))

    joblib.dump(ohe_encoder, os.path.join(artifacts_dir, "ohe_encoder.joblib"))
    joblib.dump(label_encoder, os.path.join(artifacts_dir, "label_encoder.joblib"))

    best_model_name = max(results, key=lambda m: results[m]["accuracy"])

    print(f"✅ L'entraînement se réalise sur {X_train.shape[0]} pitches et le modèle est testé sur {X_test.shape[0]} pitches")
    print(f"✅ Modèles entraînés et sauvegardés. Meilleur modèle : {best_model_name} avec accuracy {results[best_model_name]}")

    return best_model_name, results, len(X_train)