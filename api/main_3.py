from fastapi import FastAPI, UploadFile, File
import joblib
import os
import pandas as pd
from utils.config_loader import load_config
from training.preprocess import preprocess_for_inference
import subprocess
from fastapi.responses import FileResponse
from .interpretability import (
    init_explainer,
    compute_local_shap,
    compute_global_shap,
    generate_summary_plot,
    generate_dependence_plot,
    generate_force_plot,
)


app = FastAPI(title="Interpretable ML API")

config = load_config()

# Chemins locaux
DATA_DIR = config["data_dir"]
ARTIFACTS_DIR = config["artifacts_dir"]
MODEL_LOCAL_PATH = os.path.join(ARTIFACTS_DIR, "rf_pitch_model.joblib")
OHE_LOCAL_PATH = os.path.join(ARTIFACTS_DIR, "ohe_encoder.joblib")

S3_BUCKET = config["s3_bucket"]

@app.on_event("startup")
def load_artifacts():
    global model, ohe_encoder

    # Télécharger les artefacts depuis S3 vers le local si ils n'existent pas
    if not os.path.exists(ARTIFACTS_DIR):
        os.makedirs(ARTIFACTS_DIR)

    if not os.path.exists(MODEL_LOCAL_PATH):
        subprocess.run(f"dagshub download --bucket FrancescoDatascientest/mlops_interpretability artifacts/rf_pitch_model.joblib {MODEL_LOCAL_PATH}", shell=True, check=True)
    if not os.path.exists(OHE_LOCAL_PATH):
        subprocess.run(f"dagshub download --bucket FrancescoDatascientest/mlops_interpretability artifacts/ohe_encoder.joblib {OHE_LOCAL_PATH}", shell=True, check=True)

    # Charger les artefacts
    model = joblib.load(MODEL_LOCAL_PATH)
    ohe_encoder = joblib.load(OHE_LOCAL_PATH)
    init_explainer(model)
    print("✅ Model and OHE loaded from S3")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    X_processed = preprocess_for_inference(df, ohe_encoder)
    preds = model.predict(X_processed)
    df_result = X_processed.copy()
    df_result["prediction"] = preds
    preview = df_result.head(5).to_dict(orient="records")
    
    return {
        "nb_samples": len(df_result),
        "preview": preview
    }

@app.post("/interpretability/plot/summary")
async def summary_plot(file: UploadFile = File(...)):
    # 1) Lire et préprocesser le fichier envoyé
    df = pd.read_csv(file.file)
    X_processed = preprocess_for_inference(df, ohe_encoder)

    # 2) SHAP local sur X
    shap_vals = compute_local_shap(X_processed)["shap_values"]

    # 3) Générer l’image
    output_path = "summary_plot.png"
    generate_summary_plot(X_processed, shap_vals, output_path)

    # 4) Renvoi de l’image
    return FileResponse(output_path, media_type="image/png")


@app.post("/interpretability/plot/dependence")
async def dependence_plot(feature: str, file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    X_processed = preprocess_for_inference(df, ohe_encoder)

    shap_vals = compute_local_shap(X_processed)["shap_values"]

    output_path = f"dependence_{feature}.png"
    generate_dependence_plot(X_processed, shap_vals, feature, output_path)

    return FileResponse(output_path, media_type="image/png")

@app.post("/interpretability/plot/force")
async def force_plot(index: int = 0, file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    X_processed = preprocess_for_inference(df, ohe_encoder)

    shap_vals_dict = compute_local_shap(X_processed)
    shap_vals = shap_vals_dict["shap_values"]

    output_path = f"force_plot_{index}.html"
    generate_force_plot(explainer, shap_vals, X_processed, index, output_path)

    return FileResponse(output_path, media_type="text/html")
