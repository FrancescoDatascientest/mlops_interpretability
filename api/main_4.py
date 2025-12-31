from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import pandas as pd
import os
import joblib
import subprocess

from utils.config_loader import load_config
from training.preprocess import preprocess_for_inference
from .interpretability import (
    init_explainer,
    compute_local_shap,
    generate_summary_plot,
    generate_dependence_plot,
    generate_force_plot
)

app = FastAPI(title="Interpretable ML API")

config = load_config()

# Chemins locaux
ARTIFACTS_DIR = config["artifacts_dir"]
MODEL_LOCAL_PATH = os.path.join(ARTIFACTS_DIR, "rf_pitch_model.joblib")
OHE_LOCAL_PATH = os.path.join(ARTIFACTS_DIR, "ohe_encoder.joblib")

@app.on_event("startup")
def load_artifacts():
    global model, ohe_encoder

    if not os.path.exists(ARTIFACTS_DIR):
        os.makedirs(ARTIFACTS_DIR)

    if not os.path.exists(MODEL_LOCAL_PATH):
        subprocess.run(
            f"dagshub download --bucket FrancescoDatascientest/mlops_interpretability artifacts/rf_pitch_model.joblib {MODEL_LOCAL_PATH}",
            shell=True, check=True
        )
    if not os.path.exists(OHE_LOCAL_PATH):
        subprocess.run(
            f"dagshub download --bucket FrancescoDatascientest/mlops_interpretability artifacts/ohe_encoder.joblib {OHE_LOCAL_PATH}",
            shell=True, check=True
        )

    model = joblib.load(MODEL_LOCAL_PATH)
    ohe_encoder = joblib.load(OHE_LOCAL_PATH)
    print("✅ Model and OHE loaded")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1️⃣ Lire et preprocesser le CSV
    df = pd.read_csv(file.file)
    X_processed = preprocess_for_inference(df, ohe_encoder)
    preds = model.predict(X_processed)

    # 2️⃣ Créer un explainer pour ce dataset
    explainer = init_explainer(model)
    shap_values = compute_local_shap(explainer, X_processed)

    # 3️⃣ Stocker les données pour les endpoints SHAP
    app.state.df_original = df
    app.state.X_processed = X_processed
    app.state.shap_values = shap_values
    app.state.explainer = explainer
    app.state.predictions = preds

    preview = X_processed.head(5).to_dict(orient="records")
    return {
        "nb_samples": len(X_processed),
        "preview": preview
    }


@app.get("/interpretability/plot/summary")
async def summary_plot():
    if not hasattr(app.state, "X_processed"):
        return {"error": "Aucune donnée chargée. Appelez /predict d'abord."}

    X = app.state.X_processed
    shap_values = app.state.shap_values

    output_path = "summary_plot.png"
    generate_summary_plot(X, shap_values, output_path)
    return FileResponse(output_path, media_type="image/png")


@app.get("/interpretability/plot/dependence")
async def dependence_plot(feature: str):
    if not hasattr(app.state, "X_processed"):
        return {"error": "Aucune donnée chargée. Appelez /predict d'abord."}

    X = app.state.X_processed
    shap_values = app.state.shap_values

    output_path = f"dependence_{feature}.png"
    generate_dependence_plot(X, shap_values, feature, output_path)
    return FileResponse(output_path, media_type="image/png")


@app.get("/interpretability/plot/force")
async def force_plot(index: int = 0):
    if not hasattr(app.state, "X_processed"):
        return {"error": "Aucune donnée chargée. Appelez /predict d'abord."}

    X = app.state.X_processed
    shap_values = app.state.shap_values
    explainer = app.state.explainer

    output_path = f"force_plot_{index}.html"
    generate_force_plot(explainer, shap_values, X, index, output_path)
    return FileResponse(output_path, media_type="text/html")