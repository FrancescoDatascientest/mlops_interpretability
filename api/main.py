import json
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
import joblib
import subprocess

from utils.config_loader import load_config
from training.preprocess import preprocess_for_inference
from training.train import train_and_select_best_model


from .interpretability import (
    init_explainer,
    compute_local_shap,
    generate_summary_plot,
    generate_dependence_plot,
    generate_force_plot
)
from .interpretability import (
    init_lime_explainer,
    generate_lime_explanation
)


app = FastAPI(title="Interpretable ML API")

config = load_config()

# Chemins locaux
ARTIFACTS_DIR = config["artifacts_dir"]
MODEL_LOCAL_PATH = os.path.join(ARTIFACTS_DIR, "rf_pitch_model.joblib")
OHE_LOCAL_PATH = os.path.join(ARTIFACTS_DIR, "ohe_encoder.joblib")
LE_LOCAL_PATH = os.path.join(ARTIFACTS_DIR, "label_encoder.joblib")


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

    if not os.path.exists(LE_LOCAL_PATH):
        subprocess.run(
            f"dagshub download --bucket FrancescoDatascientest/mlops_interpretability artifacts/label_encoder.joblib {LE_LOCAL_PATH}",
            shell=True, check=True
        )

    model = joblib.load(MODEL_LOCAL_PATH)
    ohe_encoder = joblib.load(OHE_LOCAL_PATH)
    le_encoder = joblib.load(LE_LOCAL_PATH)
    print("✅ Model, LE and OHE loaded")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1️⃣ Lire et preprocesser le CSV
    df = pd.read_csv(file.file)
    X_processed = preprocess_for_inference(df, ohe_encoder)
    preds_encoded = model.predict(X_processed)
    label_encoder = joblib.load(LE_LOCAL_PATH)
    preds = label_encoder.inverse_transform(preds_encoded)

    # 2️⃣ Créer un explainer pour ce dataset
    explainer = init_explainer(model)
    shap_values = compute_local_shap(explainer, X_processed)

    label_encoder = joblib.load(LE_LOCAL_PATH)
    class_names = label_encoder.classes_.tolist()
    lime_explainer = init_lime_explainer(X_processed, class_names)


    # 3️⃣ Stocker les données pour les endpoints SHAP
    app.state.df_original = df
    app.state.X_processed = X_processed
    app.state.shap_values = shap_values
    app.state.explainer = explainer
    app.state.lime_explainer = lime_explainer
    app.state.predictions = preds


    preview = X_processed.head(4).copy()
    preview['preds'] = preds[:4] 
    preview_records = preview.to_dict(orient="records")

    return {
        "nb_samples": len(X_processed),
        "preview": preview_records
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
    X = app.state.X_processed
    shap_values = app.state.shap_values
    explainer = app.state.explainer

    output_path = f"force_plot_{index}.html"
    generate_force_plot(explainer, shap_values, X, index=index, output_path=output_path)

    return FileResponse(output_path, media_type="text/html")


@app.get("/interpretability/lime")
async def lime_explanation(index: int = 0, num_features: int = 10):
    if not hasattr(app.state, "X_processed"):
        return {"error": "Aucune donnée chargée. Appelez /predict d'abord."}

    X = app.state.X_processed
    lime_explainer = app.state.lime_explainer

    output_path = f"lime_explanation_{index}.html"

    generate_lime_explanation(
        lime_explainer,
        model,
        X,
        index=index,
        num_features=num_features,
        output_path=output_path
    )

    return FileResponse(output_path, media_type="text/html")


@app.post("/retrain")
async def retrain(background_tasks: BackgroundTasks):
    """
    Endpoint pour ré-entraîner les modèles XGBoost et RandomForest,
    sélectionner le meilleur et le recharger pour les prédictions.
    """

    def training_job():
        global model

        # Entraîner et sélectionner le meilleur modèle
        best_model_name, metrics, train_size = train_and_select_best_model(
            artifacts_dir=ARTIFACTS_DIR
        )

        # Mettre à jour le modèle global pour les prédictions
        model_path = os.path.join(ARTIFACTS_DIR, f"{best_model_name}_pitch_model.joblib")
        model = joblib.load(model_path)
        print(f"✅ Meilleur modèle chargé : {best_model_name} avec metrics {metrics}")

        # Sauvegarder les metrics dans un JSON
        metrics_payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "best_model": best_model_name,
            "metrics": metrics,
            "train_size": train_size
        }

        metrics_path = os.path.join(ARTIFACTS_DIR, "latest_training_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics_payload, f, indent=2)

        print(f"✅ Metrics sauvegardées dans {metrics_path}")

    # Lancer le job en arrière-plan pour ne pas bloquer l'API
    background_tasks.add_task(training_job)

    return {"message": "Ré-entraînement lancé en arrière-plan."}
   



@app.get("/retrain/metrics")
def get_latest_training_metrics():
    metrics_path = os.path.join(ARTIFACTS_DIR, "latest_training_metrics.json")

    if not os.path.exists(metrics_path):
        return {"status": "no training run yet"}

    with open(metrics_path) as f:
        return json.load(f)
    
from fastapi.responses import FileResponse

@app.get("/retrain/confusion-matrix/{model_name}")
def get_confusion_matrix(model_name: str):
    path = os.path.join(ARTIFACTS_DIR, f"{model_name}_confusion_matrix.png")

    if not os.path.exists(path):
        return {"error": "Confusion matrix not found"}

    return FileResponse(path, media_type="image/png")