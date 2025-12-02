from fastapi import FastAPI, UploadFile, File
import joblib
import pandas as pd
import os
from training.preprocess import preprocess_for_inference

# Init API
app = FastAPI(title="Interpretable ML API")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "artifacts", "rf_pitch_model.joblib")
OHE_PATH = os.path.join(BASE_DIR, "artifacts", "ohe_encoder.joblib")

@app.on_event("startup")
def load_artifacts():
    global model, ohe_encoder
    model = joblib.load(MODEL_PATH)
    ohe_encoder = joblib.load(OHE_PATH)
    print("✅ Model and OHE loaded")

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
    "preview": preview}