from fastapi import FastAPI
from fastapi import UploadFile, File
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import os
from training.preprocess import preprocess_for_inference

# Initialiser l'application
app = FastAPI(title="Interpretable ML API")

# Chemins absolus basés sur l'emplacement du fichier main.py
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "artifacts", "xgb_pitch_model.joblib")
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "artifacts", "label_encoder.joblib")
OHE_ENCODER_PATH = os.path.join(BASE_DIR, "artifacts", "ohe_encoder.joblib")

# Charger les artefacts au démarrage
@app.on_event("startup")
def load_artifacts():
    global model, label_encoder, ohe_encoder
    try:
        model = joblib.load(MODEL_PATH)
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
        ohe_encoder = joblib.load(OHE_ENCODER_PATH)
        print("✅ Modèle et encodeurs chargés avec succès")
    except Exception as e:
        print(f"❌ Erreur de chargement des artefacts: {e}")
        model, label_encoder, ohe_encoder = None, None, None

# Route d'accueil simple
@app.get("/")
def home():
    return {"message": "API de prédiction opérationnelle ✅"}

# Fonction de prétraitement
def preprocess_input(df: pd.DataFrame):
    # Exemple : transformation custom
    df['is_home_pitcher'] = df['inning_topbot'].apply(lambda x: 1 if x == 'Top' else 0)

    variables = [
    'description',
    'player_name',
    'is_home_pitcher',
    #'home_team', 'away_team',
    # Lancer / release
    "release_speed",          # Vitesse de la balle au moment du lâcher (mph)
    "release_pos_x",          # Position horizontale du point de release (feet)
    "release_pos_y",          # Position verticale (distance depuis le monticule ou le sol)
    "release_pos_z",          # Hauteur du point de release (feet)
    "release_extension",      # Distance du pied du monticule à la main au moment du release (feet)
    "release_spin_rate",      # Vitesse de rotation de la balle au release (rpm)              
    "spin_axis",              # Idem spin_dir (peut être redondant)
    "p_throws",               # Bras du lanceur (R = droitier, L = gaucher)
    "pitch_name",             # Type de pitch (fastball, slider, curve, etc.)
    "pitch_number",           # Ordre du pitch dans l'AB

    # Trajectoire / physique
    "vx0", "vy0", "vz0",      # Vitesse de la balle sur les axes x, y, z au release (feet/sec)
    "ax", "ay", "az",         # Accélérations sur les axes x, y, z (gravité + spin)
    "pfx_x", "pfx_z",         # Déviation latérale (x) et verticale (z)
    "effective_speed",        # Vitesse perçue par le frappeur
    "sz_top", "sz_bot",       # Bornes supérieures et inférieures de la zone de strike
    "arm_angle",    # Angle du bras du lanceur au release,

    'game_type',
    'stand',

    # Contexte du joueur
    "age_bat",                # Âge du batteur
    "age_bat_legacy",         # Âge lors de la première saison MLB
    "n_priorpa_thisgame_player_at_bat",  # Nombre d'AB précédents pour ce joueur dans le match

    # Champ / défense
    "of_fielding_alignment",  # Alignement des joueurs de champ extérieur
    "if_fielding_alignment",  # Alignement des joueurs de champ intérieur

    # Compte / situation
    "balls", "strikes",       # Compte courant du batteur
    "outs_when_up",           # Nombre de retraits lors de l'AB
    "inning", "inning_topbot",# Manche et top/bottom (haut/bas)
    "home_score", "away_score", # Score courant
    # "home_score_diff",        # Différence score maison - extérieur
    "at_bat_number"           # Numéro d'AB dans le match
]

    df = df[variables].copy()
    df.drop(columns = "description", inplace = True)
    df.dropna(inplace=True)

    cols_to_convert = [
    'pitch_number',
    'age_bat',
    'age_bat_legacy',
    'n_priorpa_thisgame_player_at_bat',
    'balls',
    'strikes',
    'outs_when_up',
    'inning',
    'home_score',
    'away_score',
    'at_bat_number'
]


    for col in df[cols_to_convert] : 
        df[col] = df[col].astype(float)

    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    numeric_columns = [col for col in df.columns if col not in categorical_columns]

    df_cat = pd.DataFrame(ohe_encoder.transform(df[categorical_columns]),
    columns = ohe_encoder.get_feature_names_out(categorical_columns),
    index = df.index)
    
    df_final = pd.concat([df[numeric_columns], df_cat], axis=1)
    return df_final

# Endpoint predict avec CSV
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Modèle non chargé"}

    df = pd.read_csv(file.file)
    X_processed = preprocess_input(df)

    preds = model.predict(X_processed)
    if label_encoder is not None:
        preds = label_encoder.inverse_transform(preds)

    X_processed["prediction"] = preds
    preview = X_processed.head(5).to_dict(orient="records")

    return {
        "nb_samples": len(df),
        "preview": preview
    }