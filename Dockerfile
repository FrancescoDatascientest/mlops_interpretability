# Utiliser une image Python légère
FROM python:3.11-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier uniquement les fichiers nécessaires
COPY api/requirements.txt ./api/requirements.txt
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r api/requirements.txt

COPY api/ ./api/
COPY artifacts/xgb_pitch_model.joblib ./artifacts/xgb_pitch_model.joblib
COPY artifacts/label_encoder.joblib ./artifacts/label_encoder.joblib
COPY artifacts/ohe_encoder.joblib ./artifacts/ohe_encoder.joblib

# Exposer le port utilisé par FastAPI
EXPOSE 8000

# Lancer l'application avec uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]