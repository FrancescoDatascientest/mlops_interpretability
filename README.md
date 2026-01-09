# Projet MLOps Interprétabilité

Ce projet illustre une pipeline MLOps avec suivi d’interprétabilité des modèles.

## Structure du projet

```bash
mlops_interpretability/

├── .dvc/
├── .github/
├── api/
│   ├── main.py
│   ├── interpretability.py
│   └── requirements.txt
├── artifacts/
│   ├── xgb_pitch_model.joblib
│   ├── ohe_encoder.joblib
│   ├── rd.joblib
│   └── label_encoder.joblib
├── data/
├── notebooks/
├── training/
│   ├── preprocess.py
│   └── train.py
├── utils/
│   └── config_loader.py
├── .dvcignore
└── .gitignore