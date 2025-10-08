mlops_interpretability/
│
├── api/ # API FastAPI
│ ├── main.py
│ ├── requirements.txt
│ └── Dockerfile
│
├── data/ # données versionnées par DVC
├── artifacts/ # modèles versionnés par DVC
├── notebooks/
├── src/ # code ML (prétraitement, training, utils)
│ ├── preprocessing.py
│ ├── train.py
│ ├── evaluate.py
│ └── predict.py
│
├── .dvc/
├── .gitignore
├── data.dvc
├── artifacts.dvc
└── docker-compose.yml # orchestration