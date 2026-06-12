# Utilisation d'une image Python légère et officielle
FROM python:3.11-slim


# Définition du répertoire de travail dans le conteneur
WORKDIR /app

# Copie uniquement du fichier des dépendances dans un premier temps 
# (Permet de mettre en cache cette étape longue)
COPY requirements.txt .

# Installation des dépendances sans utiliser de cache pour garder l'image légère
RUN pip install --no-cache-dir -r requirements.txt

# Copie des fichiers vitaux pour l'API (Code, Modèles, Scaler)
# PRE-REQUIS : Run `notebooks/prepare_data.ipynb` then `notebooks/train_model.ipynb`
# Generated files: models/best_model.pkl, models/scaler.pkl, models/model_columns.pkl
COPY config.py .
COPY api/api.py api/
COPY models/best_model.pkl models/
COPY models/scaler.pkl models/

# Copie des datasets nécessaires au preprocessing initial au démarrage de l'API
COPY models/model_columns.pkl models/
COPY data/processed/raw_test_info.pkl data/processed/
COPY data/processed/X_test_scaled_arr.pkl data/processed/


# Exposition du port sur lequel l'API va écouter
EXPOSE 8000

# Commande de lancement du serveur asynchrone Uvicorn
CMD ["uvicorn", "api.api:app", "--host", "0.0.0.0", "--port", "8000"]
