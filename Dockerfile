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
COPY api.py .
COPY best_model_cv.pkl .
COPY scaler.pkl .

# Copie des datasets nécessaires au preprocessing initial au démarrage de l'API
COPY KDDTrain+.txt .
COPY KDDTest+.txt .

# Exposition du port sur lequel l'API va écouter
EXPOSE 8000

# Commande de lancement du serveur asynchrone Uvicorn
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
