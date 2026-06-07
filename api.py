from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import numpy as np
import random
import time

app = FastAPI(
    title="IDS Real-Time API",
    description="Backend de détection d'intrusions réseau, inspiré par NVIDIA Morpheus.",
    version="1.0.0"
)

# Permettre au frontend (index.html) de communiquer avec l'API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variables globales pour stocker les données en mémoire
model = None
X_test_scaled_arr = None
raw_test_info = None
CLASS_LABELS = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']

@app.on_event("startup")
def load_data():
    global model, X_test_scaled_arr, raw_test_info
    print("Chargement du modèle et préparation des données en mémoire...")
    
    # Chargement du meilleur modèle sauvegardé
    model = joblib.load('best_model_cv.pkl')
    scaler = joblib.load('scaler.pkl')
    
    columns = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
        'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
        'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
        'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
        'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
        'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty_level'
    ]
    
    train_df = pd.read_csv('KDDTrain+.txt', names=columns)
    test_df = pd.read_csv('KDDTest+.txt', names=columns)
    
    attack_mapping = {
        'neptune': 'DoS', 'smurf': 'DoS', 'back': 'DoS', 'teardrop': 'DoS', 'pod': 'DoS', 'land': 'DoS',
        'satan': 'Probe', 'ipsweep': 'Probe', 'portsweep': 'Probe', 'nmap': 'Probe',
        'warezclient': 'R2L', 'guess_passwd': 'R2L', 'warezmaster': 'R2L', 'imap': 'R2L',
        'ftp_write': 'R2L', 'multihop': 'R2L', 'phf': 'R2L', 'spy': 'R2L',
        'buffer_overflow': 'U2R', 'rootkit': 'U2R', 'loadmodule': 'U2R', 'perl': 'U2R',
        'normal': 'Normal',
        'apache2': 'DoS', 'udpstorm': 'DoS', 'processtable': 'DoS', 'mailbomb': 'DoS', 'worm': 'DoS',
        'mscan': 'Probe', 'saint': 'Probe',
        'snmpgetattack': 'R2L', 'snmpguess': 'R2L', 'sendmail': 'R2L', 'named': 'R2L',
        'xlock': 'R2L', 'xsnoop': 'R2L', 'httptunnel': 'R2L',
        'ps': 'U2R', 'xterm': 'U2R', 'sqlattack': 'U2R'
    }
    
    # Informations brutes pour affichage dans le dashboard
    raw_test_info = test_df[['protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes']].copy()
    raw_test_info['true_attack_class'] = test_df['label'].map(attack_mapping)
    
    # Preprocessing
    train_df = train_df.drop(['label', 'difficulty_level'], axis=1)
    test_df = test_df.drop(['label', 'difficulty_level'], axis=1)
    
    X_train = pd.get_dummies(train_df, columns=['protocol_type', 'service', 'flag'])
    X_test = pd.get_dummies(test_df, columns=['protocol_type', 'service', 'flag'])
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    
    X_test_scaled_df = pd.DataFrame(scaler.transform(X_test), columns=X_train.columns)
    X_test_scaled_arr = X_test_scaled_df.values
    print("API Prête ! Le modèle XGBoost est chargé en mémoire.")

@app.get("/predict/random", summary="Simuler l'analyse d'une connexion réseau en temps réel")
def predict_random():
    """
    Sélectionne aléatoirement une connexion dans le jeu de test,
    exécute le modèle d'inférence, et retourne le résultat avec la latence.
    """
    idx = random.randint(0, len(X_test_scaled_arr) - 1)
    
    # Récupération des features encodées pour l'inférence
    row_features = X_test_scaled_arr[[idx]]
    
    # Chronométrage de la prédiction
    start = time.perf_counter()
    pred = model.predict(row_features)[0]
    probas = model.predict_proba(row_features)[0]
    end = time.perf_counter()
    
    confidence = float(np.max(probas))
    pred_label = CLASS_LABELS[pred]
    
    # Récupération des infos brutes pour le Frontend
    raw_info = raw_test_info.iloc[idx].to_dict()
    true_label = raw_info.pop('true_attack_class')
    
    # Vérification de la détection (Vrai Positif / Faux Positif, etc.)
    status = "Correct" if pred_label == true_label else "Erreur"
    
    return {
        "network_data": raw_info,
        "true_label": true_label,
        "prediction": pred_label,
        "confidence": round(confidence * 100, 2),
        "status": status,
        "latency_ms": round((end - start) * 1000, 3)
    }

@app.get("/", summary="Check santé de l'API")
def root():
    return {
        "status": "online",
        "model": "XGBoost + SMOTE",
        "message": "Allez sur /docs pour l'interface Swagger UI."
    }

if __name__ == "__main__":
    import uvicorn
    # Lancement du serveur en local sur le port 8000
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
