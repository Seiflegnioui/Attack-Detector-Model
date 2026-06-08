from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import joblib
import pandas as pd
import numpy as np
import random
import time

# Global memory
model = None
scaler = None
model_columns = None
X_test_scaled_arr = None
raw_test_info = None
CLASS_LABELS = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, scaler, model_columns, X_test_scaled_arr, raw_test_info
    print("[API] Loading models...")
    
    model = joblib.load('models/best_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    model_columns = joblib.load('models/model_columns.pkl')
    
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
    test_df = pd.read_csv('data/KDDTest+.txt', names=columns)
    
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
    raw_test_info = test_df[['protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes']].copy()
    raw_test_info['true_attack_class'] = test_df['label'].map(attack_mapping)
    test_df = test_df.drop(['label', 'difficulty_level'], axis=1)
    
    X_test = pd.get_dummies(test_df, columns=['protocol_type', 'service', 'flag'])
    X_test = X_test.reindex(columns=model_columns, fill_value=0)
    X_test_scaled_arr = scaler.transform(X_test)
    print("[API] Ready to process traffic!")
    yield
    print("[API] Stopping API...")

app = FastAPI(
    title="IDS Real-Time API",
    description="Backend de détection d'intrusions réseau avec endpoints REST standards.",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class NetworkConnection(BaseModel):
    duration: float = 0.0
    protocol_type: str = "tcp"
    service: str = "http"
    flag: str = "SF"
    src_bytes: float = 0.0
    dst_bytes: float = 0.0
    land: float = 0.0
    wrong_fragment: float = 0.0
    urgent: float = 0.0
    hot: float = 0.0
    num_failed_logins: float = 0.0
    logged_in: float = 0.0
    num_compromised: float = 0.0
    root_shell: float = 0.0
    su_attempted: float = 0.0
    num_root: float = 0.0
    num_file_creations: float = 0.0
    num_shells: float = 0.0
    num_access_files: float = 0.0
    num_outbound_cmds: float = 0.0
    is_host_login: float = 0.0
    is_guest_login: float = 0.0
    count: float = 0.0
    srv_count: float = 0.0
    serror_rate: float = 0.0
    srv_serror_rate: float = 0.0
    rerror_rate: float = 0.0
    srv_rerror_rate: float = 0.0
    same_srv_rate: float = 0.0
    diff_srv_rate: float = 0.0
    srv_diff_host_rate: float = 0.0
    dst_host_count: float = 0.0
    dst_host_srv_count: float = 0.0
    dst_host_same_srv_rate: float = 0.0
    dst_host_diff_srv_rate: float = 0.0
    dst_host_same_src_port_rate: float = 0.0
    dst_host_srv_diff_host_rate: float = 0.0
    dst_host_serror_rate: float = 0.0
    dst_host_srv_serror_rate: float = 0.0
    dst_host_rerror_rate: float = 0.0
    dst_host_srv_rerror_rate: float = 0.0

@app.post("/predict", summary="Inférence Temps Réel sur une connexion personnalisée")
def predict_live(conn: NetworkConnection):
    """Accepte un JSON avec 41 features et retourne la classe d'attaque prédite."""
    start = time.perf_counter()
    df = pd.DataFrame([conn.dict()])
    df_dummies = pd.get_dummies(df, columns=['protocol_type', 'service', 'flag'])
    df_aligned = df_dummies.reindex(columns=model_columns, fill_value=0)
    df_scaled = scaler.transform(df_aligned)
    
    pred = model.predict(df_scaled)[0]
    probas = model.predict_proba(df_scaled)[0]
    end = time.perf_counter()
    
    return {
        "prediction": CLASS_LABELS[pred],
        "confidence": round(float(np.max(probas)) * 100, 2),
        "latency_ms": round((end - start) * 1000, 3)
    }

@app.get("/predict/random", summary="Simuler un flux depuis le Test Set (Dashboard)")
def predict_random():
    idx = random.randint(0, len(X_test_scaled_arr) - 1)
    row_features = X_test_scaled_arr[[idx]]
    start = time.perf_counter()
    pred = model.predict(row_features)[0]
    probas = model.predict_proba(row_features)[0]
    end = time.perf_counter()
    
    pred_label = CLASS_LABELS[pred]
    raw_info = raw_test_info.iloc[idx].to_dict()
    true_label = raw_info.pop('true_attack_class')
    
    return {
        "network_data": raw_info,
        "true_label": true_label,
        "prediction": pred_label,
        "confidence": round(float(np.max(probas)) * 100, 2),
        "status": "Correct" if pred_label == true_label else "Erreur",
        "latency_ms": round((end - start) * 1000, 3)
    }

@app.get("/", summary="Check santé de l'API")
def root():
    return {"status": "online", "model": "XGBoost + SMOTE", "message": "Allez sur /docs"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
