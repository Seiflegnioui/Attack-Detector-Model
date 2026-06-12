from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import joblib
import pandas as pd
import numpy as np
import random
import time
import os
import sys

# Ensure root is in path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Global memory
model = None
scaler = None
model_columns = None
X_test_scaled_arr = None
raw_test_info = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, scaler, model_columns, X_test_scaled_arr, raw_test_info
    print("[API] Loading models...")
    
    model = joblib.load('models/best_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    model_columns = joblib.load('models/model_columns.pkl')
    
    # Load pre-processed test data for simulation
    raw_test_info_df = joblib.load('data/processed/raw_test_info.pkl')
    raw_test_info = raw_test_info_df.copy()
    
    X_test_scaled_arr = joblib.load('data/processed/X_test_scaled_arr.pkl')
    print("[API] Ready to process traffic!")
    yield
    print("[API] Stopping API...")

app = FastAPI(
    title="IDS Real-Time API",
    description="Network intrusion detection backend with standard REST endpoints.",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://127.0.0.1:8080"],
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

@app.post("/predict", summary="Real-time inference on a custom connection")
def predict_live(conn: NetworkConnection):
    """Accepts JSON with 41 features and returns the predicted attack class."""
    start = time.perf_counter()
    df = pd.DataFrame([conn.model_dump() if hasattr(conn, 'model_dump') else conn.dict()])
    df_dummies = pd.get_dummies(df, columns=['protocol_type', 'service', 'flag'])
    df_aligned = df_dummies.reindex(columns=model_columns, fill_value=0)
    df_scaled = scaler.transform(df_aligned)
    
    pred = model.predict(df_scaled)[0]
    probas = model.predict_proba(df_scaled)[0]
    end = time.perf_counter()
    
    return {
        "prediction": config.CLASS_LABELS[pred],
        "confidence": round(float(np.max(probas)) * 100, 2),
        "latency_ms": round((end - start) * 1000, 3)
    }

@app.get("/predict/random", summary="Simulate a stream from the Test Set (Dashboard)")
def predict_random():
    idx = random.randint(0, len(X_test_scaled_arr) - 1)
    row_features = X_test_scaled_arr[[idx]]
    start = time.perf_counter()
    pred = model.predict(row_features)[0]
    probas = model.predict_proba(row_features)[0]
    end = time.perf_counter()
    
    pred_label = config.CLASS_LABELS[pred]
    raw_info = raw_test_info.iloc[idx].to_dict()
    true_label = raw_info.pop('true_attack_class')
    
    return {
        "network_data": raw_info,
        "true_label": true_label,
        "prediction": pred_label,
        "confidence": round(float(np.max(probas)) * 100, 2),
        "status": "Correct" if pred_label == true_label else "Error",
        "latency_ms": round((end - start) * 1000, 3)
    }

@app.get("/", summary="API Health Check")
def root():
    return {"status": "online", "model": "XGBoost + SMOTE", "message": "Go to /docs"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.api:app", host="0.0.0.0", port=8000, reload=True)
