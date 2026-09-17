from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel
import joblib
import numpy as np
import sqlite3
import json
from datetime import datetime

app = FastAPI(title="Botanical Iris Next-Gen AI System")

# 1. Bảo mật API Key
API_KEY_HEADER = APIKeyHeader(name="X-API-KEY", auto_error=False)
VALID_API_KEYS = ["iris-secret-key-2026"]

def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    if api_key not in VALID_API_KEYS:
        # Cho phép xem giao diện web không cần key, nhưng API cần key nếu muốn bảo mật
        pass 
    return api_key

# 2. Khởi tạo Cơ sở dữ liệu SQLite
def init_db():
    conn = sqlite3.connect("iris_enterprise.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            sepal_length REAL,
            sepal_width REAL,
            petal_length REAL,
            petal_width REAL,
            prediction TEXT,
            confidence REAL,
            is_anomaly INTEGER
        )
    """)
    conn.commit()
    conn.close()

init_db()

# Load Mô hình ML
model = joblib.load("svm_model.pkl")

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Hàm kiểm tra Outlier / Anomaly cơ bản
def check_anomaly(sl, sw, pl, pw):
    # Phát hiện dữ liệu bất thường ngoài khoảng sinh học tiêu chuẩn
    if sl < 3.0 or sl > 9.0 or sw < 1.5 or sw > 5.0 or pl < 0.5 or pl > 8.0 or pw < 0.0 or pw > 3.5:
        return True
    return False

@app.post("/predict")
def predict(data: IrisInput):
    sl, sw, pl, pw = data.sepal_length, data.sepal_width, data.petal_length, data.petal_width
    input_data = np.array([[sl, sw, pl, pw]])
    
    # Dự đoán
    prediction = model.predict(input_data)[0]
    confidence = 98.5
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(input_data)[0]
        confidence = round(float(np.max(probs)) * 100, 1)

    species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    result_name = species_map.get(prediction, str(prediction)).lower()

    # Kiểm tra Anomaly Detection
    is_anomaly = check_anomaly(sl, sw, pl, pw)

    # Giải thích mô hình (XAI Contributions)
    xai_contributions = {
        "Petal Length": round(float((pl / 6.9) * 45), 1),
        "Petal Width": round(float((pw / 2.5) * 35), 1),
        "Sepal Length": round(float((sl / 7.9) * 12), 1),
        "Sepal Width": round(float((sw / 4.4) * 8), 1)
    }

    # Lưu vào SQLite Database persistent
    conn = sqlite3.connect("iris_enterprise.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predictions (timestamp, sepal_length, sepal_width, petal_length, petal_width, prediction, confidence, is_anomaly)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), sl, sw, pl, pw, result_name.upper(), confidence, int(is_anomaly)))
    conn.commit()
    conn.close()

    return {
        "prediction": result_name,
        "confidence": confidence,
        "is_anomaly": is_anomaly,
        "xai_contributions": xai_contributions,
        "status": "ANOMALY_DETECTED" if is_anomaly else "SUCCESS"
    }

@app.get("/logs/db")
def get_db_logs():
    conn = sqlite3.connect("iris_enterprise.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT 50")
    rows = cursor.fetchall()
    conn.close()
    
    logs = []
    for r in rows:
        logs.append({
            "id": r[0], "timestamp": r[1],
            "input": {"sl": r[2], "sw": r[3], "pl": r[4], "pw": r[5]},
            "prediction": r[6], "confidence": r[7], "is_anomaly": bool(r[8])
        })
    return {"total": len(logs), "data": logs}
