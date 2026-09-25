import os
import sqlite3
import numpy as np
import joblib
from datetime import datetime
from fastapi import APIRouter
from models import IrisInput
from database import DB_FILE

router = APIRouter(tags=["Prediction"])

AVAILABLE_KERNELS = ['linear', 'rbf', 'poly', 'sigmoid']
target_names = ['Setosa', 'Versicolor', 'Virginica']

def get_svm_model(kernel_name: str):
    model_file = f"svm_{kernel_name}_model.pkl"
    if not os.path.exists(model_file):
        from sklearn.datasets import load_iris
        from sklearn.svm import SVC
        iris = load_iris()
        model = SVC(kernel=kernel_name, probability=True, random_state=42)
        model.fit(iris.data, iris.target)
        joblib.dump(model, model_file)
    return joblib.load(model_file)

@router.post("/predict")
def predict_iris(data: IrisInput):
    kernel = data.kernel if data.kernel in AVAILABLE_KERNELS else "linear"
    model = get_svm_model(kernel)

    features = np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
    probs = model.predict_proba(features)[0]
    pred_idx = np.argmax(probs)
    prediction = target_names[pred_idx]
    confidence = float(probs[pred_idx])
    
    probabilities = [round(float(p) * 100, 1) for p in probs]
    is_anomaly = 1 if (data.petal_length < 1.0 or data.petal_length > 7.5) else 0

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO predictions (timestamp, model_used, sepal_length, sepal_width, petal_length, petal_width, prediction, confidence, is_anomaly)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (timestamp, f"SVM ({kernel})", data.sepal_length, data.sepal_width, data.petal_length, data.petal_width, prediction, confidence, is_anomaly))
    conn.commit()
    conn.close()

    return {
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": probabilities,
        "is_anomaly": bool(is_anomaly),
        "model_used": f"SVM ({kernel})"
    }
