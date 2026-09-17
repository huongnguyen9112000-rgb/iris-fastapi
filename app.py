from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Iris AI Card Interactive Platform")

# Load mô hình SVM
model = joblib.load("svm_model.pkl")

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html lang="vi">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Iris AI Smart Studio</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;800&display=swap" rel="stylesheet">
        <style>
            * { font-family: 'Outfit', sans-serif; }
            body {
                background: #0f172a;
                color: #f8fafc;
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 20px;
            }
            .main-card {
                background: #1e293b;
                border-radius: 28px;
                padding: 30px;
                width: 100%;
                max-width: 520px;
                box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
                border: 1px solid #334155;
            }
            .stat-card {
                background: #0f172a;
                border: 1px solid #334155;
                border-radius: 18px;
                padding: 16px;
                text-align: center;
            }
            .btn-step {
                width: 32px;
                height: 32px;
                border-radius: 50%;
                border: none;
                background: #334155;
                color: white;
                font-weight: bold;
                font-size: 16px;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                cursor: pointer;
                transition: all 0.2s;
            }
            .btn-step:hover { background: #6366f1; }
            .preset-chip {
                background: #334155;
                border: 1px solid #475569;
                color: #cbd5e1;
                padding: 8px 14px;
                border-radius: 12px;
                font-size: 0.85rem;
                cursor: pointer;
                transition: all 0.2s;
            }
            .preset-chip:hover {
                background: #6366f1;
                color: white;
                border-color: #6366f1;
            }
            .btn-predict {
                background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
                border: none;
                color: white;
                font-weight: 800;
                border-radius: 16px;
                padding: 16px;
                font-size: 1.1rem;
                letter-spacing: 0.5px;
                width: 100%;
                box-shadow: 0 10px 20px rgba(99, 102, 241, 0.3);
                transition: transform 0.2s;
            }
            .btn-predict:active { transform: scale(0.98); }
            .result-overlay {
                display: none;
                margin-top: 20px;
                padding: 20px;
                border-radius: 20px;
                background: radial-gradient(circle at center, #312e81, #1e1b4b);
                border: 1px solid #6366f1;
                text-align: center;
            }
        </style>
    </head>
    <body>
        <div class="main-card">
            <div class="text-center mb-4">
                <span class="badge bg-indigo-500 text-indigo-200 mb-2 px-3 py-1 rounded-pill" style="background:#312e81; color:#a5b4fc;">Interactive AI Predictor</span>
                <h3 class="fw-bold m-0">Iris Flower Classifier</h3>
            </div>

            <!-- Preset Fast Options -->
            <div class="mb-4">
                <p class="small text-muted mb-2 fw-semibold">⚡ Chọn nhanh bộ mẫu thử:</p>
                <div class="d-flex gap-2 flex-wrap">
                    <span class="preset-chip" onclick="setValues(5.1, 3.5, 1.4, 0.2)">Mẫu Setosa</span>
                    <span class="preset-chip" onclick="setValues(6.0, 2.9, 4.5, 1.5)">Mẫu Versicolor</span>
                    <span class="preset-chip" onclick="setValues(6.9, 3.1, 5.4, 2.1)">Mẫu Virginica</span>
                </div>
            </div>

            <!-- Step Control Grid -->
            <div class="row g-3 mb-4">
                <div class="col-6">
                    <div class="stat-card">
                        <div class="small text-muted mb-1">Sepal Length</div>
                        <div class="d-flex align-items-center justify-content-between">
                            <button class="btn-step" onclick="adjust('sl', -0.1)">-</button>
                            <span id="sl_val" class="fw-bold fs-5 text-indigo-400">5.1</span>
                            <button class="btn-step" onclick="adjust('sl', 0.1)">+</button>
                        </div>
                    </div>
                </div>

                <div class="col-6">
                    <div class="stat-card">
                        <div class="small text-muted mb-1">Sepal Width</div>
                        <div class="d-flex align-items-center justify-content-between">
                            <button class="btn-step" onclick="adjust('sw', -0.1)">-</button>
                            <span id="sw_val" class="fw-bold fs-5 text-indigo-400">3.5</span>
                            <button class="btn-step" onclick="adjust('sw', 0.1)">+</button>
                        </div>
                    </div>
                </div>

                <div class="col-6">
                    <div class="stat-card">
                        <div class="small text-muted mb-1">Petal Length</div>
                        <div class="d-flex align-items-center justify-content-between">
                            <button class="btn-step" onclick="adjust('pl', -0.1)">-</button>
                            <span id="pl_val" class="fw-bold fs-5 text-indigo-400">1.4</span>
                            <button class="btn-step" onclick="adjust('pl', 0.1)">+</button>
                        </div>
                    </div>
                </div>

                <div class="col-6">
                    <div class="stat-card">
                        <div class="small text-muted mb-1">Petal Width</div>
                        <div class="d-flex align-items-center justify-content-between">
                            <button class="btn-step" onclick="adjust('pw', -0.1)">-</button>
                            <span id="pw_val" class="fw-bold fs-5 text-indigo-400">0.2</span>
                            <button class="btn-step" onclick="adjust('pw', 0.1)">+</button>
                        </div>
                    </div>
                </div>
            </div>

            <button class="btn-predict" onclick="runPrediction()">PHÂN LOẠI NGAY 🌸</button>

            <div id="result" class="result-overlay">
                <div class="text-uppercase small text-indigo-200 mb-1" style="letter-spacing: 1px;">KẾT QUẢ DỰ ĐOÁN</div>
                <h2 id="predText" class="fw-extrabold text-warning m-0">---</h2>
            </div>
        </div>

        <script>
            let store = { sl: 5.1, sw: 3.5, pl: 1.4, pw: 0.2 };

            function updateUI() {
                document.getElementById('sl_val').innerText = store.sl.toFixed(1);
                document.getElementById('sw_val').innerText = store.sw.toFixed(1);
                document.getElementById('pl_val').innerText = store.pl.toFixed(1);
                document.getElementById('pw_val').innerText = store.pw.toFixed(1);
            }

            function adjust(key, delta) {
                store[key] = Math.max(0.1, parseFloat((store[key] + delta).toFixed(1)));
                updateUI();
            }

            function setValues(sl, sw, pl, pw) {
                store = { sl, sw, pl, pw };
                updateUI();
                runPrediction();
            }

            async function runPrediction() {
                const data = {
                    sepal_length: store.sl,
                    sepal_width: store.sw,
                    petal_length: store.pl,
                    petal_width: store.pw
                };

                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
                
                const result = await response.json();
                const resElem = document.getElementById('predText');
                resElem.innerText = result.prediction.toUpperCase();
                document.getElementById('result').style.display = 'block';
            }
        </script>
    </body>
    </html>
    """

@app.post("/predict")
def predict(data: IrisInput):
    input_data = np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
    prediction = model.predict(input_data)[0]
    
    species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    result_name = species_map.get(prediction, str(prediction))
    
    return {"class_id": int(prediction), "prediction": result_name}
