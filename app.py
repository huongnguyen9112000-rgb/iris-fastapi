from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Iris AI Glassmorphism Platform")

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
        <title>Iris Classification AI Lab</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;800&display=swap" rel="stylesheet">
        <style>
            * { font-family: 'Plus Jakarta Sans', sans-serif; }
            body {
                background: radial-gradient(circle at top left, #1e1e38, #0d0e15);
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                color: #e2e8f0;
                overflow-x: hidden;
            }
            .glass-card {
                background: rgba(255, 255, 255, 0.04);
                backdrop-filter: blur(16px);
                -webkit-backdrop-filter: blur(16px);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 24px;
                padding: 2.5rem;
                box-shadow: 0 30px 60px rgba(0, 0, 0, 0.4);
                position: relative;
            }
            .glass-card::before {
                content: '';
                position: absolute;
                top: -2px; left: -2px; right: -2px; bottom: -2px;
                background: linear-gradient(45deg, #a855f7, #3b82f6, transparent);
                border-radius: 26px;
                z-index: -1;
                opacity: 0.3;
            }
            .badge-neon {
                background: rgba(168, 85, 247, 0.15);
                color: #c084fc;
                border: 1px solid rgba(168, 85, 247, 0.3);
                padding: 6px 16px;
                border-radius: 20px;
                font-size: 0.8rem;
                letter-spacing: 1px;
                text-transform: uppercase;
                font-weight: 700;
            }
            .form-range::-webkit-slider-thumb {
                background: #c084fc;
                box-shadow: 0 0 10px #c084fc;
            }
            .btn-cyber {
                background: linear-gradient(135deg, #8b5cf6 0%, #3b82f6 100%);
                border: none;
                color: white;
                font-weight: 700;
                border-radius: 14px;
                padding: 14px;
                letter-spacing: 0.5px;
                transition: all 0.3s ease;
                box-shadow: 0 10px 20px rgba(139, 92, 246, 0.3);
            }
            .btn-cyber:hover {
                transform: translateY(-2px);
                box-shadow: 0 15px 30px rgba(139, 92, 246, 0.5);
                color: white;
            }
            .result-card {
                display: none;
                margin-top: 1.5rem;
                padding: 1.25rem;
                border-radius: 16px;
                background: rgba(255, 255, 255, 0.05);
                border: 1px solid rgba(255, 255, 255, 0.1);
                text-align: center;
                animation: fadeIn 0.5s ease-out forwards;
            }
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }
        </style>
    </head>
    <body>
        <div class="container" style="max-width: 480px;">
            <div class="glass-card">
                <div class="text-center mb-4">
                    <span class="badge-neon mb-2 d-inline-block">SVM Intelligence Engine</span>
                    <h2 class="fw-extrabold text-white mt-2">Iris AI Predictor</h2>
                    <p class="text-secondary small">Hệ thống phân loại đa chiều thông số hoa Iris</p>
                </div>

                <form id="irisForm">
                    <div class="mb-3">
                        <div class="d-flex justify-content-between small mb-1">
                            <span class="text-light">Sepal Length</span>
                            <span id="v1" class="text-info fw-bold">5.1 cm</span>
                        </div>
                        <input type="range" class="form-range" id="sepal_length" min="4.0" max="8.0" step="0.1" value="5.1" oninput="v1.innerText=this.value+' cm'">
                    </div>

                    <div class="mb-3">
                        <div class="d-flex justify-content-between small mb-1">
                            <span class="text-light">Sepal Width</span>
                            <span id="v2" class="text-info fw-bold">3.5 cm</span>
                        </div>
                        <input type="range" class="form-range" id="sepal_width" min="2.0" max="4.5" step="0.1" value="3.5" oninput="v2.innerText=this.value+' cm'">
                    </div>

                    <div class="mb-3">
                        <div class="d-flex justify-content-between small mb-1">
                            <span class="text-light">Petal Length</span>
                            <span id="v3" class="text-info fw-bold">1.4 cm</span>
                        </div>
                        <input type="range" class="form-range" id="petal_length" min="1.0" max="7.0" step="0.1" value="1.4" oninput="v3.innerText=this.value+' cm'">
                    </div>

                    <div class="mb-4">
                        <div class="d-flex justify-content-between small mb-1">
                            <span class="text-light">Petal Width</span>
                            <span id="v4" class="text-info fw-bold">0.2 cm</span>
                        </div>
                        <input type="range" class="form-range" id="petal_width" min="0.1" max="2.5" step="0.1" value="0.2" oninput="v4.innerText=this.value+' cm'">
                    </div>

                    <button type="submit" class="btn btn-cyber w-100">RUN PREDICTION ⚡</button>
                </form>

                <div id="result" class="result-card">
                    <div class="text-secondary small text-uppercase mb-1">Predicted Class</div>
                    <h3 id="predText" class="fw-bold m-0 text-capitalize" style="color: #38bdf8;">---</h3>
                </div>
            </div>
        </div>

        <script>
            document.getElementById('irisForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                const data = {
                    sepal_length: parseFloat(document.getElementById('sepal_length').value),
                    sepal_width: parseFloat(document.getElementById('sepal_width').value),
                    petal_length: parseFloat(document.getElementById('petal_length').value),
                    petal_width: parseFloat(document.getElementById('petal_width').value)
                };

                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
                
                const result = await response.json();
                const predElem = document.getElementById('predText');
                predElem.innerText = result.prediction;

                // Dynamic colors for predicted species
                if (result.prediction === 'setosa') predElem.style.color = '#38bdf8'; // Cyan
                else if (result.prediction === 'versicolor') predElem.style.color = '#c084fc'; // Purple
                else predElem.style.color = '#f43f5e'; // Pink-Red

                document.getElementById('result').style.display = 'block';
            });
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
