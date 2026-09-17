from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Iris Classification Web App")

# Load mô hình SVM
model = joblib.load("svm_model.pkl")

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Trang web giao diện chính
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html lang="vi">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Dự Đoán Loài Hoa Iris</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; display: flex; align-items: center; justify-content: center; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
            .card { border-radius: 20px; box-shadow: 0 15px 35px rgba(0,0,0,0.2); backdrop-filter: blur(10px); background: rgba(255, 255, 255, 0.95); border: none; }
            .btn-custom { background: linear-gradient(to right, #667eea, #764ba2); border: none; color: white; border-radius: 10px; padding: 12px; font-weight: bold; width: 100%; transition: all 0.3s; }
            .btn-custom:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(118, 75, 162, 0.4); }
            .result-box { display: none; margin-top: 20px; padding: 15px; border-radius: 10px; background: #eef2ff; border-left: 5px solid #667eea; text-align: center; }
        </style>
    </head>
    <body>
        <div class="container" style="max-width: 500px;">
            <div class="card p-4 my-4">
                <h3 class="text-center mb-1 text-primary fw-bold">🌺 Iris Predictor</h3>
                <p class="text-center text-muted mb-4 fs-6">Nhập kích thước để phân loại hoa Iris bằng mô hình SVM</p>
                
                <form id="irisForm">
                    <div class="mb-3">
                        <label class="form-label fw-semibold">Độ dài đài hoa (Sepal Length): <span id="val1" class="text-primary">5.1</span> cm</label>
                        <input type="range" class="form-range" id="sepal_length" min="4.0" max="8.0" step="0.1" value="5.1" oninput="val1.innerText=this.value">
                    </div>
                    <div class="mb-3">
                        <label class="form-label fw-semibold">Độ rộng đài hoa (Sepal Width): <span id="val2" class="text-primary">3.5</span> cm</label>
                        <input type="range" class="form-range" id="sepal_width" min="2.0" max="4.5" step="0.1" value="3.5" oninput="val2.innerText=this.value">
                    </div>
                    <div class="mb-3">
                        <label class="form-label fw-semibold">Độ dài cánh hoa (Petal Length): <span id="val3" class="text-primary">1.4</span> cm</label>
                        <input type="range" class="form-range" id="petal_length" min="1.0" max="7.0" step="0.1" value="1.4" oninput="val3.innerText=this.value">
                    </div>
                    <div class="mb-3">
                        <label class="form-label fw-semibold">Độ rộng cánh hoa (Petal Width): <span id="val4" class="text-primary">0.2</span> cm</label>
                        <input type="range" class="form-range" id="petal_width" min="0.1" max="2.5" step="0.1" value="0.2" oninput="val4.innerText=this.value">
                    </div>
                    <button type="submit" class="btn btn-custom mt-2">Dự Đoán Ngay ✨</button>
                </form>

                <div id="result" class="result-box">
                    <h5 class="m-0 text-dark">Kết quả: <b id="predText" class="text-primary fs-4"></b></h5>
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
                document.getElementById('predText').innerText = result.prediction.toUpperCase();
                document.getElementById('result').style.display = 'block';
            });
        </script>
    </body>
    </html>
    """

# Endpoint dự đoán
@app.post("/predict")
def predict(data: IrisInput):
    input_data = np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
    prediction = model.predict(input_data)[0]
    
    species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    result_name = species_map.get(prediction, str(prediction))
    
    return {"class_id": int(prediction), "prediction": result_name}
