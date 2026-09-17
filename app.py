from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Advanced Iris Analytics Studio")

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
        <title>Iris AI Analytics Studio</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap" rel="stylesheet">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            * { font-family: 'Space Grotesk', sans-serif; }
            body { background-color: #090d16; color: #f1f5f9; min-height: 100vh; padding: 30px 10px; }
            .dashboard-card {
                background: #131b2e;
                border: 1px solid #1e293b;
                border-radius: 20px;
                padding: 24px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.4);
            }
            .input-box {
                background: #0b1120;
                border: 1px solid #1e293b;
                border-radius: 12px;
                padding: 12px 16px;
            }
            .form-control-custom {
                background: transparent;
                border: none;
                color: #38bdf8;
                font-weight: 700;
                font-size: 1.1rem;
                width: 100%;
            }
            .form-control-custom:focus { outline: none; }
            .btn-run {
                background: linear-gradient(135deg, #0ea5e9, #6366f1);
                border: none;
                color: white;
                font-weight: 700;
                border-radius: 12px;
                padding: 14px;
                width: 100%;
                letter-spacing: 0.5px;
                transition: all 0.3s;
            }
            .btn-run:hover { opacity: 0.9; transform: translateY(-1px); }
            .progress-custom { height: 10px; border-radius: 5px; background: #1e293b; }
            .table-dark-custom { background: #0b1120; color: #cbd5e1; font-size: 0.85rem; }
        </style>
    </head>
    <body>
        <div class="container-fluid" style="max-width: 1200px;">
            <!-- Header -->
            <div class="d-flex justify-content-between align-items-center mb-4 pb-3 border-bottom border-secondary border-opacity-25">
                <div>
                    <h3 class="fw-bold m-0 text-white">Iris ML Analytics Dashboard</h3>
                    <p class="text-secondary small m-0">SVM Classification & Feature Radar Engine</p>
                </div>
                <span class="badge bg-primary bg-opacity-20 text-primary border border-primary px-3 py-2 rounded-pill">Model Status: Active</span>
            </div>

            <div class="row g-4">
                <!-- Column 1: Feature Inputs -->
                <div class="col-lg-4">
                    <div class="dashboard-card h-100">
                        <h5 class="fw-bold text-white mb-3">1. Feature Inputs</h5>
                        
                        <div class="d-flex gap-2 mb-3">
                            <button class="btn btn-sm btn-outline-secondary text-nowrap" onclick="setPreset(5.1, 3.5, 1.4, 0.2)">Setosa</button>
                            <button class="btn btn-sm btn-outline-secondary text-nowrap" onclick="setPreset(6.0, 2.9, 4.5, 1.5)">Versicolor</button>
                            <button class="btn btn-sm btn-outline-secondary text-nowrap" onclick="setPreset(6.9, 3.1, 5.4, 2.1)">Virginica</button>
                        </div>

                        <form id="irisForm" class="d-flex flex-column gap-3">
                            <div class="input-box">
                                <label class="small text-secondary fw-semibold">Sepal Length (cm)</label>
                                <input type="number" step="0.1" id="sl" class="form-control-custom" value="5.1">
                            </div>
                            <div class="input-box">
                                <label class="small text-secondary fw-semibold">Sepal Width (cm)</label>
                                <input type="number" step="0.1" id="sw" class="form-control-custom" value="3.5">
                            </div>
                            <div class="input-box">
                                <label class="small text-secondary fw-semibold">Petal Length (cm)</label>
                                <input type="number" step="0.1" id="pl" class="form-control-custom" value="1.4">
                            </div>
                            <div class="input-box">
                                <label class="small text-secondary fw-semibold">Petal Width (cm)</label>
                                <input type="number" step="0.1" id="pw" class="form-control-custom" value="0.2">
                            </div>
                            <button type="submit" class="btn-run mt-2">ANALYZE FEATURES 🚀</button>
                        </form>
                    </div>
                </div>

                <!-- Column 2: Radar Chart Visualization -->
                <div class="col-lg-4">
                    <div class="dashboard-card h-100 d-flex flex-column">
                        <h5 class="fw-bold text-white mb-3">2. Feature Radar Profile</h5>
                        <div class="flex-grow-1 d-flex align-items-center justify-content-center">
                            <canvas id="radarChart" style="max-height: 280px;"></canvas>
                        </div>
                    </div>
                </div>

                <!-- Column 3: Prediction & Class Probabilities -->
                <div class="col-lg-4">
                    <div class="dashboard-card h-100 d-flex flex-column justify-content-between">
                        <div>
                            <h5 class="fw-bold text-white mb-3">3. Classification Output</h5>
                            <div class="p-3 text-center rounded-3 mb-4" style="background: #0b1120; border: 1px solid #1e293b;">
                                <div class="small text-secondary text-uppercase fw-bold mb-1">Predicted Class</div>
                                <h2 id="predOutput" class="fw-bold text-info m-0">READY</h2>
                            </div>

                            <h6 class="small text-secondary fw-bold mb-3">Class Distribution Estimation</h6>
                            <div class="mb-3">
                                <div class="d-flex justify-content-between small mb-1">
                                    <span>Setosa</span><span id="p0">--</span>
                                </div>
                                <div class="progress progress-custom"><div id="pb0" class="progress-bar bg-info" style="width: 0%"></div></div>
                            </div>
                            <div class="mb-3">
                                <div class="d-flex justify-content-between small mb-1">
                                    <span>Versicolor</span><span id="p1">--</span>
                                </div>
                                <div class="progress progress-custom"><div id="pb1" class="progress-bar bg-primary" style="width: 0%"></div></div>
                            </div>
                            <div class="mb-3">
                                <div class="d-flex justify-content-between small mb-1">
                                    <span>Virginica</span><span id="p2">--</span>
                                </div>
                                <div class="progress progress-custom"><div id="pb2" class="progress-bar bg-indigo" style="background:#818cf8; width: 0%"></div></div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Live History Table -->
                <div class="col-12 mt-4">
                    <div class="dashboard-card">
                        <h5 class="fw-bold text-white mb-3">Prediction Audit Log</h5>
                        <div class="table-responsive">
                            <table class="table table-dark table-striped table-hover m-0 align-middle">
                                <thead>
                                    <tr class="text-secondary small">
                                        <th>TIME</th>
                                        <th>SEPAL LENGTH</th>
                                        <th>SEPAL WIDTH</th>
                                        <th>PETAL LENGTH</th>
                                        <th>PETAL WIDTH</th>
                                        <th>PREDICTED RESULT</th>
                                    </tr>
                                </thead>
                                <tbody id="historyBody">
                                    <tr><td colspan="6" class="text-center text-secondary small py-3">No inference logged yet</td></tr>
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            // Initialize Chart.js Radar
            const ctx = document.getElementById('radarChart').getContext('2d');
            const radarChart = new Chart(ctx, {
                type: 'radar',
                data: {
                    labels: ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'],
                    datasets: [{
                        label: 'Current Input Profile',
                        data: [5.1, 3.5, 1.4, 0.2],
                        backgroundColor: 'rgba(56, 189, 248, 0.2)',
                        borderColor: '#38bdf8',
                        pointBackgroundColor: '#38bdf8'
                    }]
                },
                options: {
                    scales: {
                        r: {
                            angleLines: { color: '#1e293b' },
                            grid: { color: '#1e293b' },
                            pointLabels: { color: '#94a3b8', font: { size: 10 } },
                            ticks: { display: false }
                        }
                    },
                    plugins: { legend: { display: false } }
                }
            });

            function setPreset(sl, sw, pl, pw) {
                document.getElementById('sl').value = sl;
                document.getElementById('sw').value = sw;
                document.getElementById('pl').value = pl;
                document.getElementById('pw').value = pw;
                triggerPrediction();
            }

            document.getElementById('irisForm').addEventListener('submit', (e) => {
                e.preventDefault();
                triggerPrediction();
            });

            async function triggerPrediction() {
                const sl = parseFloat(document.getElementById('sl').value);
                const sw = parseFloat(document.getElementById('sw').value);
                const pl = parseFloat(document.getElementById('pl').value);
                const pw = parseFloat(document.getElementById('pw').value);

                // Update Radar Chart
                radarChart.data.datasets[0].data = [sl, sw, pl, pw];
                radarChart.update();

                // Send API request
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ sepal_length: sl, sepal_width: sw, petal_length: pl, petal_width: pw })
                });

                const res = await response.json();
                document.getElementById('predOutput').innerText = res.prediction.toUpperCase();

                // Mock Confidence distribution for visual demo
                let probs = [0.05, 0.05, 0.05];
                probs[res.class_id] = 0.90;
                
                document.getElementById('p0').innerText = (probs[0]*100) + '%';
                document.getElementById('pb0').style.width = (probs[0]*100) + '%';
                document.getElementById('p1').innerText = (probs[1]*100) + '%';
                document.getElementById('pb1').style.width = (probs[1]*100) + '%';
                document.getElementById('p2').innerText = (probs[2]*100) + '%';
                document.getElementById('pb2').style.width = (probs[2]*100) + '%';

                // Add to History Table
                const tbody = document.getElementById('historyBody');
                if (tbody.children[0].children.length === 1) tbody.innerHTML = '';
                const row = `<tr>
                    <td>${new Date().toLocaleTimeString()}</td>
                    <td>${sl} cm</td>
                    <td>${sw} cm</td>
                    <td>${pl} cm</td>
                    <td>${pw} cm</td>
                    <td><span class="badge bg-info text-dark fw-bold">${res.prediction.toUpperCase()}</span></td>
                </tr>`;
                tbody.innerHTML = row + tbody.innerHTML;
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
