from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
import numpy as np
import time

app = FastAPI(title="Cyberpunk Iris AI Command Center")

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
        <title>Cyber Iris AI Command Center</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Rajdhani:wght@500;600;700&display=swap" rel="stylesheet">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            :root {
                --neon-cyan: #00f3ff;
                --neon-magenta: #ff0055;
                --neon-purple: #b026ff;
                --bg-dark: #050811;
            }
            * { font-family: 'Rajdhani', sans-serif; }
            h1, h2, h3, .brand-font { font-family: 'Orbitron', sans-serif; }
            
            body {
                background-color: var(--bg-dark);
                background-image: 
                    radial-gradient(circle at 10% 20%, rgba(0, 243, 255, 0.05) 0%, transparent 20%),
                    radial-gradient(circle at 90% 80%, rgba(255, 0, 85, 0.05) 0%, transparent 20%);
                color: #e2e8f0;
                min-height: 100vh;
                padding: 20px;
            }

            .glass-panel {
                background: rgba(15, 23, 42, 0.65);
                backdrop-filter: blur(16px);
                -webkit-backdrop-filter: blur(16px);
                border: 1px solid rgba(0, 243, 255, 0.2);
                border-radius: 16px;
                box-shadow: 0 0 30px rgba(0, 243, 255, 0.05);
                padding: 24px;
                position: relative;
                overflow: hidden;
            }

            .glass-panel::before {
                content: '';
                position: absolute;
                top: 0; left: 0; width: 100%; height: 2px;
                background: linear-gradient(90deg, transparent, var(--neon-cyan), transparent);
            }

            .cyber-btn {
                background: transparent;
                border: 1px solid var(--neon-cyan);
                color: var(--neon-cyan);
                font-family: 'Orbitron', sans-serif;
                font-weight: 700;
                padding: 12px;
                border-radius: 8px;
                width: 100%;
                letter-spacing: 2px;
                transition: all 0.3s ease;
                text-shadow: 0 0 8px var(--neon-cyan);
            }

            .cyber-btn:hover {
                background: var(--neon-cyan);
                color: #000;
                box-shadow: 0 0 20px var(--neon-cyan);
            }

            .stat-badge {
                border-left: 3px solid var(--neon-cyan);
                background: rgba(0, 243, 255, 0.05);
                padding: 8px 12px;
            }

            .input-cyber {
                background: rgba(0, 0, 0, 0.5);
                border: 1px solid #1e293b;
                color: var(--neon-cyan);
                font-family: 'Orbitron', sans-serif;
                font-weight: 700;
                border-radius: 8px;
                padding: 8px 12px;
                width: 100%;
            }

            .input-cyber:focus {
                outline: none;
                border-color: var(--neon-cyan);
                box-shadow: 0 0 10px rgba(0, 243, 255, 0.3);
            }

            .pulse-dot {
                width: 8px; height: 8px;
                background-color: var(--neon-cyan);
                border-radius: 50%;
                display: inline-block;
                box-shadow: 0 0 10px var(--neon-cyan);
                animation: pulse 1.5s infinite;
            }

            @keyframes pulse {
                0% { opacity: 0.3; }
                50% { opacity: 1; }
                100% { opacity: 0.3; }
            }
        </style>
    </head>
    <body>
        <div class="container-fluid" style="max-width: 1300px;">
            <!-- Header HUD -->
            <div class="glass-panel mb-4 d-flex justify-content-between align-items-center">
                <div>
                    <h2 class="brand-font m-0 text-transparent bg-clip-text" style="color: var(--neon-cyan);">CYBER_IRIS // AI CORE</h2>
                    <div class="small text-muted"><span class="pulse-dot me-2"></span>NEURAL NETWORK INFERENCE ENGINE v3.0</div>
                </div>
                <div class="d-flex gap-4 text-end">
                    <div class="stat-badge">
                        <div class="small text-muted">API LATENCY</div>
                        <div id="latencyVal" class="fw-bold text-warning">0 ms</div>
                    </div>
                    <div class="stat-badge">
                        <div class="small text-muted">TOTAL CALLS</div>
                        <div id="callsVal" class="fw-bold text-info">0</div>
                    </div>
                </div>
            </div>

            <div class="row g-4">
                <!-- Controls Panel -->
                <div class="col-lg-4">
                    <div class="glass-panel h-100">
                        <h5 class="brand-font text-white mb-3">// INPUT MATRIX</h5>
                        
                        <div class="d-flex gap-2 mb-4">
                            <button class="btn btn-sm btn-outline-info flex-grow-1" onclick="loadPreset(5.1, 3.5, 1.4, 0.2)">SETOSA</button>
                            <button class="btn btn-sm btn-outline-info flex-grow-1" onclick="loadPreset(6.0, 2.9, 4.5, 1.5)">VERSICOLOR</button>
                            <button class="btn btn-sm btn-outline-info flex-grow-1" onclick="loadPreset(6.9, 3.1, 5.4, 2.1)">VIRGINICA</button>
                        </div>

                        <div class="d-flex flex-column gap-3 mb-4">
                            <div>
                                <label class="small text-muted mb-1">SEPAL LENGTH (CM)</label>
                                <input type="number" step="0.1" id="sl" class="input-cyber" value="5.1">
                            </div>
                            <div>
                                <label class="small text-muted mb-1">SEPAL WIDTH (CM)</label>
                                <input type="number" step="0.1" id="sw" class="input-cyber" value="3.5">
                            </div>
                            <div>
                                <label class="small text-muted mb-1">PETAL LENGTH (CM)</label>
                                <input type="number" step="0.1" id="pl" class="input-cyber" value="1.4">
                            </div>
                            <div>
                                <label class="small text-muted mb-1">PETAL WIDTH (CM)</label>
                                <input type="number" step="0.1" id="pw" class="input-cyber" value="0.2">
                            </div>
                        </div>

                        <button class="cyber-btn" onclick="executeInference()">EXECUTE INFERENCE ⚡</button>
                    </div>
                </div>

                <!-- Radar Display -->
                <div class="col-lg-4">
                    <div class="glass-panel h-100 d-flex flex-column">
                        <h5 class="brand-font text-white mb-3">// FEATURE SPECTRUM</h5>
                        <div class="flex-grow-1 d-flex align-items-center justify-content-center">
                            <canvas id="radarCanvas" style="max-height: 300px;"></canvas>
                        </div>
                    </div>
                </div>

                <!-- AI Output HUD -->
                <div class="col-lg-4">
                    <div class="glass-panel h-100 d-flex flex-column justify-content-between">
                        <div>
                            <h5 class="brand-font text-white mb-3">// CLASSIFICATION HUD</h5>
                            
                            <div class="p-4 rounded-3 text-center mb-4" style="background: rgba(0,0,0,0.6); border: 1px solid var(--neon-cyan);">
                                <div class="small text-muted mb-1">DETECTED SPECIES</div>
                                <h1 id="targetClass" class="brand-font fw-bold m-0" style="color: var(--neon-cyan); text-shadow: 0 0 15px var(--neon-cyan);">STANDBY</h1>
                            </div>

                            <div class="mb-2 d-flex justify-content-between align-items-center">
                                <span class="small text-muted">VOICE FEEDBACK</span>
                                <div class="form-check form-switch">
                                    <input class="form-check-input" type="checkbox" id="voiceToggle" checked>
                                </div>
                            </div>
                        </div>

                        <div class="border-top border-secondary pt-3">
                            <div class="small text-muted mb-2">// MODEL METADATA</div>
                            <div class="d-flex justify-content-between small text-light">
                                <span>Algorithm:</span><span class="text-info">Support Vector Machine (SVM)</span>
                            </div>
                            <div class="d-flex justify-content-between small text-light">
                                <span>Precision:</span><span class="text-success">96.67%</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            let totalCalls = 0;

            // Chart.js Setup
            const ctx = document.getElementById('radarCanvas').getContext('2d');
            const radarChart = new Chart(ctx, {
                type: 'radar',
                data: {
                    labels: ['Sepal Len', 'Sepal Wid', 'Petal Len', 'Petal Wid'],
                    datasets: [{
                        label: 'Metrics',
                        data: [5.1, 3.5, 1.4, 0.2],
                        backgroundColor: 'rgba(0, 243, 255, 0.15)',
                        borderColor: '#00f3ff',
                        pointBackgroundColor: '#00f3ff',
                        pointBorderColor: '#fff'
                    }]
                },
                options: {
                    scales: {
                        r: {
                            angleLines: { color: 'rgba(255, 255, 255, 0.1)' },
                            grid: { color: 'rgba(255, 255, 255, 0.1)' },
                            pointLabels: { color: '#00f3ff', font: { family: 'Orbitron', size: 11 } },
                            ticks: { display: false }
                        }
                    },
                    plugins: { legend: { display: false } }
                }
            });

            function loadPreset(sl, sw, pl, pw) {
                document.getElementById('sl').value = sl;
                document.getElementById('sw').value = sw;
                document.getElementById('pl').value = pl;
                document.getElementById('pw').value = pw;
                executeInference();
            }

            async function executeInference() {
                const startTime = performance.now();
                
                const sl = parseFloat(document.getElementById('sl').value);
                const sw = parseFloat(document.getElementById('sw').value);
                const pl = parseFloat(document.getElementById('pl').value);
                const pw = parseFloat(document.getElementById('pw').value);

                // Update Chart
                radarChart.data.datasets[0].data = [sl, sw, pl, pw];
                radarChart.update();

                // Call API
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ sepal_length: sl, sepal_width: sw, petal_length: pl, petal_width: pw })
                });

                const result = await response.json();
                const endTime = performance.now();

                // Metrics Update
                totalCalls++;
                document.getElementById('callsVal').innerText = totalCalls;
                document.getElementById('latencyVal').innerText = Math.round(endTime - startTime) + ' ms';

                // Display Prediction
                const species = result.prediction.toUpperCase();
                document.getElementById('targetClass').innerText = species;

                // AI Voice Synthesis
                if (document.getElementById('voiceToggle').checked) {
                    speakResult("Predicted species is " + species);
                }
            }

            function speakResult(text) {
                if ('speechSynthesis' in window) {
                    window.speechSynthesis.cancel(); // Stop previous voice
                    const msg = new SpeechSynthesisUtterance(text);
                    msg.rate = 1.0;
                    msg.pitch = 0.9;
                    window.speechSynthesis.speak(msg);
                }
            }
        </script>
    </body>
    </html>
    """

@app.post("/predict")
def predict(data: IrisInput):
    start_time = time.time()
    
    input_data = np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
    prediction = model.predict(input_data)[0]
    
    species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    result_name = species_map.get(prediction, str(prediction))
    
    latency = round((time.time() - start_time) * 1000, 2)
    
    return {
        "class_id": int(prediction), 
        "prediction": result_name,
        "server_latency_ms": latency
    }
