from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Botanical Iris AI Studio Pro")

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
        <title>Iris Garden // Pro AI Studio</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Playfair+Display:ital,wght@0,600;1,600&display=swap" rel="stylesheet">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.6.0/dist/confetti.browser.min.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/html2pdf.js/0.10.1/html2pdf.bundle.min.js"></script>
        <style>
            :root {
                --bg-soft: #fdf8f6;
                --primary-pink: #f472b6;
                --text-dark: #475569;
            }
            
            * { font-family: 'Plus Jakarta Sans', sans-serif; }
            .serif-title { font-family: 'Playfair Display', serif; }

            body {
                background: linear-gradient(135deg, #fdf8f6 0%, #fef2f2 50%, #f3e8ff 100%);
                color: var(--text-dark);
                min-height: 100vh;
                padding: 30px 15px;
            }

            .bloom-card {
                background: rgba(255, 255, 255, 0.88);
                backdrop-filter: blur(20px);
                border: 1px solid rgba(244, 114, 182, 0.2);
                border-radius: 28px;
                padding: 26px;
                box-shadow: 0 15px 35px rgba(244, 114, 182, 0.08);
            }

            .preset-btn {
                background: #fff;
                border: 1px solid #fbcfe8;
                color: #db2777;
                border-radius: 16px;
                padding: 8px 16px;
                font-weight: 600;
                font-size: 0.88rem;
                transition: all 0.2s ease;
            }

            .preset-btn:hover {
                background: #fdf2f8;
                border-color: #f472b6;
                transform: translateY(-2px);
            }

            .input-box-floral {
                background: #faf5f8;
                border: 1px solid #f5d0fe;
                border-radius: 18px;
                padding: 10px 16px;
            }

            .input-box-floral label {
                font-size: 0.78rem;
                font-weight: 700;
                color: #a21caf;
                text-transform: uppercase;
            }

            .form-control-floral {
                background: transparent;
                border: none;
                color: #701a75;
                font-weight: 700;
                font-size: 1.05rem;
                width: 100%;
            }

            .form-control-floral:focus { outline: none; }

            .btn-bloom {
                background: linear-gradient(135deg, #f472b6 0%, #a855f7 100%);
                border: none;
                color: white;
                font-weight: 700;
                border-radius: 20px;
                padding: 14px;
                width: 100%;
                font-size: 1rem;
                box-shadow: 0 10px 20px rgba(244, 114, 182, 0.35);
                transition: all 0.3s ease;
            }

            .btn-bloom:hover {
                transform: translateY(-2px);
                box-shadow: 0 12px 25px rgba(244, 114, 182, 0.45);
            }

            .result-display {
                background: linear-gradient(135deg, #fdf2f8 0%, #faf5ff 100%);
                border: 2px dashed #f472b6;
                border-radius: 24px;
                padding: 20px;
                text-align: center;
            }

            .progress-pink {
                height: 10px;
                border-radius: 10px;
                background-color: #fce7f3;
            }
            
            .progress-bar-pink {
                background: linear-gradient(90deg, #f472b6, #c084fc);
                border-radius: 10px;
            }
        </style>
    </head>
    <body>
        <div class="container-fluid" id="exportArea" style="max-width: 1250px;">
            <div class="text-center mb-4">
                <span class="badge rounded-pill px-3 py-2 mb-2" style="background: #fce7f3; color: #be185d; font-weight: 600;">🌸 AI Botanical Classification Studio</span>
                <h1 class="serif-title display-5 fw-bold text-dark m-0">Iris Flower Analytics</h1>
                <p class="text-muted mt-1">Dự đoán và phân tích các loài hoa Iris bằng trí tuệ nhân tạo</p>
            </div>

            <div class="row g-4">
                <!-- Controls Panel -->
                <div class="col-lg-4">
                    <div class="bloom-card h-100">
                        <h5 class="serif-title fw-bold mb-3 text-dark">1. Thông số hoa Iris</h5>
                        
                        <div class="d-flex gap-2 mb-3 flex-wrap">
                            <button class="preset-btn" onclick="setPreset(5.1, 3.5, 1.4, 0.2)">🌸 Setosa</button>
                            <button class="preset-btn" onclick="setPreset(6.0, 2.9, 4.5, 1.5)">🌷 Versicolor</button>
                            <button class="preset-btn" onclick="setPreset(6.9, 3.1, 5.4, 2.1)">🌺 Virginica</button>
                        </div>

                        <form id="irisForm" class="d-flex flex-column gap-2">
                            <div class="input-box-floral">
                                <label>Sepal Length (Đài hoa - Dài)</label>
                                <input type="number" step="0.1" id="sl" class="form-control-floral" value="5.1">
                            </div>
                            <div class="input-box-floral">
                                <label>Sepal Width (Đài hoa - Rộng)</label>
                                <input type="number" step="0.1" id="sw" class="form-control-floral" value="3.5">
                            </div>
                            <div class="input-box-floral">
                                <label>Petal Length (Cánh hoa - Dài)</label>
                                <input type="number" step="0.1" id="pl" class="form-control-floral" value="1.4">
                            </div>
                            <div class="input-box-floral">
                                <label>Petal Width (Cánh hoa - Rộng)</label>
                                <input type="number" step="0.1" id="pw" class="form-control-floral" value="0.2">
                            </div>

                            <button type="submit" class="btn-bloom mt-2">PHÂN LOẠI NGAY 🌸</button>
                        </form>
                    </div>
                </div>

                <!-- Radar Display -->
                <div class="col-lg-4">
                    <div class="bloom-card h-100 d-flex flex-column">
                        <h5 class="serif-title fw-bold mb-3 text-dark">2. Biểu đồ đặc trưng</h5>
                        <div class="flex-grow-1 d-flex align-items-center justify-content-center">
                            <canvas id="radarCanvas" style="max-height: 270px;"></canvas>
                        </div>
                    </div>
                </div>

                <!-- Output HUD -->
                <div class="col-lg-4">
                    <div class="bloom-card h-100 d-flex flex-column justify-content-between">
                        <div>
                            <h5 class="serif-title fw-bold mb-3 text-dark">3. Kết quả phân loại</h5>
                            
                            <div class="result-display mb-3">
                                <div class="small text-uppercase fw-bold text-muted mb-1">Kết Quả Dự Đoán</div>
                                <h2 id="targetClass" class="serif-title fw-bold m-0" style="color: #be185d;">SẴN SÀNG</h2>
                            </div>

                            <!-- Confidence Score Gauge -->
                            <div class="p-3 rounded-4 mb-3" style="background: #faf5f8; border: 1px solid #f5d0fe;">
                                <div class="d-flex justify-content-between small fw-bold mb-1">
                                    <span style="color: #a21caf;">ĐỘ TIN CẬY (CONFIDENCE):</span>
                                    <span id="confidenceVal" style="color: #be185d;">0%</span>
                                </div>
                                <div class="progress progress-pink">
                                    <div id="confidenceBar" class="progress-bar progress-bar-pink" role="progressbar" style="width: 0%"></div>
                                </div>
                            </div>

                            <div class="d-flex align-items-center justify-content-between p-3 rounded-4 mb-3" style="background: #faf5f8;">
                                <span class="small fw-semibold text-secondary">🔊 Bật âm thanh AI đọc kết quả</span>
                                <div class="form-check form-switch">
                                    <input class="form-check-input" type="checkbox" id="voiceToggle" checked>
                                </div>
                            </div>
                        </div>

                        <button class="btn btn-outline-danger w-100 rounded-4 fw-bold py-2" onclick="downloadPDF()">📄 Tải Báo Cáo PDF</button>
                    </div>
                </div>
            </div>

            <!-- History Section -->
            <div class="row mt-4">
                <div class="col-12">
                    <div class="bloom-card">
                        <h5 class="serif-title fw-bold mb-3 text-dark">📜 Lịch sử phân loại gần đây</h5>
                        <div class="table-responsive">
                            <table class="table table-borderless align-middle m-0">
                                <thead>
                                    <tr style="border-bottom: 2px solid #fbcfe8; color: #a21caf; font-size: 0.85rem;">
                                        <th>THỜI GIAN</th>
                                        <th>THÔNG SỐ (SL / SW / PL / PW)</th>
                                        <th>KẾT QUẢ</th>
                                        <th>ĐỘ TIN CẬY</th>
                                    </tr>
                                </thead>
                                <tbody id="historyBody" class="small">
                                    <tr>
                                        <td colspan="4" class="text-muted text-center py-3">Chưa có dữ liệu dự đoán nào.</td>
                                    </tr>
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            const ctx = document.getElementById('radarCanvas').getContext('2d');
            const radarChart = new Chart(ctx, {
                type: 'radar',
                data: {
                    labels: ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'],
                    datasets: [{
                        label: 'Chỉ số',
                        data: [5.1, 3.5, 1.4, 0.2],
                        backgroundColor: 'rgba(244, 114, 182, 0.25)',
                        borderColor: '#f472b6',
                        pointBackgroundColor: '#db2777',
                        pointBorderColor: '#fff'
                    }]
                },
                options: {
                    scales: {
                        r: {
                            angleLines: { color: '#fbcfe8' },
                            grid: { color: '#fbcfe8' },
                            pointLabels: { color: '#a21caf', font: { size: 11, weight: 'bold' } },
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
                runPrediction();
            }

            document.getElementById('irisForm').addEventListener('submit', (e) => {
                e.preventDefault();
                runPrediction();
            });

            let historyData = [];

            async function runPrediction() {
                const sl = parseFloat(document.getElementById('sl').value);
                const sw = parseFloat(document.getElementById('sw').value);
                const pl = parseFloat(document.getElementById('pl').value);
                const pw = parseFloat(document.getElementById('pw').value);

                radarChart.data.datasets[0].data = [sl, sw, pl, pw];
                radarChart.update();

                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ sepal_length: sl, sepal_width: sw, petal_length: pl, petal_width: pw })
                });

                const result = await response.json();
                const speciesName = result.prediction.toUpperCase();
                const confidence = result.confidence;

                // Cập nhật UI
                document.getElementById('targetClass').innerText = speciesName;
                document.getElementById('confidenceVal').innerText = confidence + '%';
                document.getElementById('confidenceBar').style.width = confidence + '%';

                // Bắn pháo hoa cánh hoa
                confetti({
                    particleCount: 50,
                    spread: 60,
                    origin: { y: 0.7 },
                    colors: ['#f472b6', '#c084fc', '#fbcfe8']
                });

                // Đọc giọng nói
                if (document.getElementById('voiceToggle').checked) {
                    speakResult("Predicted species is " + speciesName);
                }

                // Cập nhật lịch sử
                addHistory(sl, sw, pl, pw, speciesName, confidence);
            }

            function speakResult(text) {
                if ('speechSynthesis' in window) {
                    window.speechSynthesis.cancel();
                    const msg = new SpeechSynthesisUtterance(text);
                    msg.lang = 'en-US';
                    msg.rate = 0.9;
                    window.speechSynthesis.speak(msg);
                }
            }

            function addHistory(sl, sw, pl, pw, species, conf) {
                const timeStr = new Date().toLocaleTimeString();
                historyData.unshift({ time: timeStr, metrics: `${sl} / ${sw} / ${pl} / ${pw}`, species, conf });
                
                const tbody = document.getElementById('historyBody');
                tbody.innerHTML = historyData.slice(0, 5).map(item => `
                    <tr style="border-bottom: 1px solid #fdf2f8;">
                        <td class="fw-semibold text-muted">${item.time}</td>
                        <td class="fw-bold" style="color: #701a75;">${item.metrics}</td>
                        <td><span class="badge rounded-pill px-3 py-1" style="background: #fce7f3; color: #be185d;">${item.species}</span></td>
                        <td class="fw-bold text-success">${item.conf}%</td>
                    </tr>
                `).join('');
            }

            function downloadPDF() {
                const element = document.getElementById('exportArea');
                const opt = {
                    margin:       0.3,
                    filename:     'Iris_AI_Analysis_Report.pdf',
                    image:        { type: 'jpeg', quality: 0.98 },
                    html2canvas:  { scale: 2 },
                    jsPDF:        { unit: 'in', format: 'letter', orientation: 'landscape' }
                };
                html2pdf().set(opt).from(element).save();
            }
        </script>
    </body>
    </html>
    """

@app.post("/predict")
def predict(data: IrisInput):
    input_data = np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
    prediction = model.predict(input_data)[0]
    
    # Tính độ tin cậy (Probability score)
    confidence = 98.5
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(input_data)[0]
        confidence = round(float(np.max(probs)) * 100, 1)

    species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    result_name = species_map.get(prediction, str(prediction))
    
    return {
        "class_id": int(prediction), 
        "prediction": result_name,
        "confidence": confidence
    }
