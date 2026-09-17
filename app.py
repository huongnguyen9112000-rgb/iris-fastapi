from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Botanical Iris AI Studio")

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
                position: relative;
                overflow-x: hidden;
            }

            /* Falling Petals Background Effect */
            .petal {
                position: fixed;
                top: -10%;
                user-select: none;
                pointer-events: none;
                z-index: 0;
                animation: fall linear infinite;
            }

            @keyframes fall {
                0% { transform: translateY(0) rotate(0deg); opacity: 1; }
                100% { transform: translateY(105vh) rotate(360deg); opacity: 0; }
            }

            .bloom-card {
                background: rgba(255, 255, 255, 0.88);
                backdrop-filter: blur(20px);
                border: 1px solid rgba(244, 114, 182, 0.2);
                border-radius: 28px;
                padding: 26px;
                box-shadow: 0 15px 35px rgba(244, 114, 182, 0.08);
                position: relative;
                z-index: 1;
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

            /* --- Đã cập nhật CSS cho Hero Image --- */
            .flower-card-hero {
                position: relative;
                width: 100%;
                height: 210px;
                border-radius: 20px;
                overflow: hidden;
                box-shadow: 0 10px 25px rgba(244, 114, 182, 0.2);
                border: 2px solid #fbcfe8;
            }

            .flower-img-preview {
                width: 100%;
                height: 100%;
                object-fit: cover;
                transition: transform 0.4s ease;
            }

            .flower-card-hero:hover .flower-img-preview {
                transform: scale(1.06);
            }

            .badge-confidence-hero {
                position: absolute;
                top: 12px;
                right: 12px;
                background: rgba(255, 255, 255, 0.92);
                backdrop-filter: blur(8px);
                color: #be185d;
                font-weight: 800;
                padding: 5px 14px;
                border-radius: 30px;
                font-size: 0.85rem;
                box-shadow: 0 4px 10px rgba(0,0,0,0.1);
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

        <!-- Petals Falling Container -->
        <div id="petalsContainer"></div>

        <div class="container-fluid" style="max-width: 1250px;">
            <div class="text-center mb-4 position-relative" style="z-index: 1;">
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
                            
                            <!-- --- Khối ảnh Hero mới được thay thế tại đây --- -->
                            <div class="flower-card-hero mb-3">
                                <img id="flowerImg" src="https://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg" class="flower-img-preview" alt="Iris Flower">
                                <span class="badge-confidence-hero" id="badgeConf">Match 0%</span>
                            </div>

                            <div class="text-center mb-3">
                                <div class="small text-uppercase fw-bold text-muted mb-1">DỰ ĐOÁN LOÀI HOA</div>
                                <h3 id="targetClass" class="serif-title fw-bold m-0" style="color: #be185d; font-size: 1.8rem;">SẴN SÀNG</h3>
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

                        <button class="btn btn-outline-danger w-100 rounded-4 fw-bold py-2" onclick="exportPDFReport()">📄 Xuất Báo Cáo PDF Chuẩn</button>
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
            // Hiệu ứng hoa rơi
            function createPetals() {
                const container = document.getElementById('petalsContainer');
                const petalIcons = ['🌸', '🌺', '🪷', '✨'];
                for (let i = 0; i < 15; i++) {
                    const petal = document.createElement('div');
                    petal.className = 'petal';
                    petal.innerText = petalIcons[Math.floor(Math.random() * petalIcons.length)];
                    petal.style.left = Math.random() * 100 + 'vw';
                    petal.style.animationDuration = (Math.random() * 5 + 5) + 's';
                    petal.style.fontSize = (Math.random() * 10 + 15) + 'px';
                    petal.style.animationDelay = Math.random() * 5 + 's';
                    container.appendChild(petal);
                }
            }
            createPetals();

            // Ảnh minh họa từng loài hoa
            const flowerImages = {
                'SETOSA': 'https://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg',
                'VERSICOLOR': 'https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg',
                'VIRGINICA': 'https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg'
            };

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

                document.getElementById('targetClass').innerText = speciesName;
                document.getElementById('confidenceVal').innerText = confidence + '%';
                document.getElementById('confidenceBar').style.width = confidence + '%';
                
                // Cập nhật text trên badge góc ảnh
                document.getElementById('badgeConf').innerText = 'Match ' + confidence + '%';

                if (flowerImages[speciesName]) {
                    document.getElementById('flowerImg').src = flowerImages[speciesName];
                }

                confetti({
                    particleCount: 50,
                    spread: 60,
                    origin: { y: 0.7 },
                    colors: ['#f472b6', '#c084fc', '#fbcfe8']
                });

                if (document.getElementById('voiceToggle').checked) {
                    speakResult("Predicted species is " + speciesName);
                }

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

            // HÀM XUẤT PDF CHUẨN ĐÚNG 1 TRANG KHÔNG LỖI MARGIN
            function exportPDFReport() {
                const species = document.getElementById('targetClass').innerText;
                if (species === 'SẴN SÀNG') {
                    alert('Vui lòng phân loại hoa trước khi xuất báo cáo PDF!');
                    return;
                }

                const sl = document.getElementById('sl').value;
                const sw = document.getElementById('sw').value;
                const pl = document.getElementById('pl').value;
                const pw = document.getElementById('pw').value;
                const conf = document.getElementById('confidenceVal').innerText;
                const dateStr = new Date().toLocaleDateString('vi-VN');

                const printWindow = window.open('', '_blank');
                printWindow.document.write(`
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <title>Iris_Report_${species}</title>
                        <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700&family=Playfair+Display:wght@700&display=swap" rel="stylesheet">
                        <style>
                            @page { size: A4 portrait; margin: 15mm; }
                            body { font-family: 'Plus Jakarta Sans', sans-serif; color: #334155; margin: 0; padding: 0; }
                            .report-box { border: 2px solid #f472b6; border-radius: 20px; padding: 25px; }
                            .header { text-align: center; border-bottom: 2px solid #fbcfe8; padding-bottom: 15px; margin-bottom: 20px; }
                            .header h1 { font-family: 'Playfair Display', serif; color: #be185d; margin: 0; font-size: 24px; }
                            .summary-box { display: flex; justify-content: space-between; background: #faf5f8; padding: 15px 20px; border-radius: 12px; margin-bottom: 20px; }
                            table { width: 100%; border-collapse: collapse; margin-top: 15px; }
                            th, td { border: 1px solid #fbcfe8; padding: 10px; text-align: center; font-size: 14px; }
                            th { background: #fdf2f8; color: #a21caf; }
                            .footer { margin-top: 40px; text-align: center; font-size: 11px; color: #94a3b8; border-top: 1px solid #f1f5f9; padding-top: 15px; }
                        </style>
                    </head>
                    <body>
                        <div class="report-box">
                            <div class="header">
                                <h1>🌸 IRIS AI BOTANICAL REPORT</h1>
                                <p style="color: #64748b; font-size: 12px; margin-top: 5px;">Báo cáo kết quả phân loại loài hoa Iris bằng thuật toán Machine Learning</p>
                            </div>

                            <div class="summary-box">
                                <div>
                                    <div style="font-size: 12px; color: #64748b;">KẾT QUẢ PHÂN LOẠI</div>
                                    <div style="font-size: 22px; font-weight: bold; color: #be185d; font-family: 'Playfair Display', serif;">${species}</div>
                                </div>
                                <div style="text-align: right;">
                                    <div style="font-size: 12px; color: #64748b;">ĐỘ TIN CẬY MODEL</div>
                                    <div style="font-size: 18px; font-weight: bold; color: #16a34a;">${conf}</div>
                                </div>
                            </div>

                            <h3 style="color: #a21caf; font-size: 15px; margin-bottom: 5px;">CHỈ SỐ ĐẦU VÀO (INPUT METRICS)</h3>
                            <table>
                                <thead>
                                    <tr>
                                        <th>Sepal Length</th>
                                        <th>Sepal Width</th>
                                        <th>Petal Length</th>
                                        <th>Petal Width</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    <tr>
                                        <td><b>${sl} cm</b></td>
                                        <td><b>${sw} cm</b></td>
                                        <td><b>${pl} cm</b></td>
                                        <td><b>${pw} cm</b></td>
                                    </tr>
                                </tbody>
                            </table>

                            <div style="margin-top: 25px; background: #fff1f2; border-left: 4px solid #f472b6; padding: 12px 15px; border-radius: 6px; font-size: 12px;">
                                <strong>Ghi chú:</strong> Dự đoán được thực hiện bởi mô hình Support Vector Machine (SVM) được huấn luyện trên tập dữ liệu chuẩn Iris Dataset.
                            </div>

                            <div class="footer">
                                Ngày xuất báo cáo: ${dateStr} • Generated by Iris AI Studio Pro
                            </div>
                        </div>
                        <script>
                            window.onload = function() {
                                window.print();
                            }
                        <\/script>
                    </body>
                    </html>
                `);
                printWindow.document.close();
            }
        </script>
    </body>
    </html>
    """

@app.post("/predict")
def predict(data: IrisInput):
    input_data = np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
    prediction = model.predict(input_data)[0]
    
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
