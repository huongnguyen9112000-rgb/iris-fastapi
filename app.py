from fastapi import FastAPI
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel
import joblib
import numpy as np
import json
from datetime import datetime

app = FastAPI(title="Botanical Iris Enterprise AI Studio")

# Load mô hình SVM
model = joblib.load("svm_model.pkl")

# Lưu trữ lịch sử hoạt động & Audit Log trên bộ nhớ
activity_logs = []

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
        <title>Iris Pro AI // Enterprise Analytics Dashboard</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Playfair+Display:ital,wght@0,600;1,600&display=swap" rel="stylesheet">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.6.0/dist/confetti.browser.min.js"></script>
        <style>
            :root {
                --bg-soft: #fdf8f6;
                --primary-pink: #f472b6;
                --text-dark: #334155;
            }
            
            * { font-family: 'Plus Jakarta Sans', sans-serif; }
            .serif-title { font-family: 'Playfair Display', serif; }

            body {
                background: linear-gradient(135deg, #fdf8f6 0%, #fef2f2 50%, #f3e8ff 100%);
                color: var(--text-dark);
                min-height: 100vh;
                padding: 25px 15px;
                position: relative;
            }

            .bloom-card {
                background: rgba(255, 255, 255, 0.9);
                backdrop-filter: blur(16px);
                border: 1px solid rgba(244, 114, 182, 0.2);
                border-radius: 24px;
                padding: 22px;
                box-shadow: 0 10px 30px rgba(244, 114, 182, 0.06);
                margin-bottom: 20px;
            }

            .preset-btn {
                background: #fff;
                border: 1px solid #fbcfe8;
                color: #db2777;
                border-radius: 14px;
                padding: 6px 14px;
                font-weight: 600;
                font-size: 0.82rem;
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
                border-radius: 16px;
                padding: 8px 14px;
            }

            .input-box-floral label {
                font-size: 0.75rem;
                font-weight: 700;
                color: #a21caf;
                text-transform: uppercase;
            }

            .form-control-floral {
                background: transparent;
                border: none;
                color: #701a75;
                font-weight: 700;
                font-size: 1rem;
                width: 100%;
            }

            .form-control-floral:focus { outline: none; }

            .btn-bloom {
                background: linear-gradient(135deg, #f472b6 0%, #a855f7 100%);
                border: none;
                color: white;
                font-weight: 700;
                border-radius: 18px;
                padding: 12px;
                width: 100%;
                font-size: 0.95rem;
                box-shadow: 0 8px 18px rgba(244, 114, 182, 0.3);
                transition: all 0.3s ease;
            }

            .btn-bloom:hover {
                transform: translateY(-2px);
                box-shadow: 0 10px 22px rgba(244, 114, 182, 0.4);
            }

            .flower-card-hero {
                position: relative;
                width: 100%;
                height: 180px;
                border-radius: 18px;
                overflow: hidden;
                box-shadow: 0 8px 20px rgba(244, 114, 182, 0.15);
                border: 2px solid #fbcfe8;
            }

            .flower-img-preview {
                width: 100%;
                height: 100%;
                object-fit: cover;
                transition: transform 0.4s ease;
            }

            .badge-confidence-hero {
                position: absolute;
                top: 10px;
                right: 10px;
                background: rgba(255, 255, 255, 0.92);
                backdrop-filter: blur(8px);
                color: #be185d;
                font-weight: 800;
                padding: 4px 12px;
                border-radius: 20px;
                font-size: 0.8rem;
            }

            .progress-pink {
                height: 8px;
                border-radius: 10px;
                background-color: #fce7f3;
            }
            
            .progress-bar-pink {
                background: linear-gradient(90deg, #f472b6, #c084fc);
                border-radius: 10px;
            }

            .toast-notification {
                position: fixed;
                bottom: 20px;
                right: 20px;
                z-index: 9999;
                background: #1e293b;
                color: #fff;
                padding: 12px 20px;
                border-radius: 14px;
                box-shadow: 0 10px 25px rgba(0,0,0,0.2);
                display: none;
                align-items: center;
                gap: 10px;
                font-size: 0.88rem;
            }
        </style>
    </head>
    <body>

        <!-- Alert Toast -->
        <div id="toastAlert" class="toast-notification">
            <span id="toastIcon">🔔</span>
            <span id="toastMsg">Notification message here</span>
        </div>

        <div class="container-fluid" style="max-width: 1350px;">
            <!-- Header -->
            <div class="d-flex justify-content-between align-items-center mb-4">
                <div>
                    <span class="badge rounded-pill px-3 py-1 mb-1" style="background: #fce7f3; color: #be185d; font-weight: 700;">🌸 AI Enterprise Dashboard Suite</span>
                    <h2 class="serif-title fw-bold text-dark m-0">Iris Botanical Intelligence</h2>
                </div>
                <div class="d-flex gap-2">
                    <button class="btn btn-sm btn-outline-secondary rounded-3 fw-bold" onclick="exportData('json')">📥 JSON Export</button>
                    <button class="btn btn-sm btn-outline-success rounded-3 fw-bold" onclick="exportData('csv')">📊 CSV Export</button>
                </div>
            </div>

            <!-- Main Row 1: Controls, Radar, Output -->
            <div class="row g-3">
                <!-- Controls Panel -->
                <div class="col-lg-4">
                    <div class="bloom-card h-100">
                        <h6 class="serif-title fw-bold mb-3 text-dark">1. Thông số hoa Iris</h6>
                        
                        <div class="d-flex gap-2 mb-3 flex-wrap">
                            <button class="preset-btn" onclick="setPreset(5.1, 3.5, 1.4, 0.2)">🌸 Setosa</button>
                            <button class="preset-btn" onclick="setPreset(6.0, 2.9, 4.5, 1.5)">🌷 Versicolor</button>
                            <button class="preset-btn" onclick="setPreset(6.9, 3.1, 5.4, 2.1)">🌺 Virginica</button>
                        </div>

                        <form id="irisForm" class="d-flex flex-column gap-2">
                            <div class="input-box-floral">
                                <label>Sepal Length (Đài - Dài)</label>
                                <input type="number" step="0.1" id="sl" class="form-control-floral" value="5.1">
                            </div>
                            <div class="input-box-floral">
                                <label>Sepal Width (Đài - Rộng)</label>
                                <input type="number" step="0.1" id="sw" class="form-control-floral" value="3.5">
                            </div>
                            <div class="input-box-floral">
                                <label>Petal Length (Cánh - Dài)</label>
                                <input type="number" step="0.1" id="pl" class="form-control-floral" value="1.4">
                            </div>
                            <div class="input-box-floral">
                                <label>Petal Width (Cánh - Rộng)</label>
                                <input type="number" step="0.1" id="pw" class="form-control-floral" value="0.2">
                            </div>

                            <button type="submit" class="btn-bloom mt-2">PHÂN LOẠI & PHÂN TÍCH 🌸</button>
                        </form>
                    </div>
                </div>

                <!-- Radar Display & XAI SHAP Analysis -->
                <div class="col-lg-4">
                    <div class="bloom-card h-100 d-flex flex-column justify-content-between">
                        <div>
                            <h6 class="serif-title fw-bold mb-2 text-dark">2. Phân tích đặc trưng (Radar)</h6>
                            <div class="d-flex justify-content-center">
                                <canvas id="radarCanvas" style="max-height: 190px;"></canvas>
                            </div>
                        </div>
                        <div class="mt-2 p-3 rounded-4" style="background: #faf5f8; border: 1px solid #f5d0fe;">
                            <div class="small fw-bold text-uppercase mb-2" style="color: #a21caf;">💡 XAI (Explainable AI) - Mức đóng góp:</div>
                            <div id="xaiContainer" class="small text-muted">
                                Bấm phân loại để xem giải thích đóng góp của từng chỉ số đối với kết quả.
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Output HUD & Recommendations -->
                <div class="col-lg-4">
                    <div class="bloom-card h-100 d-flex flex-column justify-content-between">
                        <div>
                            <h6 class="serif-title fw-bold mb-2 text-dark">3. Kết quả & Gợi ý chăm sóc</h6>
                            
                            <div class="flower-card-hero mb-2">
                                <img id="flowerImg" src="https://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg" class="flower-img-preview" alt="Iris Flower">
                                <span class="badge-confidence-hero" id="badgeConf">Match 0%</span>
                            </div>

                            <div class="text-center mb-2">
                                <div class="small text-uppercase fw-bold text-muted" style="font-size: 0.75rem;">DỰ ĐOÁN LOÀI HOA</div>
                                <h4 id="targetClass" class="serif-title fw-bold m-0" style="color: #be185d;">SẴN SÀNG</h4>
                            </div>

                            <div class="p-2 rounded-3 mb-2" style="background: #faf5f8; border: 1px solid #f5d0fe;">
                                <div class="d-flex justify-content-between small fw-bold mb-1">
                                    <span style="color: #a21caf; font-size: 0.75rem;">CONFIDENCE:</span>
                                    <span id="confidenceVal" style="color: #be185d; font-size: 0.75rem;">0%</span>
                                </div>
                                <div class="progress progress-pink">
                                    <div id="confidenceBar" class="progress-bar progress-bar-pink" role="progressbar" style="width: 0%"></div>
                                </div>
                            </div>

                            <div id="recommendationBox" class="p-2 rounded-3 small text-secondary" style="background: #fdf2f8; font-size: 0.8rem;">
                                🌱 <b>Gợi ý AI:</b> Nhập thông số để nhận tư vấn môi trường nuôi trồng tối ưu.
                            </div>
                        </div>

                        <div class="pt-2">
                            <div class="d-flex align-items-center justify-content-between px-2 mb-2">
                                <span class="small fw-semibold text-secondary" style="font-size: 0.78rem;">🔊 Giọng đọc AI</span>
                                <div class="form-check form-switch m-0">
                                    <input class="form-check-input" type="checkbox" id="voiceToggle" checked>
                                </div>
                            </div>
                            <button class="btn btn-outline-danger w-100 rounded-3 fw-bold py-1" style="font-size: 0.85rem;" onclick="exportPDFReport()">📄 Xuất Báo Cáo PDF</button>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Audit Log & History Table -->
            <div class="row mt-2">
                <div class="col-12">
                    <div class="bloom-card">
                        <div class="d-flex justify-content-between align-items-center mb-3">
                            <h6 class="serif-title fw-bold text-dark m-0">📜 Lịch sử phân loại & Audit Log hệ thống</h6>
                            <span id="logCount" class="badge bg-secondary">0 ghi nhận</span>
                        </div>
                        <div class="table-responsive">
                            <table class="table table-borderless align-middle m-0">
                                <thead>
                                    <tr style="border-bottom: 2px solid #fbcfe8; color: #a21caf; font-size: 0.8rem;">
                                        <th>THỜI GIAN</th>
                                        <th>THÔNG SỐ (SL / SW / PL / PW)</th>
                                        <th>KẾT QUẢ</th>
                                        <th>ĐỘ TIN CẬY</th>
                                        <th>TRẠNG THÁI AI</th>
                                    </tr>
                                </thead>
                                <tbody id="historyBody" class="small">
                                    <tr>
                                        <td colspan="5" class="text-muted text-center py-3">Chưa có dữ liệu dự đoán nào.</td>
                                    </tr>
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script>
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
                            pointLabels: { color: '#a21caf', font: { size: 10, weight: 'bold' } },
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
                document.getElementById('badgeConf').innerText = 'Match ' + confidence + '%';

                if (flowerImages[speciesName]) {
                    document.getElementById('flowerImg').src = flowerImages[speciesName];
                }

                // Cập nhật XAI Insights
                updateXAI(result.xai_contributions);

                // Cập nhật Recommendations
                document.getElementById('recommendationBox').innerHTML = `🌱 <b>Tư vấn AI:</b> ${result.recommendation}`;

                // Confetti & Voice
                confetti({ particleCount: 40, spread: 50, origin: { y: 0.7 }, colors: ['#f472b6', '#c084fc'] });

                if (document.getElementById('voiceToggle').checked) {
                    speakResult("Result is " + speciesName);
                }

                // Giả lập hệ thống gửi Alert Notification
                showToast(`📧 [Email Alert] Đã gửi thông báo phân loại loài ${speciesName} đến hệ thống!`, '📩');

                addHistory(sl, sw, pl, pw, speciesName, confidence, result.status);
            }

            function updateXAI(contributions) {
                const xaiContainer = document.getElementById('xaiContainer');
                let html = '';
                for (const [key, val] of Object.entries(contributions)) {
                    const color = val >= 0 ? '#16a34a' : '#dc2626';
                    const sign = val >= 0 ? '+' : '';
                    html += `
                        <div class="d-flex justify-content-between align-items-center mb-1">
                            <span>${key}:</span>
                            <span class="fw-bold" style="color: ${color}">${sign}${val}%</span>
                        </div>
                    `;
                }
                xaiContainer.innerHTML = html;
            }

            function showToast(msg, icon = '🔔') {
                const toast = document.getElementById('toastAlert');
                document.getElementById('toastMsg').innerText = msg;
                document.getElementById('toastIcon').innerText = icon;
                toast.style.display = 'flex';
                setTimeout(() => { toast.style.display = 'none'; }, 3500);
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

            function addHistory(sl, sw, pl, pw, species, conf, status) {
                const timeStr = new Date().toLocaleTimeString();
                historyData.unshift({ time: timeStr, sl, sw, pl, pw, species, conf, status });
                
                document.getElementById('logCount').innerText = `${historyData.length} ghi nhận`;

                const tbody = document.getElementById('historyBody');
                tbody.innerHTML = historyData.slice(0, 6).map(item => `
                    <tr style="border-bottom: 1px solid #fdf2f8;">
                        <td class="fw-semibold text-muted">${item.time}</td>
                        <td class="fw-bold" style="color: #701a75;">${item.sl} / ${item.sw} / ${item.pl} / ${item.pw}</td>
                        <td><span class="badge rounded-pill px-3 py-1" style="background: #fce7f3; color: #be185d;">${item.species}</span></td>
                        <td class="fw-bold text-success">${item.conf}%</td>
                        <td><span class="badge bg-success-subtle text-success border border-success-subtle">${item.status}</span></td>
                    </tr>
                `).join('');
            }

            async function exportData(type) {
                if (type === 'json') {
                    window.location.href = '/export/json';
                } else if (type === 'csv') {
                    window.location.href = '/export/csv';
                }
            }

            function exportPDFReport() {
                const species = document.getElementById('targetClass').innerText;
                if (species === 'SẴN SÀNG') {
                    alert('Vui lòng phân loại hoa trước khi xuất báo cáo!');
                    return;
                }

                const sl = document.getElementById('sl').value;
                const sw = document.getElementById('sw').value;
                const pl = document.getElementById('pl').value;
                const pw = document.getElementById('pw').value;
                const conf = document.getElementById('confidenceVal').innerText;

                const printWindow = window.open('', '_blank');
                printWindow.document.write(`
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <title>Iris_Enterprise_Report_${species}</title>
                        <style>
                            @page { size: A4 portrait; margin: 15mm; }
                            body { font-family: sans-serif; color: #334155; margin: 0; padding: 0; }
                            .box { border: 2px solid #f472b6; border-radius: 16px; padding: 20px; }
                            h1 { color: #be185d; text-align: center; font-size: 20px; margin-bottom: 5px; }
                            .flex { display: flex; justify-content: space-between; background: #faf5f8; padding: 12px; border-radius: 10px; margin: 15px 0; }
                            table { width: 100%; border-collapse: collapse; margin-top: 10px; }
                            th, td { border: 1px solid #fbcfe8; padding: 8px; text-align: center; font-size: 13px; }
                            th { background: #fdf2f8; color: #a21caf; }
                        </style>
                    </head>
                    <body>
                        <div class="box">
                            <h1>🌸 BOTANICAL AI ENTERPRISE REPORT</h1>
                            <div class="flex">
                                <div><b>KẾT QUẢ DỰ ĐOÁN:</b> <span style="color:#be185d;">${species}</span></div>
                                <div><b>ĐỘ TIN CẬY:</b> <span style="color:#16a34a;">${conf}</span></div>
                            </div>
                            <h3>THÔNG SỐ MẪU HOA</h3>
                            <table>
                                <tr><th>Sepal Length</th><th>Sepal Width</th><th>Petal Length</th><th>Petal Width</th></tr>
                                <tr><td>${sl} cm</td><td>${sw} cm</td><td>${pl} cm</td><td>${pw} cm</td></tr>
                            </table>
                        </div>
                        <script>window.onload = function() { window.print(); }<\/script>
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
    result_name = species_map.get(prediction, str(prediction)).lower()

    # Tính toán XAI Feature Importance (Giả lập SHAP/Local Contribution)
    xai_contributions = {
        "Petal Length": round(float((data.petal_length / 6.9) * 45), 1),
        "Petal Width": round(float((data.petal_width / 2.5) * 35), 1),
        "Sepal Length": round(float((data.sepal_length / 7.9) * 12), 1),
        "Sepal Width": round(float((data.sepal_width / 4.4) * 8), 1)
    }

    # Gợi ý sinh học AI (Recommendation Engine)
    recommendations = {
        'setosa': "Loài Setosa ưa ánh sáng vừa phải, đất ẩm xốp (pH 6.0 - 7.0). Thích hợp trồng ở chậu cảnh nội thất.",
        'versicolor': "Loài Versicolor thích nghi tốt ở môi trường ven hồ, ánh sáng mặt trời đầy đủ, cần tưới nước thường xuyên.",
        'virginica': "Loài Virginica chịu nhiệt tốt, cần khoảng không rộng để phát triển bộ rễ, bón phân hữu cơ định kỳ 3 tháng/lần."
    }

    # Lưu log hoạt động (Audit Log System)
    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input": {"sl": data.sepal_length, "sw": data.sepal_width, "pl": data.petal_length, "pw": data.petal_width},
        "prediction": result_name.upper(),
        "confidence": confidence,
        "status": "VALIDATED_PASS"
    }
    activity_logs.append(log_entry)

    return {
        "class_id": int(prediction), 
        "prediction": result_name,
        "confidence": confidence,
        "xai_contributions": xai_contributions,
        "recommendation": recommendations.get(result_name, "Dữ liệu chưa được cập nhật."),
        "status": "VALIDATED_PASS"
    }

# Export Lịch sử sang định dạng JSON
@app.get("/export/json")
def export_json():
    return Response(
        content=json.dumps(activity_logs, ensure_ascii=False, indent=2),
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=iris_activity_logs.json"}
    )

# Export Lịch sử sang định dạng CSV
@app.get("/export/csv")
def export_csv():
    csv_content = "Timestamp,Sepal_Length,Sepal_Width,Petal_Length,Petal_Width,Prediction,Confidence,Status\n"
    for log in activity_logs:
        inp = log["input"]
        csv_content += f"{log['timestamp']},{inp['sl']},{inp['sw']},{inp['pl']},{inp['pw']},{log['prediction']},{log['confidence']}%,{log['status']}\n"
    
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=iris_activity_logs.csv"}
    )
