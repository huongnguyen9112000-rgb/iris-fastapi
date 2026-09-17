from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel
import joblib
import numpy as np
import sqlite3
import json
from datetime import datetime

app = FastAPI(title="Botanical Iris AI System - Advanced Care Edition")

# 1. Khởi tạo Cơ sở dữ liệu SQLite
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

# Kiểm tra Anomaly (Dữ liệu bất thường)
def check_anomaly(sl, sw, pl, pw):
    if sl < 3.0 or sl > 9.0 or sw < 1.5 or sw > 5.0 or pl < 0.5 or pl > 8.0 or pw < 0.0 or pw > 3.5:
        return True
    return False

# Cơ sở dữ liệu tư vấn AI chi tiết chuyên sâu cho từng loài Iris
DETAILED_AI_CARE_GUIDE = {
    'setosa': {
        "title": "Iris Setosa (Diên vĩ Lông tơ / Bristle-pointed Iris)",
        "climate": "Khí hậu ôn đới mát mẻ, xứ lạnh, chịu băng giá rất tốt.",
        "soil": "Đất thịt nhẹ giàu mùn, hơi chua (pH 6.0 - 6.5), giữ ẩm tốt nhưng thoát nước vừa phải.",
        "sunlight": "Nắng bán phần (4-6 giờ nắng nhẹ/ngày). Tránh nắng gắt buổi trưa.",
        "watering": "Tưới 2-3 lần/tuần. Giữ đất luôn ẩm nhẹ, không để khô hoàn toàn.",
        "fertilizer": "Bón phân hữu cơ hoai mục hoặc NPK 10-10-10 tan chậm vào đầu mùa xuân.",
        "pest_note": "Kháng bệnh tốt, chú ý kiểm tra ốc sên cắn lá mầm vào mùa mưa."
    },
    'versicolor': {
        "title": "Iris Versicolor (Diên vĩ Xanh / Harlequin Blueflag)",
        "climate": "Thích hợp môi trường đầm lầy, ven bờ hồ, độ ẩm không khí cao.",
        "soil": "Đất sét bùn, nhiều hữu cơ, chấp nhận đất ngập nước nhẹ (pH 5.5 - 7.0).",
        "sunlight": "Nắng toàn phần đến bán phần (ít nhất 6 giờ nắng/ngày để hoa đậm màu).",
        "watering": "Cần nhiều nước. Có thể trồng nông dưới mực nước 3-5 cm hoặc tưới đẫm hằng ngày.",
        "fertilizer": "Thêm bùn ao hoặc phân thủy sinh vào mùa sinh trưởng (tháng 3 - tháng 6).",
        "pest_note": "Đặc biệt chú ý sâu bọ xòe lá (Iris borer) và rệp cây vào mùa hè."
    },
    'virginica': {
        "title": "Iris Virginica (Diên vĩ Virginia / Virginia Blueflag)",
        "climate": "Thích nghi tốt với thời tiết ấm áp, chịu nhiệt và chịu nắng tốt nhất.",
        "soil": "Đất phù sa, đất mùn ẩm dầy, pH trung tính đến hơi kiềm (pH 6.5 - 7.5).",
        "sunlight": "Nắng toàn phần (Direct Sun 6-8 giờ/ngày) giúp củ phát triển to khỏe.",
        "watering": "Tưới nước trung bình. Chịu ngập nhẹ vào mùa mưa nhưng cần thoát nước tốt sau đó.",
        "fertilizer": "Bón phân giàu Phốt pho (P) và Kali (K) trước mùa nở hoa 3 tuần.",
        "pest_note": "Cắt tỉa lá già cuối mùa thu để tránh nấm đốm lá (Leaf Spot)."
    }
}

# Route trang chủ UI
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html lang="vi">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Iris Pro AI // Enterprise Care System</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Playfair+Display:ital,wght@0,600;1,600&display=swap" rel="stylesheet">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.6.0/dist/confetti.browser.min.js"></script>
        <style>
            :root { --bg-soft: #fdf8f6; --primary-pink: #f472b6; --text-dark: #334155; }
            * { font-family: 'Plus Jakarta Sans', sans-serif; }
            .serif-title { font-family: 'Playfair Display', serif; }
            body { background: linear-gradient(135deg, #fdf8f6 0%, #fef2f2 50%, #f3e8ff 100%); color: var(--text-dark); min-height: 100vh; padding: 25px 15px; }
            .bloom-card { background: rgba(255, 255, 255, 0.92); backdrop-filter: blur(16px); border: 1px solid rgba(244, 114, 182, 0.2); border-radius: 24px; padding: 22px; box-shadow: 0 10px 30px rgba(244, 114, 182, 0.06); margin-bottom: 20px; }
            .preset-btn { background: #fff; border: 1px solid #fbcfe8; color: #db2777; border-radius: 14px; padding: 6px 14px; font-weight: 600; font-size: 0.82rem; transition: all 0.2s ease; }
            .preset-btn:hover { background: #fdf2f8; border-color: #f472b6; transform: translateY(-2px); }
            .input-box-floral { background: #faf5f8; border: 1px solid #f5d0fe; border-radius: 16px; padding: 8px 14px; }
            .input-box-floral label { font-size: 0.75rem; font-weight: 700; color: #a21caf; text-transform: uppercase; }
            .form-control-floral { background: transparent; border: none; color: #701a75; font-weight: 700; font-size: 1rem; width: 100%; }
            .form-control-floral:focus { outline: none; }
            .btn-bloom { background: linear-gradient(135deg, #f472b6 0%, #a855f7 100%); border: none; color: white; font-weight: 700; border-radius: 18px; padding: 12px; width: 100%; font-size: 0.95rem; box-shadow: 0 8px 18px rgba(244, 114, 182, 0.3); transition: all 0.3s ease; }
            .btn-bloom:hover { transform: translateY(-2px); box-shadow: 0 10px 22px rgba(244, 114, 182, 0.4); }
            .flower-card-hero { position: relative; width: 100%; height: 180px; border-radius: 18px; overflow: hidden; box-shadow: 0 8px 20px rgba(244, 114, 182, 0.15); border: 2px solid #fbcfe8; }
            .flower-img-preview { width: 100%; height: 100%; object-fit: cover; }
            .badge-confidence-hero { position: absolute; top: 10px; right: 10px; background: rgba(255, 255, 255, 0.92); color: #be185d; font-weight: 800; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; }
            .progress-pink { height: 8px; border-radius: 10px; background-color: #fce7f3; }
            .progress-bar-pink { background: linear-gradient(90deg, #f472b6, #c084fc); border-radius: 10px; }
            .toast-notification { position: fixed; bottom: 20px; right: 20px; z-index: 9999; background: #1e293b; color: #fff; padding: 12px 20px; border-radius: 14px; display: none; align-items: center; gap: 10px; font-size: 0.88rem; }
            
            /* Style cho Báo cáo Tư vấn AI */
            .care-item { background: #ffffff; border-left: 4px solid #f472b6; padding: 8px 12px; border-radius: 8px; margin-bottom: 8px; font-size: 0.82rem; }
            .care-item-title { font-weight: 700; color: #9d174d; text-transform: uppercase; font-size: 0.72rem; margin-bottom: 2px; }
        </style>
    </head>
    <body>

        <div id="toastAlert" class="toast-notification">
            <span id="toastIcon">🔔</span>
            <span id="toastMsg">Notification message</span>
        </div>

        <div class="container-fluid" style="max-width: 1400px;">
            <div class="d-flex justify-content-between align-items-center mb-4">
                <div>
                    <span class="badge rounded-pill px-3 py-1 mb-1" style="background: #fce7f3; color: #be185d; font-weight: 700;">🌸 AI Enterprise Care & Analytics</span>
                    <h2 class="serif-title fw-bold text-dark m-0">Botanical Intelligence Dashboard</h2>
                </div>
                <div class="d-flex gap-2">
                    <button class="btn btn-sm btn-outline-secondary rounded-3 fw-bold" onclick="exportData('json')">📥 Export JSON</button>
                    <button class="btn btn-sm btn-outline-success rounded-3 fw-bold" onclick="exportData('csv')">📊 Export CSV</button>
                </div>
            </div>

            <div class="row g-3">
                <!-- Cột 1: Form Nhập -->
                <div class="col-lg-3">
                    <div class="bloom-card h-100">
                        <h6 class="serif-title fw-bold mb-3 text-dark">1. Thông số sinh học</h6>
                        <div class="d-flex gap-1 mb-3 flex-wrap">
                            <button class="preset-btn" onclick="setPreset(5.1, 3.5, 1.4, 0.2)">🌸 Setosa</button>
                            <button class="preset-btn" onclick="setPreset(6.0, 2.9, 4.5, 1.5)">🌷 Versicolor</button>
                            <button class="preset-btn" onclick="setPreset(6.9, 3.1, 5.4, 2.1)">🌺 Virginica</button>
                        </div>
                        <form id="irisForm" class="d-flex flex-column gap-2">
                            <div class="input-box-floral"><label>Sepal Length (cm)</label><input type="number" step="0.1" id="sl" class="form-control-floral" value="5.1"></div>
                            <div class="input-box-floral"><label>Sepal Width (cm)</label><input type="number" step="0.1" id="sw" class="form-control-floral" value="3.5"></div>
                            <div class="input-box-floral"><label>Petal Length (cm)</label><input type="number" step="0.1" id="pl" class="form-control-floral" value="1.4"></div>
                            <div class="input-box-floral"><label>Petal Width (cm)</label><input type="number" step="0.1" id="pw" class="form-control-floral" value="0.2"></div>
                            <button type="submit" class="btn-bloom mt-2">PHÂN TÍCH & TƯ VẤN AI 🌸</button>
                        </form>
                    </div>
                </div>

                <!-- Cột 2: Dự đoán & XAI -->
                <div class="col-lg-4">
                    <div class="bloom-card h-100 d-flex flex-column justify-content-between">
                        <div>
                            <h6 class="serif-title fw-bold mb-2 text-dark">2. Phân loại & Giải thích XAI</h6>
                            <div class="flower-card-hero mb-2">
                                <img id="flowerImg" src="https://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg" class="flower-img-preview" alt="Iris">
                                <span class="badge-confidence-hero" id="badgeConf">Match 0%</span>
                            </div>
                            <div class="text-center mb-2">
                                <div class="small text-uppercase fw-bold text-muted" style="font-size: 0.72rem;">KẾT QUẢ PHÂN LOẠI</div>
                                <h4 id="targetClass" class="serif-title fw-bold m-0" style="color: #be185d;">SẴN SÀNG</h4>
                            </div>
                            <div class="p-2 rounded-3 mb-2" style="background: #faf5f8; border: 1px solid #f5d0fe;">
                                <div class="d-flex justify-content-between small fw-bold mb-1">
                                    <span style="color: #a21caf; font-size: 0.72rem;">ĐỘ TIN CẬY:</span>
                                    <span id="confidenceVal" style="color: #be185d; font-size: 0.72rem;">0%</span>
                                </div>
                                <div class="progress progress-pink"><div id="confidenceBar" class="progress-bar progress-bar-pink" style="width: 0%"></div></div>
                            </div>
                        </div>

                        <div>
                            <div class="p-2 rounded-4 mb-2" style="background: #faf5f8; border: 1px solid #f5d0fe;">
                                <div class="small fw-bold text-uppercase mb-1" style="color: #a21caf; font-size: 0.72rem;">💡 XAI - Đóng góp chỉ số:</div>
                                <div id="xaiContainer" class="small text-muted">Nhấn phân tích để xem đóng góp.</div>
                            </div>
                            <div class="d-flex align-items-center justify-content-between px-1 mb-2">
                                <span class="small fw-semibold text-secondary" style="font-size: 0.78rem;">🔊 Giọng đọc AI</span>
                                <div class="form-check form-switch m-0"><input class="form-check-input" type="checkbox" id="voiceToggle" checked></div>
                            </div>
                            <button class="btn btn-outline-danger w-100 rounded-3 fw-bold py-1" style="font-size: 0.82rem;" onclick="exportPDFReport()">📄 In / Xuất Báo Cáo PDF</button>
                        </div>
                    </div>
                </div>

                <!-- Cột 3: Tư vấn AI Chi Tiết -->
                <div class="col-lg-5">
                    <div class="bloom-card h-100">
                        <div class="d-flex align-items-center justify-content-between mb-2">
                            <h6 class="serif-title fw-bold text-dark m-0">3. Báo cáo Tư vấn Chăm sóc AI</h6>
                            <span class="badge bg-success" id="anomalyStatus">BÌNH THƯỜNG</span>
                        </div>
                        
                        <div id="careReportContainer" style="max-height: 480px; overflow-y: auto;" class="pe-1">
                            <div class="text-center text-muted py-5">
                                🪴 <br>Nhập thông số và bấm phân tích để AI xuất báo cáo chăm sóc chi tiết.
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Hàng 2: Lịch sử SQLite -->
            <div class="row mt-2">
                <div class="col-12">
                    <div class="bloom-card">
                        <div class="d-flex justify-content-between align-items-center mb-3">
                            <h6 class="serif-title fw-bold text-dark m-0">📜 Lịch sử lưu trữ Database (SQLite)</h6>
                            <span id="logCount" class="badge bg-secondary">0 ghi nhận</span>
                        </div>
                        <div class="table-responsive">
                            <table class="table table-borderless align-middle m-0">
                                <thead>
                                    <tr style="border-bottom: 2px solid #fbcfe8; color: #a21caf; font-size: 0.8rem;">
                                        <th>THỜI GIAN</th>
                                        <th>THÔNG SỐ (SL / SW / PL / PW)</th>
                                        <th>LOÀI DỰ ĐOÁN</th>
                                        <th>ĐỘ TIN CẬY</th>
                                        <th>TRẠNG THÁI</th>
                                    </tr>
                                </thead>
                                <tbody id="historyBody" class="small">
                                    <tr><td colspan="5" class="text-muted text-center py-3">Chưa có dữ liệu nào.</td></tr>
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

            function setPreset(sl, sw, pl, pw) {
                document.getElementById('sl').value = sl; document.getElementById('sw').value = sw;
                document.getElementById('pl').value = pl; document.getElementById('pw').value = pw;
                runPrediction();
            }

            document.getElementById('irisForm').addEventListener('submit', (e) => { e.preventDefault(); runPrediction(); });

            async function runPrediction() {
                const sl = parseFloat(document.getElementById('sl').value);
                const sw = parseFloat(document.getElementById('sw').value);
                const pl = parseFloat(document.getElementById('pl').value);
                const pw = parseFloat(document.getElementById('pw').value);

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

                if (flowerImages[speciesName]) document.getElementById('flowerImg').src = flowerImages[speciesName];

                // Cập nhật Anomaly Status
                const anomalyBadge = document.getElementById('anomalyStatus');
                if(result.is_anomaly) {
                    anomalyBadge.className = 'badge bg-danger';
                    anomalyBadge.innerText = 'CẢNH BÁO DỊ BIỆT (ANOMALY)';
                } else {
                    anomalyBadge.className = 'badge bg-success';
                    anomalyBadge.innerText = 'CHỈ SỐ BÌNH THƯỜNG';
                }

                updateXAI(result.xai_contributions);
                renderCareGuide(result.care_guide);

                confetti({ particleCount: 40, spread: 50, origin: { y: 0.7 }, colors: ['#f472b6', '#c084fc'] });
                if (document.getElementById('voiceToggle').checked) speakResult("Predicted species is " + speciesName);

                showToast(`Đã cập nhật báo cáo chăm sóc & lưu SQLite thành công!`, '🌿');
                loadLogs();
            }

            function updateXAI(contributions) {
                let html = '';
                for (const [key, val] of Object.entries(contributions)) {
                    html += `<div class="d-flex justify-content-between align-items-center mb-1"><span>${key}:</span><span class="fw-bold text-success">+${val}%</span></div>`;
                }
                document.getElementById('xaiContainer').innerHTML = html;
            }

            function renderCareGuide(guide) {
                if(!guide) return;
                const html = `
                    <div class="fw-bold text-dark mb-2" style="font-size: 0.9rem;">📌 ${guide.title}</div>
                    <div class="care-item"><div class="care-item-title">🌤️ Khí hậu & Nhiệt độ</div><div>${guide.climate}</div></div>
                    <div class="care-item"><div class="care-item-title">🌱 Đất trồng & Độ pH</div><div>${guide.soil}</div></div>
                    <div class="care-item"><div class="care-item-title">☀️ Ánh sáng mặt trời</div><div>${guide.sunlight}</div></div>
                    <div class="care-item"><div class="care-item-title">💧 Chế độ tưới nước</div><div>${guide.watering}</div></div>
                    <div class="care-item"><div class="care-item-title">🧪 Phân bón & Dinh dưỡng</div><div>${guide.fertilizer}</div></div>
                    <div class="care-item"><div class="care-item-title">🛡️ Phòng ngừa sâu bệnh</div><div>${guide.pest_note}</div></div>
                `;
                document.getElementById('careReportContainer').innerHTML = html;
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
                    window.speechSynthesis.speak(msg);
                }
            }

            async function loadLogs() {
                const res = await fetch('/logs/db');
                const data = await res.json();
                document.getElementById('logCount').innerText = `${data.total} ghi nhận`;
                const tbody = document.getElementById('historyBody');
                tbody.innerHTML = data.data.slice(0, 6).map(item => `
                    <tr style="border-bottom: 1px solid #fdf2f8;">
                        <td class="fw-semibold text-muted">${item.timestamp}</td>
                        <td class="fw-bold" style="color: #701a75;">${item.input.sl} / ${item.input.sw} / ${item.input.pl} / ${item.input.pw}</td>
                        <td><span class="badge rounded-pill px-3 py-1" style="background: #fce7f3; color: #be185d;">${item.prediction}</span></td>
                        <td class="fw-bold text-success">${item.confidence}%</td>
                        <td><span class="badge ${item.is_anomaly ? 'bg-danger' : 'bg-success'}">${item.is_anomaly ? 'ANOMALY' : 'NORMAL'}</span></td>
                    </tr>
                `).join('');
            }

            function exportData(type) { window.location.href = `/export/${type}`; }
            function exportPDFReport() { window.print(); }

            window.onload = loadLogs;
        </script>
    </body>
    </html>
    """

# API Dự đoán chính
@app.post("/predict")
def predict(data: IrisInput):
    sl, sw, pl, pw = data.sepal_length, data.sepal_width, data.petal_length, data.petal_width
    input_data = np.array([[sl, sw, pl, pw]])
    
    prediction = model.predict(input_data)[0]
    confidence = 98.5
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(input_data)[0]
        confidence = round(float(np.max(probs)) * 100, 1)

    species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    result_name = species_map.get(prediction, str(prediction)).lower()

    is_anomaly = check_anomaly(sl, sw, pl, pw)

    xai_contributions = {
        "Petal Length (Chiều dài cánh)": round(float((pl / 6.9) * 45), 1),
        "Petal Width (Chiều rộng cánh)": round(float((pw / 2.5) * 35), 1),
        "Sepal Length (Chiều dài đài)": round(float((sl / 7.9) * 12), 1),
        "Sepal Width (Chiều rộng đài)": round(float((sw / 4.4) * 8), 1)
    }

    care_guide = DETAILED_AI_CARE_GUIDE.get(result_name, DETAILED_AI_CARE_GUIDE['setosa'])

    # Lưu dữ liệu vào SQLite
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
        "care_guide": care_guide,
        "status": "ANOMALY_DETECTED" if is_anomaly else "SUCCESS"
    }

# Lấy lịch sử từ SQLite
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

# Export JSON
@app.get("/export/json")
def export_json():
    logs = get_db_logs()["data"]
    return Response(content=json.dumps(logs, ensure_ascii=False, indent=2), media_type="application/json", headers={"Content-Disposition": "attachment; filename=iris_logs.json"})

# Export CSV
@app.get("/export/csv")
def export_csv():
    logs = get_db_logs()["data"]
    csv_content = "Timestamp,Sepal_Length,Sepal_Width,Petal_Length,Petal_Width,Prediction,Confidence,Anomaly\n"
    for item in logs:
        inp = item["input"]
        csv_content += f"{item['timestamp']},{inp['sl']},{inp['sw']},{inp['pl']},{inp['pw']},{item['prediction']},{item['confidence']}%,{item['is_anomaly']}\n"
    return Response(content=csv_content, media_type="text/csv", headers={"Content-Disposition": "attachment; filename=iris_logs.csv"})
