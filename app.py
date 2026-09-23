import os
import sqlite3
import csv
import io
from datetime import datetime

import numpy as np
import joblib
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

from google import genai

app = FastAPI(title="Iris Botanical Enterprise Suite", version="9.0")

# --- 1. CƠ SỞ DỮ LIỆU SQLITE ---
DB_FILE = "iris_system.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
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
    ''')
    conn.commit()
    conn.close()

init_db()

# --- 2. MÔ HÌNH MACHINE LEARNING ---
MODEL_FILE = "iris_model.pkl"

def get_or_create_model():
    if not os.path.exists(MODEL_FILE):
        iris = load_iris()
        X, y = iris.data, iris.target
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        joblib.dump(model, MODEL_FILE)
    return joblib.load(MODEL_FILE)

model = get_or_create_model()
target_names = ['Setosa', 'Versicolor', 'Virginica']

# --- 3. GOOGLE GEMINI AI ---
try:
    gemini_client = genai.Client()
except Exception:
    gemini_client = None

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# --- 4. API ENDPOINTS ---
@app.post("/predict")
def predict_iris(data: IrisInput):
    features = np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
    
    probs = model.predict_proba(features)[0]
    pred_idx = np.argmax(probs)
    prediction = target_names[pred_idx]
    confidence = float(probs[pred_idx])
    
    probabilities = [round(float(p) * 100, 1) for p in probs]
    is_anomaly = 1 if (data.petal_length < 1.0 or data.petal_length > 7.5) else 0

    ai_report = "Hệ thống AI chưa được cấu hình khóa API."
    if gemini_client:
        try:
            prompt = f"Phân tích loài hoa Iris {prediction} với các thông số: Sepal Length={data.sepal_length}, Sepal Width={data.sepal_width}, Petal Length={data.petal_length}, Petal Width={data.petal_width}. Hãy đưa ra tư vấn chuyên sâu, chuyên nghiệp về điều kiện sinh trưởng, loại đất và cách chăm sóc tối ưu bằng tiếng Việt."
            response = gemini_client.models.generate_content(
                model='gemini-2.0-flash',
                contents=prompt
            )
            ai_report = response.text
        except Exception as e:
            ai_report = f"Lỗi gọi Gemini AI: {str(e)}"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO predictions (timestamp, sepal_length, sepal_width, petal_length, petal_width, prediction, confidence, is_anomaly)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (timestamp, data.sepal_length, data.sepal_width, data.petal_length, data.petal_width, prediction, confidence, is_anomaly))
    conn.commit()
    conn.close()

    return {
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": probabilities,
        "is_anomaly": bool(is_anomaly),
        "ai_report": ai_report
    }

@app.get("/logs")
def get_logs():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT id, timestamp, sepal_length, sepal_width, petal_length, petal_width, prediction, confidence, is_anomaly FROM predictions ORDER BY id DESC LIMIT 10")
    rows = cursor.fetchall()
    conn.close()
    logs = [{
        "id": r[0], "timestamp": r[1], 
        "sepal_length": r[2], "sepal_width": r[3], "petal_length": r[4], "petal_width": r[5],
        "prediction": r[6], "confidence": r[7], "is_anomaly": bool(r[8])
    } for r in rows]
    return logs

@app.get("/export/{format_type}")
def export_data(format_type: str):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT id, timestamp, sepal_length, sepal_width, petal_length, petal_width, prediction, confidence, is_anomaly FROM predictions")
    rows = cursor.fetchall()
    conn.close()

    if format_type == "json":
        data = [{"id": r[0], "timestamp": r[1], "sepal_length": r[2], "sepal_width": r[3], "petal_length": r[4], "petal_width": r[5], "prediction": r[6], "confidence": r[7], "is_anomaly": bool(r[8])} for r in rows]
        return JSONResponse(content=data)
    elif format_type == "csv":
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["ID", "Timestamp", "Sepal Length", "Sepal Width", "Petal Length", "Petal Width", "Prediction", "Confidence", "Is Anomaly"])
        writer.writerows(rows)
        response = Response(output.getvalue(), media_type="text/csv")
        response.headers["Content-Disposition"] = "attachment; filename=iris_export.csv"
        return response
    else:
        raise HTTPException(status_code=400, detail="Không hỗ trợ.")

# --- 5. GIAO DIỆN WEB ENTERPRISE DASHBOARD THỰC THỤ ---
@app.get("/", response_class=HTMLResponse)
def get_dashboard():
    return """
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Iris Botanical Enterprise Suite</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
        body { font-family: 'Plus Jakarta Sans', sans-serif; background: #f8fafc; min-height: 100vh; }
        .glass-card { background: #ffffff; border: 1px solid #e2e8f0; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03); }
    </style>
</head>
<body class="flex flex-col min-h-screen text-slate-800">

    <!-- NAVBAR CHÍNH -->
    <header class="bg-white border-b border-slate-200 sticky top-0 z-40 px-6 py-3.5 shadow-sm">
        <div class="max-w-7xl mx-auto flex items-center justify-between">
            <div class="flex items-center space-x-3">
                <div class="w-10 h-10 rounded-xl bg-gradient-to-tr from-pink-500 to-rose-500 text-white flex items-center justify-center shadow-md shadow-pink-500/20">
                    <i class="fa-solid fa-seedling text-sm"></i>
                </div>
                <div>
                    <h1 class="font-extrabold text-slate-900 text-sm tracking-tight">Iris Botanical Intelligence</h1>
                    <span class="text-[10px] text-emerald-600 font-bold bg-emerald-50 px-2 py-0.5 rounded-full inline-flex items-center gap-1">
                        <span class="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></span> System Operational
                    </span>
                </div>
            </div>
            
            <div class="flex items-center gap-3">
                <div class="relative">
                    <button onclick="toggleDropdown()" class="px-4 py-2 bg-slate-900 hover:bg-slate-800 text-white font-bold rounded-xl text-xs transition flex items-center gap-2 shadow-sm">
                        <i class="fa-solid fa-download"></i> Xuất dữ liệu <i class="fa-solid fa-chevron-down text-[10px]"></i>
                    </button>
                    <div id="exportDropdown" class="hidden absolute right-0 mt-2 w-48 bg-white rounded-xl shadow-xl border border-slate-100 py-2 z-50">
                        <a href="/export/json" class="block px-4 py-2.5 text-xs font-bold text-slate-700 hover:bg-pink-50 hover:text-pink-600"><i class="fa-solid fa-file-code mr-2 text-pink-500"></i> Tải JSON</a>
                        <a href="/export/csv" class="block px-4 py-2.5 text-xs font-bold text-slate-700 hover:bg-pink-50 hover:text-pink-600"><i class="fa-solid fa-file-csv mr-2 text-emerald-500"></i> Tải CSV</a>
                    </div>
                </div>
            </div>
        </div>
    </header>

    <!-- NỘI DUNG CHÍNH (GRID LAYOUT CHUYÊN NGHIỆP) -->
    <main class="max-w-7xl mx-auto px-6 py-6 flex-grow grid grid-cols-1 lg:grid-cols-12 gap-6 w-full items-start">
        
        <!-- CỘT TRÁI: FORM NHẬP THÔNG SỐ & MẪU NHANH (Col-4) -->
        <section class="lg:col-span-4 space-y-6">
            <div class="glass-card p-5 rounded-2xl">
                <h3 class="text-xs font-extrabold text-slate-900 uppercase tracking-wider mb-3 flex items-center gap-2">
                    <i class="fa-solid fa-sliders text-pink-500"></i> Cấu hình thông số & Mẫu
                </h3>
                
                <div class="grid grid-cols-3 gap-1.5 mb-4">
                    <button type="button" onclick="loadSample('setosa')" class="text-[11px] bg-pink-50 hover:bg-pink-100 text-pink-700 font-extrabold py-2 rounded-xl transition border border-pink-200">🌸 Setosa</button>
                    <button type="button" onclick="loadSample('versicolor')" class="text-[11px] bg-purple-50 hover:bg-purple-100 text-purple-700 font-extrabold py-2 rounded-xl transition border border-purple-200">🌷 Versicolor</button>
                    <button type="button" onclick="loadSample('virginica')" class="text-[11px] bg-rose-50 hover:bg-rose-100 text-rose-700 font-extrabold py-2 rounded-xl transition border border-rose-200">🌺 Virginica</button>
                </div>

                <form id="predictionForm" class="space-y-3">
                    <div>
                        <label class="block text-[10px] font-bold text-slate-500 uppercase mb-1">Sepal Length (cm)</label>
                        <input type="number" step="0.1" id="sepal_length" value="5.1" class="w-full bg-slate-50 border border-slate-200 rounded-xl px-3 py-2 text-xs font-extrabold text-slate-800 focus:outline-none focus:ring-2 focus:ring-pink-500">
                    </div>
                    <div>
                        <label class="block text-[10px] font-bold text-slate-500 uppercase mb-1">Sepal Width (cm)</label>
                        <input type="number" step="0.1" id="sepal_width" value="3.5" class="w-full bg-slate-50 border border-slate-200 rounded-xl px-3 py-2 text-xs font-extrabold text-slate-800 focus:outline-none focus:ring-2 focus:ring-pink-500">
                    </div>
                    <div>
                        <label class="block text-[10px] font-bold text-slate-500 uppercase mb-1">Petal Length (cm)</label>
                        <input type="number" step="0.1" id="petal_length" value="1.4" class="w-full bg-slate-50 border border-slate-200 rounded-xl px-3 py-2 text-xs font-extrabold text-slate-800 focus:outline-none focus:ring-2 focus:ring-pink-500">
                    </div>
                    <div>
                        <label class="block text-[10px] font-bold text-slate-500 uppercase mb-1">Petal Width (cm)</label>
                        <input type="number" step="0.1" id="petal_width" value="0.2" class="w-full bg-slate-50 border border-slate-200 rounded-xl px-3 py-2 text-xs font-extrabold text-slate-800 focus:outline-none focus:ring-2 focus:ring-pink-500">
                    </div>
                    
                    <button type="submit" class="w-full py-3 bg-gradient-to-r from-pink-500 to-rose-500 hover:from-pink-600 hover:to-rose-600 text-white font-extrabold rounded-xl text-xs shadow-md shadow-pink-500/20 transition mt-2">
                        Phân tích Machine Learning & AI
                    </button>
                </form>
            </div>
        </section>

        <!-- CỘT GIỮA: KẾT QUẢ & BIỂU ĐỒ XÁC SUẤT (Col-4) -->
        <section class="lg:col-span-4 space-y-6">
            <div class="glass-panel p-5 rounded-2xl flex flex-col items-center text-center">
                <div class="w-24 h-24 rounded-2xl overflow-hidden bg-slate-100 border-2 border-slate-200 shadow-sm mb-3">
                    <img id="resultImage" src="https://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg" class="w-full h-full object-cover">
                </div>
                <span id="anomalyBadge" class="text-[10px] bg-emerald-100 text-emerald-800 font-extrabold px-2.5 py-0.5 rounded-full uppercase tracking-wider mb-1">Bình thường</span>
                <h2 id="predClass" class="text-2xl font-black text-slate-900 tracking-tight">Setosa</h2>
                <p class="text-xs text-slate-500 mt-0.5">Độ tin cậy: <span id="predConf" class="font-extrabold text-pink-600">99.8%</span></p>
            </div>

            <div class="glass-panel p-5 rounded-2xl">
                <h3 class="text-xs font-bold text-slate-800 uppercase tracking-wider mb-3 flex items-center gap-2">
                    <i class="fa-solid fa-chart-simple text-pink-500"></i> Phân bố xác suất 3 loài
                </h3>
                <div class="h-32">
                    <canvas id="probChart"></canvas>
                </div>
            </div>
        </section>

        <!-- CỘT PHẢI: BÁO CÁO CHĂM SÓC AI (Col-4) -->
        <section class="lg:col-span-4 space-y-6">
            <div class="glass-panel p-5 rounded-2xl h-full flex flex-col">
                <div class="flex justify-between items-center mb-3">
                    <h3 class="text-xs font-bold text-slate-800 uppercase tracking-wider flex items-center gap-2">
                        <i class="fa-solid fa-sparkles text-pink-500"></i> Báo cáo Chăm sóc AI
                    </h3>
                    <span class="text-[10px] text-pink-600 bg-pink-50 px-2 py-0.5 rounded-full font-bold">Gemini 2.0</span>
                </div>
                <div id="aiReportContent" class="text-xs text-slate-600 leading-relaxed overflow-y-auto whitespace-pre-line bg-slate-50 p-4 rounded-xl border border-slate-200 font-medium flex-grow min-h-[220px]">
                    Hệ thống đã sẵn sàng. Nhấn nút phân tích để xem báo cáo chi tiết từ trợ lý AI.
                </div>
            </div>
        </section>

        <!-- BẢNG LỊCH SỬ DỮ LIỆU SQLITE (FULL WIDTH) -->
        <section class="lg:col-span-12">
            <div class="glass-panel p-5 rounded-2xl">
                <div class="flex justify-between items-center mb-4">
                    <h3 class="text-xs font-extrabold text-slate-900 uppercase tracking-wider flex items-center gap-2">
                        <i class="fa-solid fa-database text-pink-500"></i> Lịch sử dự đoán (SQLite Database)
                    </h3>
                    <button onclick="loadLogs()" class="text-xs text-pink-600 font-bold hover:underline"><i class="fa-solid fa-rotate-right"></i> Làm mới</button>
                </div>
                <div class="overflow-x-auto">
                    <table class="w-full text-left text-xs">
                        <thead>
                            <tr class="border-b border-slate-200 text-slate-400 font-bold uppercase">
                                <th class="pb-2.5">Thời gian</th>
                                <th class="pb-2.5">Thông số (SL/SW/PL/PW)</th>
                                <th class="pb-2.5">Dự đoán</th>
                                <th class="pb-2.5">Độ tin cậy</th>
                                <th class="pb-2.5">Trạng thái</th>
                            </tr>
                        </thead>
                        <tbody id="logsTableBody" class="divide-y divide-slate-100 font-semibold text-slate-700">
                            <tr><td colspan="5" class="py-3 text-center text-slate-400">Đang tải dữ liệu lịch sử...</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </section>

    </main>

    <!-- FOOTER -->
    <footer class="text-center py-4 text-xs text-slate-400 font-medium border-t border-slate-200 bg-white">
        Iris Botanical Intelligence Suite &bull; FastAPI & Google Gemini AI
    </footer>

    <!-- JAVASCRIPT -->
    <script>
        const ctxProb = document.getElementById('probChart').getContext('2d');
        const probChart = new Chart(ctxProb, {
            type: 'bar',
            data: {
                labels: ['Setosa', 'Versicolor', 'Virginica'],
                datasets: [{ data: [100, 0, 0], backgroundColor: ['#ec4899', '#a855f7', '#3b82f6'], borderRadius: 6 }]
            },
            options: {
                plugins: { legend: { display: false } },
                scales: { 
                    y: { max: 100, ticks: { display: false }, grid: { display: false } }, 
                    x: { grid: { display: false }, ticks: { font: { size: 11, weight: 'bold' } } } 
                }
            }
        });

        function toggleDropdown() {
            document.getElementById('exportDropdown').classList.toggle('hidden');
        }

        window.onclick = function(event) {
            if (!event.target.closest('button')) {
                const dropdown = document.getElementById('exportDropdown');
                if (dropdown && !dropdown.classList.contains('hidden')) {
                    dropdown.classList.add('hidden');
                }
            }
        }

        function loadSample(type) {
            const samples = {
                'setosa': [5.1, 3.5, 1.4, 0.2],
                'versicolor': [6.0, 2.9, 4.5, 1.5],
                'virginica': [6.5, 3.0, 5.8, 2.2]
            };
            const val = samples[type];
            document.getElementById('sepal_length').value = val[0];
            document.getElementById('sepal_width').value = val[1];
            document.getElementById('petal_length').value = val[2];
            document.getElementById('petal_width').value = val[3];
            document.getElementById('predictionForm').dispatchEvent(new Event('submit'));
        }

        const sampleImages = {
            'Setosa': 'https://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg',
            'Versicolor': 'https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg',
            'Virginica': 'https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg'
        };

        async function loadLogs() {
            try {
                let res = await fetch('/logs');
                let logs = await res.json();
                let tbody = document.getElementById('logsTableBody');
                if(logs.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="5" class="py-3 text-center text-slate-400">Chưa có bản ghi nào.</td></tr>';
                    return;
                }
                tbody.innerHTML = logs.map(item => `
                    <tr class="hover:bg-slate-50 transition">
                        <td class="py-2.5 text-slate-500 font-normal">${item.timestamp}</td>
                        <td class="py-2.5 font-mono">${item.sepal_length} / ${item.sepal_width} / ${item.petal_length} / ${item.petal_width}</td>
                        <td class="py-2.5 text-pink-600 font-extrabold">${item.prediction}</td>
                        <td class="py-2.5 text-emerald-600 font-bold">${(item.confidence * 100).toFixed(1)}%</td>
                        <td class="py-2.5">
                            <span class="px-2 py-0.5 rounded-full text-[10px] font-bold ${item.is_anomaly ? 'bg-amber-100 text-amber-800' : 'bg-emerald-100 text-emerald-800'}">
                                ${item.is_anomaly ? 'Bất thường' : 'Bình thường'}
                            </span>
                        </td>
                    </tr>
                `).join('');
            } catch(e) {
                console.error("Lỗi tải logs:", e);
            }
        }

        document.getElementById('predictionForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            const data = {
                sepal_length: parseFloat(document.getElementById('sepal_length').value),
                sepal_width: parseFloat(document.getElementById('sepal_width').value),
                petal_length: parseFloat(document.getElementById('petal_length').value),
                petal_width: parseFloat(document.getElementById('petal_width').value)
            };

            document.getElementById('aiReportContent').innerText = "⏳ Đang phân tích mô hình ML và gọi Gemini AI tạo báo cáo...";

            try {
                let res = await fetch('/predict', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(data)
                });
                let result = await res.json();
                
                document.getElementById('predClass').innerText = result.prediction;
                document.getElementById('predConf').innerText = (result.confidence * 100).toFixed(1) + '%';
                
                if (sampleImages[result.prediction]) {
                    document.getElementById('resultImage').src = sampleImages[result.prediction];
                }

                if(result.probabilities) {
                    probChart.data.datasets[0].data = result.probabilities;
                    probChart.update();
                }

                const badge = document.getElementById('anomalyBadge');
                if (result.is_anomaly) {
                    badge.className = "text-[10px] bg-amber-100 text-amber-800 font-extrabold px-2.5 py-0.5 rounded-full uppercase tracking-wider mb-1";
                    badge.innerText = "Cảnh báo bất thường";
                } else {
                    badge.className = "text-[10px] bg-emerald-100 text-emerald-800 font-extrabold px-2.5 py-0.5 rounded-full uppercase tracking-wider mb-1";
                    badge.innerText = "Bình thường";
                }

                if(result.ai_report) {
                    document.getElementById('aiReportContent').innerText = result.ai_report;
                }

                loadLogs();
            } catch(err) {
                document.getElementById('aiReportContent').innerText = "❌ Lỗi kết nối đến máy chủ.";
            }
        });

        // Tải logs lần đầu khi mở trang
        window.onload = function() {
            loadLogs();
        };
    </script>
</body>
</html>
    """
