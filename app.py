import os
import sqlite3
import csv
import io
from datetime import datetime

import numpy as np
import joblib
from fastapi import FastAPI, HTTPException, Response, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

from google import genai

app = FastAPI(title="Botanical Iris App", version="5.0")

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

class ChatInput(BaseModel):
    message: str

# --- 4. API ENDPOINTS ---
@app.post("/predict")
def predict_iris(data: IrisInput):
    features = np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
    
    probs = model.predict_proba(features)[0]
    pred_idx = np.argmax(probs)
    prediction = target_names[pred_idx]
    confidence = float(probs[pred_idx])
    
    # Tính xác suất 3 loài cho biểu đồ
    probabilities = [round(float(p) * 100, 1) for p in probs]
    is_anomaly = 1 if (data.petal_length < 1.0 or data.petal_length > 7.5) else 0

    ai_report = "Hệ thống AI chưa được cấu hình khóa API."
    if gemini_client:
        try:
            prompt = f"Phân tích loài hoa Iris {prediction} với các thông số: Sepal Length={data.sepal_length}, Sepal Width={data.sepal_width}, Petal Length={data.petal_length}, Petal Width={data.petal_width}. Hãy đưa ra tư vấn ngắn gọn về điều kiện sinh trưởng và cách chăm sóc tối ưu bằng tiếng Việt."
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

@app.post("/chat")
def chat_with_assistant(data: ChatInput):
    if not gemini_client:
        return {"response": "Trợ lý AI chưa được cấu hình API Key."}
    try:
        response = gemini_client.models.generate_content(
            model='gemini-2.0-flash',
            contents=f"Bạn là chuyên gia nông học. Trả lời ngắn gọn bằng tiếng Việt: {data.message}"
        )
        return {"response": response.text}
    except Exception as e:
        return {"response": f"Lỗi kết nối Gemini: {str(e)}"}

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

# --- 5. GIAO DIỆN APP DASHBOARD HIỆN ĐẠI (GIỐNG MOBILE/WEB APP THỰC THỤ) ---
@app.get("/", response_class=HTMLResponse)
def get_dashboard():
    return """
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Iris Plant App</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
        body { font-family: 'Plus Jakarta Sans', sans-serif; background-color: #f8fafc; }
    </style>
</head>
<body class="flex items-center justify-center min-h-screen p-2 sm:p-6">

    <!-- KHUNG ỨNG DỤNG (APP CONTAINER CHÍNH) -->
    <div class="w-full max-w-md bg-white rounded-3xl shadow-2xl border border-gray-100 overflow-hidden flex flex-col h-[90vh] max-h-[850px] relative">
        
        <!-- HEADER APP -->
        <div class="px-6 py-4 bg-white border-b border-gray-100 flex items-center justify-between z-10">
            <div class="flex items-center space-x-3">
                <div class="w-10 h-10 rounded-2xl bg-pink-500 text-white flex items-center justify-center shadow-md shadow-pink-500/30">
                    <i class="fa-solid fa-seedling"></i>
                </div>
                <div>
                    <h1 class="font-bold text-gray-900 text-sm">Iris Care App</h1>
                    <span class="text-[10px] text-emerald-600 font-semibold bg-emerald-50 px-2 py-0.5 rounded-full">● Online AI</span>
                </div>
            </div>
            <!-- Nút mở Menu chức năng (Action Sheet) -->
            <button onclick="toggleMenuModal()" class="w-9 h-9 rounded-xl bg-gray-100 hover:bg-gray-200 text-gray-700 flex items-center justify-center transition">
                <i class="fa-solid fa-ellipsis-vertical text-sm"></i>
            </button>
        </div>

        <!-- MÀN HÌNH NỘI DUNG CHÍNH (SCROLLABLE) -->
        <div class="flex-grow overflow-y-auto p-5 space-y-4 bg-gradient-to-b from-gray-50/50 to-white">
            
            <!-- Thẻ Kết quả & Hình ảnh (Compact Card) -->
            <div class="bg-white p-4 rounded-2xl shadow-sm border border-gray-100 flex items-center gap-4">
                <div class="w-20 h-20 rounded-xl overflow-hidden bg-gray-100 flex-shrink-0 border border-gray-200">
                    <img id="resultImage" src="https://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg" class="w-full h-full object-cover">
                </div>
                <div class="flex-grow">
                    <span id="anomalyBadge" class="text-[10px] bg-emerald-100 text-emerald-800 font-bold px-2 py-0.5 rounded-full">Bình thường</span>
                    <h2 id="predClass" class="text-xl font-black text-gray-900 mt-1">Setosa</h2>
                    <p class="text-xs text-gray-500">Độ tin cậy: <span id="predConf" class="font-bold text-pink-600">99.8%</span></p>
                </div>
            </div>

            <!-- Nút bấm mở Form nhập thông số (Modal Trigger) -->
            <button onclick="toggleInputModal()" class="w-full py-3 px-4 bg-pink-50 hover:bg-pink-100 text-pink-700 font-bold rounded-2xl text-xs flex items-center justify-between transition border border-pink-200 shadow-sm">
                <span class="flex items-center gap-2"><i class="fa-solid fa-sliders text-pink-500"></i> Nhập thông số & Mẫu nhanh</span>
                <i class="fa-solid fa-chevron-right text-[10px]"></i>
            </button>

            <!-- Biểu đồ xác suất thu gọn -->
            <div class="bg-white p-4 rounded-2xl shadow-sm border border-gray-100">
                <p class="text-[11px] font-bold text-gray-700 mb-2 flex items-center gap-2">
                    <i class="fa-solid fa-chart-simple text-pink-500"></i> Phân bố xác suất 3 loài
                </p>
                <div class="h-28">
                    <canvas id="probChart"></canvas>
                </div>
            </div>

            <!-- Khung Báo cáo AI Tư vấn (Có thể cuộn) -->
            <div class="bg-white p-4 rounded-2xl shadow-sm border border-gray-100">
                <div class="flex justify-between items-center mb-2">
                    <p class="text-[11px] font-bold text-gray-700 flex items-center gap-2">
                        <i class="fa-solid fa-robot text-pink-500"></i> Báo cáo Chăm sóc AI
                    </p>
                    <span class="text-[10px] text-pink-600 bg-pink-50 px-2 py-0.5 rounded-md font-semibold">Gemini 2.0</span>
                </div>
                <div id="aiReportContent" class="text-xs text-gray-600 leading-relaxed max-h-36 overflow-y-auto whitespace-pre-line bg-gray-50 p-3 rounded-xl">
                    Nhấn nút tùy chỉnh thông số phía trên để chạy phân tích và nhận báo cáo chăm sóc chi tiết từ AI.
                </div>
            </div>

        </div>

        <!-- BOTTOM NAV BAR (Thanh điều hướng dưới giống App di động) -->
        <div class="bg-white border-t border-gray-100 px-6 py-3 flex justify-around items-center z-10">
            <button onclick="location.reload()" class="flex flex-col items-center text-pink-600 gap-1">
                <i class="fa-solid fa-house text-base"></i>
                <span class="text-[10px] font-bold">Trang chủ</span>
            </button>
            <button onclick="toggleChatModal()" class="flex flex-col items-center text-gray-400 hover:text-pink-600 gap-1 transition">
                <i class="fa-solid fa-comments text-base"></i>
                <span class="text-[10px] font-semibold">Trợ lý AI</span>
            </button>
            <button onclick="toggleMenuModal()" class="flex flex-col items-center text-gray-400 hover:text-pink-600 gap-1 transition">
                <i class="fa-solid fa-bars text-base"></i>
                <span class="text-[10px] font-semibold">Menu</span>
            </button>
        </div>

        <!-- ================= MODAL 1: FORM NHẬP THÔNG SỐ (POPUP) ================= -->
        <div id="inputModal" class="hidden absolute inset-0 bg-black/50 backdrop-blur-sm z-50 flex flex-col justify-end transition-all">
            <div class="bg-white rounded-t-3xl p-6 space-y-4 max-h-[80vh] overflow-y-auto shadow-2xl">
                <div class="flex justify-between items-center border-b pb-3">
                    <h3 class="font-bold text-sm text-gray-900">Thông số & Mẫu hoa</h3>
                    <button onclick="toggleInputModal()" class="w-8 h-8 rounded-full bg-gray-100 text-gray-500 hover:bg-gray-200 flex items-center justify-center font-bold">&times;</button>
                </div>
                
                <!-- Nút chọn mẫu nhanh -->
                <div class="flex gap-2">
                    <button type="button" onclick="loadSample('setosa'); toggleInputModal();" class="flex-1 text-xs bg-pink-50 hover:bg-pink-100 text-pink-700 font-bold py-2 rounded-xl transition">Setosa</button>
                    <button type="button" onclick="loadSample('versicolor'); toggleInputModal();" class="flex-1 text-xs bg-purple-50 hover:bg-purple-100 text-purple-700 font-bold py-2 rounded-xl transition">Versicolor</button>
                    <button type="button" onclick="loadSample('virginica'); toggleInputModal();" class="flex-1 text-xs bg-rose-50 hover:bg-rose-100 text-rose-700 font-bold py-2 rounded-xl transition">Virginica</button>
                </div>

                <form id="predictionForm" class="space-y-3">
                    <div>
                        <label class="block text-[10px] font-bold text-gray-500 uppercase mb-1">Sepal Length (cm)</label>
                        <input type="number" step="0.1" id="sepal_length" value="5.1" class="w-full bg-gray-50 border border-gray-200 rounded-xl px-3 py-2 text-xs font-semibold focus:outline-none focus:ring-2 focus:ring-pink-400">
                    </div>
                    <div>
                        <label class="block text-[10px] font-bold text-gray-500 uppercase mb-1">Sepal Width (cm)</label>
                        <input type="number" step="0.1" id="sepal_width" value="3.5" class="w-full bg-gray-50 border border-gray-200 rounded-xl px-3 py-2 text-xs font-semibold focus:outline-none focus:ring-2 focus:ring-pink-400">
                    </div>
                    <div>
                        <label class="block text-[10px] font-bold text-gray-500 uppercase mb-1">Petal Length (cm)</label>
                        <input type="number" step="0.1" id="petal_length" value="1.4" class="w-full bg-gray-50 border border-gray-200 rounded-xl px-3 py-2 text-xs font-semibold focus:outline-none focus:ring-2 focus:ring-pink-400">
                    </div>
                    <div>
                        <label class="block text-[10px] font-bold text-gray-500 uppercase mb-1">Petal Width (cm)</label>
                        <input type="number" step="0.1" id="petal_width" value="0.2" class="w-full bg-gray-50 border border-gray-200 rounded-xl px-3 py-2 text-xs font-semibold focus:outline-none focus:ring-2 focus:ring-pink-400">
                    </div>
                    
                    <button type="submit" onclick="toggleInputModal()" class="w-full py-3 bg-gradient-to-r from-pink-500 to-rose-500 text-white font-bold rounded-xl text-xs shadow-md shadow-pink-500/30 transition">
                        Xác nhận & Phân tích AI
                    </button>
                </form>
            </div>
        </div>

        <!-- ================= MODAL 2: TRỢ LÝ AI CHAT ================= -->
        <div id="chatModal" class="hidden absolute inset-0 bg-white z-50 flex flex-col">
            <div class="px-6 py-4 bg-white border-b flex items-center justify-between">
                <h3 class="font-bold text-sm text-gray-900 flex items-center gap-2">
                    <i class="fa-solid fa-robot text-pink-500"></i> Trợ lý Chăm sóc Cây
                </h3>
                <button onclick="toggleChatModal()" class="w-8 h-8 rounded-full bg-gray-100 text-gray-600 flex items-center justify-center font-bold">&times;</button>
            </div>
            <div id="chatMessages" class="flex-grow p-4 overflow-y-auto space-y-3 text-xs bg-gray-50">
                <div class="bg-pink-100 text-pink-900 p-3 rounded-2xl rounded-tl-none max-w-[85%] font-medium">
                    Xin chào! Tôi có thể giúp gì cho việc chăm sóc hoa của bạn?
                </div>
            </div>
            <div class="p-3 bg-white border-t flex gap-2">
                <input type="text" id="chatInput" placeholder="Nhập câu hỏi..." class="flex-grow bg-gray-100 border rounded-xl px-3 py-2 text-xs focus:outline-none focus:ring-2 focus:ring-pink-400" onkeypress="if(event.key==='Enter') sendChatMessage()">
                <button onclick="sendChatMessage()" class="bg-pink-500 text-white px-4 py-2 rounded-xl text-xs font-bold"><i class="fa-solid fa-paper-plane"></i></button>
            </div>
        </div>

        <!-- ================= MODAL 3: MENU CHỨC NĂNG PHỤ (XUẤT FILE) ================= -->
        <div id="menuModal" class="hidden absolute inset-0 bg-black/50 backdrop-blur-sm z-50 flex flex-col justify-end">
            <div class="bg-white rounded-t-3xl p-6 space-y-3 shadow-2xl">
                <div class="flex justify-between items-center border-b pb-3">
                    <h3 class="font-bold text-sm text-gray-900">Tính năng hệ thống</h3>
                    <button onclick="toggleMenuModal()" class="w-8 h-8 rounded-full bg-gray-100 text-gray-500 flex items-center justify-center font-bold">&times;</button>
                </div>
                <a href="/export/json" class="block w-full py-3 px-4 bg-gray-50 hover:bg-gray-100 rounded-xl text-xs font-bold text-gray-700 transition flex items-center gap-3">
                    <i class="fa-solid fa-file-code text-pink-500 text-base"></i> Tải dữ liệu JSON
                </a>
                <a href="/export/csv" class="block w-full py-3 px-4 bg-gray-50 hover:bg-gray-100 rounded-xl text-xs font-bold text-gray-700 transition flex items-center gap-3">
                    <i class="fa-solid fa-file-csv text-emerald-500 text-base"></i> Tải dữ liệu CSV
                </a>
            </div>
        </div>

    </div>

    <!-- SCRIPT XỬ LÝ -->
    <script>
        const ctxProb = document.getElementById('probChart').getContext('2d');
        const probChart = new Chart(ctxProb, {
            type: 'bar',
            data: {
                labels: ['Setosa', 'Versicolor', 'Virginica'],
                datasets: [{ data: [100, 0, 0], backgroundColor: ['#f472b6', '#c084fc', '#60a5fa'], borderRadius: 6 }]
            },
            options: {
                plugins: { legend: { display: false } },
                scales: { y: { max: 100, ticks: { display: false }, grid: { display: false } }, x: { grid: { display: false }, ticks: { font: { size: 10 } } } }
            }
        });

        function toggleInputModal() { document.getElementById('inputModal').classList.toggle('hidden'); }
        function toggleChatModal() { document.getElementById('chatModal').classList.toggle('hidden'); }
        function toggleMenuModal() { document.getElementById('menuModal').classList.toggle('hidden'); }

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

        document.getElementById('predictionForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            const data = {
                sepal_length: parseFloat(document.getElementById('sepal_length').value),
                sepal_width: parseFloat(document.getElementById('sepal_width').value),
                petal_length: parseFloat(document.getElementById('petal_length').value),
                petal_width: parseFloat(document.getElementById('petal_width').value)
            };

            document.getElementById('aiReportContent').innerText = "Đang phân tích và gọi Gemini AI...";

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
                    badge.className = "text-[10px] bg-amber-100 text-amber-800 font-bold px-2 py-0.5 rounded-full";
                    badge.innerText = "Bất thường";
                } else {
                    badge.className = "text-[10px] bg-emerald-100 text-emerald-800 font-bold px-2 py-0.5 rounded-full";
                    badge.innerText = "Bình thường";
                }

                if(result.ai_report) {
                    document.getElementById('aiReportContent').innerText = result.ai_report;
                }
            } catch(err) {
                document.getElementById('aiReportContent').innerText = "Lỗi kết nối server.";
            }
        });

        async function sendChatMessage() {
            const input = document.getElementById('chatInput');
            const container = document.getElementById('chatMessages');
            if(!input.value.trim()) return;

            const text = input.value;
            container.innerHTML += `<div class="bg-gray-800 text-white p-2.5 rounded-2xl rounded-tr-none max-w-[85%] ml-auto font-medium">${text}</div>`;
            input.value = '';
            container.scrollTop = container.scrollHeight;

            try {
                let res = await fetch('/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({message: text})
                });
                let data = await res.json();
                container.innerHTML += `<div class="bg-pink-100 text-pink-900 p-2.5 rounded-2xl rounded-tl-none max-w-[85%] font-medium">${data.response}</div>`;
                container.scrollTop = container.scrollHeight;
            } catch(e) {
                container.innerHTML += `<div class="bg-pink-100 text-pink-900 p-2.5 rounded-2xl rounded-tl-none max-w-[85%] font-medium">Lỗi kết nối AI.</div>`;
            }
        }
    </script>
</body>
</html>
    """
