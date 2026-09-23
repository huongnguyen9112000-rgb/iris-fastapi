import os
import sqlite3
import csv
import io
from datetime import datetime
from typing import Optional

import numpy as np
import joblib
from fastapi import FastAPI, HTTPException, Response, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Google GenAI SDK (Sử dụng thư viện google-genai mới nhất)
from google import genai
from google.genai import types

app = FastAPI(title="Botanical Iris AI Ultra System", version="4.0")

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

# --- 2. MÔ HÌNH MACHINE LEARNING (Tự động train và lưu nếu chưa có) ---
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

# --- 3. KHỞI TẠO GOOGLE GEMINI AI ---
# Đảm bảo bạn đã thiết lập biến môi trường GEMINI_API_KEY trước khi chạy
try:
    gemini_client = genai.Client()
except Exception as e:
    gemini_client = None

# --- PYDANTIC SCHEMAS ---
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class ChatInput(BaseModel):
    message: str

# --- 4. CÁC API ENDPOINTS ---

@app.post("/predict")
def predict_iris(data: IrisInput):
    features = np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
    
    # Dự đoán mô hình
    probs = model.predict_proba(features)[0]
    pred_idx = np.argmax(probs)
    prediction = target_names[pred_idx]
    confidence = float(probs[pred_idx])
    
    # Kiểm tra dị thường (Ví dụ: kích thước cánh hoa quá ngắn/dài bất thường)
    is_anomaly = 1 if (data.petal_length < 1.0 or data.petal_length > 7.5) else 0

    # Gọi Google Gemini AI để tạo báo cáo chăm sóc
    ai_report = "Hệ thống AI đang offline hoặc chưa cấu hình GEMINI_API_KEY."
    if gemini_client:
        try:
            prompt = f"Phân tích loài hoa Iris {prediction} với các thông số: Sepal Length={data.sepal_length}, Sepal Width={data.sepal_width}, Petal Length={data.petal_length}, Petal Width={data.petal_width}. Hãy đưa ra tư vấn ngắn gọn về điều kiện sinh trưởng, đất trồng, độ ẩm và cách chăm sóc tối ưu bằng tiếng Việt."
            response = gemini_client.models.generate_content(
                model='gemini-2.0-flash',
                contents=prompt
            )
            ai_report = response.text
        except Exception as e:
            ai_report = f"Lỗi gọi Gemini AI: {str(e)}"

    # Lưu lịch sử vào SQLite
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
        "is_anomaly": bool(is_anomaly),
        "ai_report": ai_report
    }

@app.post("/chat")
def chat_with_assistant(data: ChatInput):
    if not gemini_client:
        return {"response": "Trợ lý AI chưa được cấu hình API Key. Vui lòng thiết lập biến môi trường GEMINI_API_KEY."}
    try:
        response = gemini_client.models.generate_content(
            model='gemini-2.0-flash',
            contents=f"Bạn là chuyên gia nông học, chuyên gia chăm sóc hoa và thực vật Iris. Hãy trả lời câu hỏi sau của người dùng bằng tiếng Việt một cách chuyên nghiệp, thân thiện: {data.message}"
        )
        return {"response": response.text}
    except Exception as e:
        return {"response": f"Đã xảy ra lỗi khi kết nối với Gemini: {str(e)}"}

@app.get("/export/{format_type}")
def export_data(format_type: str):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT id, timestamp, sepal_length, sepal_width, petal_length, petal_width, prediction, confidence, is_anomaly FROM predictions")
    rows = cursor.fetchall()
    conn.close()

    if format_type == "json":
        data = []
        for r in rows:
            data.append({
                "id": r[0], "timestamp": r[1],
                "sepal_length": r[2], "sepal_width": r[3],
                "petal_length": r[4], "petal_width": r[5],
                "prediction": r[6], "confidence": r[7], "is_anomaly": bool(r[8])
            })
        return JSONResponse(content=data)

    elif format_type == "csv":
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["ID", "Timestamp", "Sepal Length", "Sepal Width", "Petal Length", "Petal Width", "Prediction", "Confidence", "Is Anomaly"])
        for r in rows:
            writer.writerow(r)
        
        response = Response(output.getvalue(), media_type="text/csv")
        response.headers["Content-Disposition"] = "attachment; filename=iris_predictions_export.csv"
        return response

    else:
        raise HTTPException(status_code=400, detail="Định dạng xuất không hỗ trợ (chỉ chấp nhận json hoặc csv).")

# --- 5. GIAO DIỆN WEB DASHBOARD (Single Page Application hiện đại) ---
@app.get("/", response_class=HTMLResponse)
def get_dashboard():
    return """
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Botanical Iris AI - Enterprise Suite</title>
    <!-- Tailwind CSS & FontAwesome -->
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
        body { font-family: 'Plus Jakarta Sans', sans-serif; }
        .glass-panel { background: rgba(255, 255, 255, 0.85); backdrop-filter: blur(12px); border: 1px solid rgba(255, 182, 193, 0.3); }
    </style>
</head>
<body class="bg-gradient-to-br from-pink-50 via-purple-50 to-rose-100 min-h-screen text-gray-800 flex flex-col">

    <!-- NAVBAR CHUYÊN NGHIỆP -->
    <header class="bg-white/85 backdrop-blur-md sticky top-0 z-50 border-b border-pink-100 shadow-sm">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
            <div class="flex items-center space-x-3">
                <div class="bg-gradient-to-tr from-pink-500 to-rose-400 text-white p-2.5 rounded-xl shadow-md">
                    <i class="fa-solid fa-seedling text-lg"></i>
                </div>
                <div>
                    <h1 class="font-bold text-gray-900 tracking-tight text-base sm:text-lg">Iris Botanical AI</h1>
                    <span class="text-[10px] text-pink-600 font-semibold bg-pink-50 px-2 py-0.5 rounded-full">Enterprise v4.0</span>
                </div>
            </div>

            <!-- Menu Phải & Nút ẩn chức năng (Dropdown) -->
            <div class="flex items-center space-x-3">
                <!-- Nút Trợ lý AI (Mở Modal) -->
                <button onclick="toggleChatModal()" class="bg-pink-100 hover:bg-pink-200 text-pink-700 px-3.5 py-2 rounded-xl text-xs sm:text-sm font-semibold transition flex items-center gap-2">
                    <i class="fa-solid fa-robot"></i> <span class="hidden sm:inline">Trợ lý AI</span>
                </button>

                <!-- Menu Dropdown Xuất Dữ Liệu -->
                <div class="relative">
                    <button onclick="toggleDropdown()" class="bg-gray-900 hover:bg-gray-800 text-white px-3.5 py-2 rounded-xl text-xs sm:text-sm font-semibold transition flex items-center gap-2 shadow-sm">
                        <i class="fa-solid fa-download"></i> <span class="hidden sm:inline">Xuất dữ liệu</span> <i class="fa-solid fa-chevron-down text-[10px]"></i>
                    </button>
                    <div id="exportDropdown" class="hidden absolute right-0 mt-2 w-48 bg-white rounded-xl shadow-xl border border-gray-100 py-2 z-50">
                        <a href="/export/json" class="block px-4 py-2 text-xs sm:text-sm text-gray-700 hover:bg-pink-50 hover:text-pink-600"><i class="fa-solid fa-file-code mr-2"></i> Tải JSON</a>
                        <a href="/export/csv" class="block px-4 py-2 text-xs sm:text-sm text-gray-700 hover:bg-pink-50 hover:text-pink-600"><i class="fa-solid fa-file-csv mr-2"></i> Tải CSV</a>
                    </div>
                </div>
            </div>
        </div>
    </header>

    <!-- NỘI DUNG CHÍNH -->
    <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 flex-grow grid grid-cols-1 lg:grid-cols-12 gap-8 w-full items-start">
        
        <!-- CỘT TRÁI: THÔNG SỐ & MẪU NHANH -->
        <section class="lg:col-span-5 space-y-6">
            <div class="glass-panel p-6 rounded-2xl shadow-sm">
                <div class="flex justify-between items-center mb-4">
                    <h2 class="text-sm font-bold text-gray-900 flex items-center gap-2">
                        <i class="fa-solid fa-sliders text-pink-500"></i> Thông số & Mẫu nhanh
                    </h2>
                    <div class="flex gap-1.5">
                        <button type="button" onclick="loadSample('setosa')" class="text-[11px] bg-pink-50 hover:bg-pink-100 text-pink-700 font-semibold px-2.5 py-1 rounded-lg transition">Setosa</button>
                        <button type="button" onclick="loadSample('versicolor')" class="text-[11px] bg-purple-50 hover:bg-purple-100 text-purple-700 font-semibold px-2.5 py-1 rounded-lg transition">Versicolor</button>
                        <button type="button" onclick="loadSample('virginica')" class="text-[11px] bg-rose-50 hover:bg-rose-100 text-rose-700 font-semibold px-2.5 py-1 rounded-lg transition">Virginica</button>
                    </div>
                </div>

                <!-- Form nhập liệu chính -->
                <form id="predictionForm" class="space-y-4">
                    <div class="grid grid-cols-2 gap-4">
                        <div>
                            <label class="block text-[11px] font-bold text-gray-600 uppercase mb-1">Sepal Length (cm)</label>
                            <input type="number" step="0.1" id="sepal_length" value="5.1" class="w-full bg-white/70 border border-pink-200 rounded-xl px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-pink-400 font-medium">
                        </div>
                        <div>
                            <label class="block text-[11px] font-bold text-gray-600 uppercase mb-1">Sepal Width (cm)</label>
                            <input type="number" step="0.1" id="sepal_width" value="3.5" class="w-full bg-white/70 border border-pink-200 rounded-xl px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-pink-400 font-medium">
                        </div>
                        <div>
                            <label class="block text-[11px] font-bold text-gray-600 uppercase mb-1">Petal Length (cm)</label>
                            <input type="number" step="0.1" id="petal_length" value="1.4" class="w-full bg-white/70 border border-pink-200 rounded-xl px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-pink-400 font-medium">
                        </div>
                        <div>
                            <label class="block text-[11px] font-bold text-gray-600 uppercase mb-1">Petal Width (cm)</label>
                            <input type="number" step="0.1" id="petal_width" value="0.2" class="w-full bg-white/70 border border-pink-200 rounded-xl px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-pink-400 font-medium">
                        </div>
                    </div>

                    <!-- Upload file thu gọn -->
                    <div class="border-2 border-dashed border-pink-200 hover:border-pink-400 rounded-xl p-3 text-center cursor-pointer transition bg-white/40">
                        <input type="file" id="fileUpload" class="hidden">
                        <label for="fileUpload" class="cursor-pointer flex items-center justify-center gap-2">
                            <i class="fa-solid fa-cloud-arrow-up text-pink-400 text-base"></i>
                            <span class="text-xs font-medium text-gray-600">Tải lên file hoặc hình ảnh hoa</span>
                        </label>
                    </div>

                    <button type="submit" class="w-full bg-gradient-to-r from-pink-500 to-rose-500 hover:from-pink-600 hover:to-rose-600 text-white font-bold py-2.5 rounded-xl text-sm shadow-md transition flex items-center justify-center gap-2">
                        <i class="fa-solid fa-wand-magic-sparkles"></i> Phân loại & Phân tích AI
                    </button>
                </form>
            </div>
        </section>

        <!-- CỘT PHẢI: KẾT QUẢ & BÁO CÁO AI -->
        <section class="lg:col-span-7 space-y-6">
            <!-- Kết quả ML -->
            <div class="glass-panel p-6 rounded-2xl shadow-sm flex flex-col sm:flex-row items-center gap-6">
                <div class="w-28 h-28 flex-shrink-0 bg-pink-100/50 rounded-2xl flex items-center justify-center overflow-hidden border border-pink-200">
                    <img id="resultImage" src="https://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg" alt="Iris" class="object-cover w-full h-full">
                </div>
                <div class="flex-grow text-center sm:text-left">
                    <span id="anomalyBadge" class="inline-block bg-emerald-100 text-emerald-800 text-[11px] font-bold px-2.5 py-1 rounded-full mb-1">BÌNH THƯỜNG</span>
                    <h3 class="text-[11px] uppercase tracking-wider text-gray-500 font-bold">Kết quả dự đoán mô hình</h3>
                    <p id="predClass" class="text-2xl font-black text-gray-900 mt-0.5">Setosa</p>
                    <p class="text-xs text-gray-600 mt-1">Độ tin cậy: <span id="predConf" class="font-bold text-pink-600">99.8%</span></p>
                </div>
            </div>

            <!-- Báo cáo chăm sóc AI -->
            <div class="glass-panel p-6 rounded-2xl shadow-sm">
                <div class="flex justify-between items-center mb-3">
                    <h3 class="text-sm font-bold text-gray-900 flex items-center gap-2">
                        <i class="fa-solid fa-file-lines text-pink-500"></i> Báo cáo Tư vấn Chăm sóc AI
                    </h3>
                    <span class="text-[11px] text-pink-600 bg-pink-50 px-2.5 py-1 rounded-lg font-semibold flex items-center gap-1"><i class="fa-solid fa-sparkles"></i> Gemini 2.0 Flash</span>
                </div>
                <div id="aiReportContent" class="text-xs sm:text-sm text-gray-700 bg-white/60 p-4 rounded-xl border border-pink-100 min-h-[120px] leading-relaxed whitespace-pre-line">
                    Nhấn "Phân loại & Phân tích AI" để AI tự động xuất báo cáo chi tiết về điều kiện sinh trưởng, đặc điểm hình thái và phương pháp chăm sóc loài hoa này.
                </div>
            </div>
        </section>
    </main>

    <!-- CHAT MODAL (Ẩn sẵn, hiện khi bấm nút Trợ lý AI) -->
    <div id="chatModal" class="hidden fixed inset-0 bg-black/45 backdrop-blur-sm z-50 flex items-center justify-center p-4">
        <div class="bg-white w-full max-w-lg rounded-2xl shadow-2xl overflow-hidden flex flex-col h-[520px]">
            <div class="bg-gradient-to-r from-pink-500 to-rose-500 p-4 text-white flex justify-between items-center">
                <h3 class="font-bold text-sm flex items-center gap-2"><i class="fa-solid fa-robot"></i> Trợ lý Chăm sóc Cây trồng</h3>
                <button onclick="toggleChatModal()" class="text-white hover:text-pink-100 text-lg font-bold px-1">&times;</button>
            </div>
            <div id="chatMessages" class="flex-grow p-4 overflow-y-auto space-y-3 text-xs sm:text-sm bg-gray-50">
                <div class="bg-pink-100 text-pink-900 p-3 rounded-2xl rounded-tl-none max-w-[85%] font-medium">
                    Xin chào! Tôi là trợ lý AI chuyên về các giống hoa Iris. Bạn có thắc mắc gì về cách chăm sóc, tưới nước hay ánh sáng không?
                </div>
            </div>
            <div class="p-3 bg-white border-t border-gray-100 flex gap-2">
                <input type="text" id="chatInput" placeholder="Nhập câu hỏi cho trợ lý AI..." class="flex-grow bg-gray-100 border border-gray-200 rounded-xl px-4 py-2 text-xs sm:text-sm focus:outline-none focus:ring-2 focus:ring-pink-400" onkeypress="if(event.key === 'Enter') sendChatMessage()">
                <button onclick="sendChatMessage()" class="bg-pink-500 text-white px-4 py-2 rounded-xl text-sm font-semibold hover:bg-pink-600 transition"><i class="fa-solid fa-paper-plane"></i></button>
            </div>
        </div>
    </div>

    <!-- SCRIPT XỬ LÝ GIAO DIỆN VÀ API -->
    <script>
        function toggleDropdown() {
            const dropdown = document.getElementById('exportDropdown');
            dropdown.classList.toggle('hidden');
        }

        function toggleChatModal() {
            const modal = document.getElementById('chatModal');
            modal.classList.toggle('hidden');
        }

        // Đóng dropdown khi click ra ngoài
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
        }

        const sampleImages = {
            'Setosa': 'https://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg',
            'Versicolor': 'https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg',
            'Virginica': 'https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg'
        };

        // Gửi dự đoán mô hình
        document.getElementById('predictionForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            const data = {
                sepal_length: parseFloat(document.getElementById('sepal_length').value),
                sepal_width: parseFloat(document.getElementById('sepal_width').value),
                petal_length: parseFloat(document.getElementById('petal_length').value),
                petal_width: parseFloat(document.getElementById('petal_width').value)
            };

            document.getElementById('aiReportContent').innerText = "Đang phân tích thông số và gọi Gemini AI tạo báo cáo...";

            try {
                let response = await fetch('/predict', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(data)
                });
                let result = await response.json();
                
                document.getElementById('predClass').innerText = result.prediction;
                document.getElementById('predConf').innerText = (result.confidence * 100).toFixed(1) + '%';
                
                if (sampleImages[result.prediction]) {
                    document.getElementById('resultImage').src = sampleImages[result.prediction];
                }

                const badge = document.getElementById('anomalyBadge');
                if (result.is_anomaly) {
                    badge.className = "inline-block bg-amber-100 text-amber-800 text-[11px] font-bold px-2.5 py-1 rounded-full mb-1";
                    badge.innerText = "CẢNH BÁO DỊ THƯỜNG";
                } else {
                    badge.className = "inline-block bg-emerald-100 text-emerald-800 text-[11px] font-bold px-2.5 py-1 rounded-full mb-1";
                    badge.innerText = "BÌNH THƯỜNG";
                }

                if(result.ai_report) {
                    document.getElementById('aiReportContent').innerText = result.ai_report;
                }
            } catch(err) {
                console.error(err);
                document.getElementById('aiReportContent').innerText = "Đã xảy ra lỗi khi kết nối đến server.";
            }
        });

        async function sendChatMessage() {
            const input = document.getElementById('chatInput');
            const container = document.getElementById('chatMessages');
            if(!input.value.trim()) return;

            const userText = input.value;
            container.innerHTML += `<div class="bg-gray-800 text-white p-3 rounded-2xl rounded-tr-none max-w-[85%] ml-auto font-medium">${userText}</div>`;
            input.value = '';
            container.scrollTop = container.scrollHeight;

            try {
                let res = await fetch('/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({message: userText})
                });
                let data = await res.json();
                container.innerHTML += `<div class="bg-pink-100 text-pink-900 p-3 rounded-2xl rounded-tl-none max-w-[85%] font-medium">${data.response}</div>`;
                container.scrollTop = container.scrollHeight;
            } catch(e) {
                container.innerHTML += `<div class="bg-pink-100 text-pink-900 p-3 rounded-2xl rounded-tl-none max-w-[85%] font-medium">Xin lỗi, trợ lý AI đang bận kết nối.</div>`;
            }
        }
    </script>
</body>
</html>
    """
