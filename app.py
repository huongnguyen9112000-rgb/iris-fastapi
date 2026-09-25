import sqlite3
import csv
import json
import os
import io
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from database import init_db
from routers import predict_router

app = FastAPI(title="Iris Botanical Enterprise Suite", version="18.0")

# Khởi tạo cơ sở dữ liệu khi khởi động
init_db()

# Gắn router dự đoán chính
app.include_router(predict_router.router)

# Model Pydantic cho phần chẩn đoán
class DiagnosisInput(BaseModel):
    symptom: str

# API Chẩn đoán bệnh
@app.post("/diagnose")
def diagnose_plant(data: DiagnosisInput):
    symptom = data.symptom.lower()
    if "vàng" in symptom:
        return {"disease": "Bệnh vàng lá, thối rễ do úng nước", "solution": "Giảm lượng nước tưới, bổ sung vi sinh Trichoderma và cắt tỉa lá hỏng."}
    elif "đốm" in symptom:
        return {"disease": "Bệnh đốm lá do nấm Cercospora", "solution": "Sử dụng thuốc trừ nấm gốc đồng, thu gom và tiêu hủy lá bị bệnh nặng."}
    elif "sâu" in symptom:
        return {"disease": "Sâu ăn lá / Sâu đục thân", "solution": "Dùng chế phẩm sinh học Bt hoặc bắt sâu thủ công vào lúc sáng sớm."}
    elif "gỉ" in symptom:
        return {"disease": "Bệnh gỉ sắt (Rust)", "solution": "Phun thuốc đặc trị nấm gỉ sắt, giữ thông thoáng cho vườn hoa."}
    elif "thối" in symptom:
        return {"disease": "Bệnh thối mềm vi khuẩn (Soft Rot)", "solution": "Ngừng tưới nước ngay lập tức, cách ly cây bệnh và xử lý đất bằng vôi bột."}
    else:
        return {"disease": "Thiếu hụt dinh dưỡng hoặc quang hợp kém", "solution": "Bón phân NPK cân đối, đảm bảo cây nhận đủ ánh sáng mặt trời tự nhiên."}

# API Lịch sử dự đoán
@app.get("/logs")
def get_logs():
    db_path = "iris_database.db" # Hoặc đường dẫn file db của bạn
    if not os.path.exists(db_path):
        return []
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT 50")
        rows = cursor.fetchall()
        result = [dict(row) for row in rows]
    except Exception:
        result = []
    conn.close()
    return result

# API Xuất file JSON
@app.get("/export/json")
def export_json():
    db_path = "iris_database.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * FROM predictions")
        rows = [dict(row) for row in cursor.fetchall()]
    except Exception:
        rows = []
    conn.close()
    return Response(content=json.dumps(rows, ensure_ascii=False, indent=4), media_type="application/json", headers={"Content-Disposition": "attachment; filename=iris_logs.json"})

# API Xuất file CSV
@app.get("/export/csv")
def export_csv():
    db_path = "iris_database.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * FROM predictions")
        rows = cursor.fetchall()
        column_names = [description[0] for description in cursor.description]
    except Exception:
        rows = []
        column_names = []
    conn.close()
    
    output = io.StringIO()
    writer = csv.writer(output)
    if column_names:
        writer.writerow(column_names)
    writer.writerows(rows)
    
    return Response(content=output.getvalue(), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=iris_logs.csv"})

@app.get("/", response_class=HTMLResponse)
def get_dashboard():
    return """
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Iris Botanical Enterprise Suite - Multi-Model SVM</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
        body { font-family: 'Plus Jakarta Sans', sans-serif; background: #f8fafc; min-height: 100vh; }
        .glass-card { background: #ffffff; border: 1px solid #e2e8f0; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); }
        .tab-content { display: none; }
        .tab-content.active { display: grid; }
    </style>
</head>
<body class="flex flex-col min-h-screen text-slate-800">

    <header class="bg-white border-b border-slate-200 sticky top-0 z-40 px-6 py-3.5 shadow-sm">
        <div class="max-w-6xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-3">
            <div class="flex items-center space-x-3">
                <div class="w-10 h-10 rounded-xl bg-gradient-to-tr from-pink-500 to-rose-500 text-white flex items-center justify-center shadow-md shadow-pink-500/20">
                    <i class="fa-solid fa-seedling text-sm"></i>
                </div>
                <div>
                    <h1 class="font-extrabold text-slate-900 text-sm tracking-tight">Iris Botanical Intelligence</h1>
                    <span id="activeModelBadge" class="text-[10px] text-pink-600 font-bold bg-pink-50 px-2 py-0.5 rounded-full inline-flex items-center gap-1">
                        <span class="w-1.5 h-1.5 rounded-full bg-pink-500"></span> SVM Model: Linear
                    </span>
                </div>
            </div>
            
            <nav class="flex items-center bg-slate-100 p-1 rounded-2xl gap-1 flex-wrap justify-center">
                <button onclick="switchTab('dashboard')" id="nav-dashboard" class="px-3.5 py-1.5 rounded-xl text-xs font-extrabold transition bg-white text-pink-600 shadow-sm"><i class="fa-solid fa-chart-pie mr-1"></i> Dashboard</button>
                <button onclick="switchTab('config')" id="nav-config" class="px-3.5 py-1.5 rounded-xl text-xs font-bold transition text-slate-600 hover:text-slate-900"><i class="fa-solid fa-sliders mr-1"></i> Cấu hình & Kernel</button>
                <button onclick="switchTab('diagnosis')" id="nav-diagnosis" class="px-3.5 py-1.5 rounded-xl text-xs font-bold transition text-slate-600 hover:text-slate-900"><i class="fa-solid fa-stethoscope mr-1"></i> Chẩn đoán</button>
                <button onclick="switchTab('logs')" id="nav-logs" class="px-3.5 py-1.5 rounded-xl text-xs font-bold transition text-slate-600 hover:text-slate-900"><i class="fa-solid fa-database mr-1"></i> Lịch sử</button>
            </nav>

            <div class="relative">
                <button onclick="toggleDropdown()" class="px-3.5 py-2 bg-slate-900 hover:bg-slate-800 text-white font-bold rounded-xl text-xs transition flex items-center gap-2 shadow-sm"><i class="fa-solid fa-download"></i> Xuất file <i class="fa-solid fa-chevron-down text-[10px]"></i></button>
                <div id="exportDropdown" class="hidden absolute right-0 mt-2 w-48 bg-white rounded-xl shadow-xl border border-slate-100 py-2 z-50">
                    <a href="/export/json" class="block px-4 py-2.5 text-xs font-bold text-slate-700 hover:bg-pink-50"><i class="fa-solid fa-file-code mr-2 text-pink-500"></i> Tải JSON</a>
                    <a href="/export/csv" class="block px-4 py-2.5 text-xs font-bold text-slate-700 hover:bg-pink-50"><i class="fa-solid fa-file-csv mr-2 text-emerald-500"></i> Tải CSV</a>
                </div>
            </div>
        </div>
    </header>

    <main class="max-w-6xl mx-auto px-6 py-8 flex-grow w-full">
        <div id="tab-dashboard" class="tab-content active grid-cols-1 md:grid-cols-2 gap-6 items-start">
            <div class="space-y-6">
                <div class="glass-card p-6 rounded-2xl flex flex-col items-center text-center">
                    <div class="w-24 h-24 rounded-2xl overflow-hidden bg-slate-100 border-2 border-slate-200 shadow-sm mb-3">
                        <img id="resultImage" src="https://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg" class="w-full h-full object-cover">
                    </div>
                    <span id="anomalyBadge" class="text-[10px] bg-emerald-100 text-emerald-800 font-extrabold px-2.5 py-0.5 rounded-full uppercase tracking-wider mb-1">Bình thường</span>
                    <h2 id="predClass" class="text-2xl font-black text-slate-900 tracking-tight">Setosa</h2>
                    <p class="text-xs text-slate-500 mt-0.5">Độ tin cậy: <span id="predConf" class="font-extrabold text-pink-600">99.8%</span></p>
                </div>
                <div class="glass-card p-6 rounded-2xl">
                    <h3 class="text-xs font-bold text-slate-800 uppercase tracking-wider mb-3 flex items-center gap-2"><i class="fa-solid fa-chart-simple text-pink-500"></i> Phân bố xác suất 3 loài</h3>
                    <div class="h-36"><canvas id="probChart"></canvas></div>
                </div>
            </div>
            <div class="space-y-6">
                <div class="glass-card p-6 rounded-2xl h-full flex flex-col">
                    <div class="flex justify-between items-center mb-3">
                        <h3 class="text-xs font-bold text-slate-800 uppercase tracking-wider flex items-center gap-2"><i class="fa-solid fa-file-lines text-pink-500"></i> Báo cáo Chăm sóc Chuyên biệt</h3>
                        <span id="reportModelBadge" class="text-[10px] text-pink-600 bg-pink-50 px-2 py-0.5 rounded-full font-bold">SVM (linear)</span>
                    </div>
                    <div id="aiReportContent" class="text-xs text-slate-600 leading-relaxed overflow-y-auto whitespace-pre-line bg-slate-50 p-4 rounded-xl border border-slate-200 font-medium flex-grow min-h-[280px]">Hệ thống sẵn sàng. Chọn cấu hình Kernel hoặc mẫu hoa để bắt đầu.</div>
                </div>
            </div>
        </div>

        <div id="tab-config" class="tab-content grid-cols-1 md:grid-cols-2 gap-6 items-start">
            <div class="glass-card p-6 rounded-2xl space-y-4">
                <h3 class="text-xs font-extrabold text-slate-900 uppercase tracking-wider flex items-center gap-2"><i class="fa-solid fa-sliders text-pink-500"></i> Chọn SVM Kernel & Mẫu hoa</h3>
                <div class="space-y-1">
                    <label class="block text-[10px] font-extrabold text-slate-500 uppercase">Chọn Kernel Model (SVM):</label>
                    <div class="grid grid-cols-4 gap-1.5">
                        <button type="button" onclick="setKernel('linear')" id="btn-kernel-linear" class="py-2 text-xs font-extrabold rounded-xl border transition bg-pink-500 text-white border-pink-500 shadow-sm">Linear</button>
                        <button type="button" onclick="setKernel('rbf')" id="btn-kernel-rbf" class="py-2 text-xs font-bold rounded-xl border transition bg-slate-50 text-slate-700 border-slate-200 hover:bg-slate-100">RBF</button>
                        <button type="button" onclick="setKernel('poly')" id="btn-kernel-poly" class="py-2 text-xs font-bold rounded-xl border transition bg-slate-50 text-slate-700 border-slate-200 hover:bg-slate-100">Poly</button>
                        <button type="button" onclick="setKernel('sigmoid')" id="btn-kernel-sigmoid" class="py-2 text-xs font-bold rounded-xl border transition bg-slate-50 text-slate-700 border-slate-200 hover:bg-slate-100">Sigmoid</button>
                    </div>
                </div>
                <div class="grid grid-cols-3 gap-2 pt-2">
                    <button type="button" onclick="loadSample('setosa')" class="text-xs bg-pink-50 hover:bg-pink-100 text-pink-700 font-extrabold py-2 rounded-xl border border-pink-200">🌸 Setosa</button>
                    <button type="button" onclick="loadSample('versicolor')" class="text-xs bg-purple-50 hover:bg-purple-100 text-purple-700 font-extrabold py-2 rounded-xl border border-purple-200">🌷 Versicolor</button>
                    <button type="button" onclick="loadSample('virginica')" class="text-xs bg-rose-50 hover:bg-rose-100 text-rose-700 font-extrabold py-2 rounded-xl border border-rose-200">🌺 Virginica</button>
                </div>
                <form id="predictionForm" class="space-y-3 pt-1">
                    <div class="grid grid-cols-2 gap-3">
                        <div class="bg-slate-50 p-3 rounded-2xl border border-slate-200"><label class="block text-[10px] font-bold text-slate-500 uppercase mb-1">Sepal Length</label><input type="number" step="0.1" id="sepal_length" value="5.1" class="w-full bg-transparent font-extrabold text-sm focus:outline-none"></div>
                        <div class="bg-slate-50 p-3 rounded-2xl border border-slate-200"><label class="block text-[10px] font-bold text-slate-500 uppercase mb-1">Sepal Width</label><input type="number" step="0.1" id="sepal_width" value="3.5" class="w-full bg-transparent font-extrabold text-sm focus:outline-none"></div>
                        <div class="bg-slate-50 p-3 rounded-2xl border border-slate-200"><label class="block text-[10px] font-bold text-slate-500 uppercase mb-1">Petal Length</label><input type="number" step="0.1" id="petal_length" value="1.4" class="w-full bg-transparent font-extrabold text-sm focus:outline-none"></div>
                        <div class="bg-slate-50 p-3 rounded-2xl border border-slate-200"><label class="block text-[10px] font-bold text-slate-500 uppercase mb-1">Petal Width</label><input type="number" step="0.1" id="petal_width" value="0.2" class="w-full bg-transparent font-extrabold text-sm focus:outline-none"></div>
                    </div>
                    <button type="submit" class="w-full py-3.5 bg-gradient-to-r from-pink-500 to-rose-500 text-white font-extrabold rounded-2xl text-xs shadow-lg shadow-pink-500/30 transition">Chạy Phân tích Model SVM</button>
                </form>
            </div>
            <div class="glass-card p-6 rounded-2xl space-y-4">
                <h3 class="text-xs font-extrabold text-slate-900 uppercase tracking-wider flex items-center gap-2"><i class="fa-solid fa-cloud-arrow-up text-pink-500"></i> Phân tích qua File hoặc Hình ảnh</h3>
                <p class="text-xs text-slate-500">Tải lên hình ảnh hoa hoặc file để hệ thống nhận diện tự động bằng model SVM đang chọn.</p>
                <div class="border-2 border-dashed border-pink-200 hover:border-pink-400 rounded-2xl p-8 text-center cursor-pointer transition bg-pink-50/50" onclick="document.getElementById('fileInput').click()">
                    <i class="fa-solid fa-image text-pink-400 text-3xl mb-2"></i>
                    <p class="text-xs font-bold text-slate-700" id="uploadStatusText">Nhấn để tải lên file hoặc Ảnh hoa</p>
                    <input type="file" id="fileInput" accept=".csv, .json, image/*" class="hidden" onchange="handleFileUpload(event)">
                </div>
            </div>
        </div>

        <div id="tab-diagnosis" class="tab-content grid-cols-1 md:grid-cols-2 gap-6 items-start">
            <div class="glass-card p-6 rounded-2xl space-y-3">
                <h3 class="text-xs font-extrabold text-slate-900 uppercase tracking-wider flex items-center gap-2"><i class="fa-solid fa-stethoscope text-pink-500"></i> Trợ lý Chẩn đoán Bệnh</h3>
                <div class="bg-slate-50 p-2.5 rounded-2xl border border-slate-200 space-y-1.5">
                    <label class="block text-[10px] font-extrabold text-slate-600 uppercase">💬 Nhập gộp triệu chứng:</label>
                    <div class="flex gap-2">
                        <input type="text" id="customSymptomInput" placeholder="Nhập triệu chứng..." class="flex-grow bg-white border border-slate-200 rounded-xl px-3 py-2 text-xs font-bold focus:outline-none focus:ring-2 focus:ring-pink-500" onkeypress="if(event.key==='Enter') submitCustomDiagnosis()">
                        <button onclick="submitCustomDiagnosis()" class="bg-pink-500 text-white px-4 py-2 rounded-xl text-xs font-bold">Hỏi</button>
                    </div>
                </div>
                <p class="text-[10px] font-extrabold text-slate-400 uppercase tracking-wider pt-1">Hoặc chọn nhanh triệu chứng phổ biến:</p>
                <div class="space-y-1.5 max-h-[260px] overflow-y-auto pr-1">
                    <button onclick="diagnose('vàng')" class="w-full text-left p-2.5 rounded-xl border border-slate-200 hover:border-pink-400 hover:bg-pink-50/50 text-xs font-bold transition flex items-center justify-between"><span>🍂 Lá bị úa vàng, mềm nhũn hoặc thối gốc</span> <i class="fa-solid fa-chevron-right text-[10px] text-slate-400"></i></button>
                    <button onclick="diagnose('đốm')" class="w-full text-left p-2.5 rounded-xl border border-slate-200 hover:border-pink-400 hover:bg-pink-50/50 text-xs font-bold transition flex items-center justify-between"><span>🦠 Xuất hiện đốm nâu hoặc đen trên lá</span> <i class="fa-solid fa-chevron-right text-[10px] text-slate-400"></i></button>
                    <button onclick="diagnose('sâu')" class="w-full text-left p-2.5 rounded-xl border border-slate-200 hover:border-pink-400 hover:bg-pink-50/50 text-xs font-bold transition flex items-center justify-between"><span>🐛 Có sâu ăn lá hoặc thân bị đục</span> <i class="fa-solid fa-chevron-right text-[10px] text-slate-400"></i></button>
                    <button onclick="diagnose('gỉ')" class="w-full text-left p-2.5 rounded-xl border border-slate-200 hover:border-pink-400 hover:bg-pink-50/50 text-xs font-bold transition flex items-center justify-between"><span>🟠 Bệnh gỉ sắt (mụn cam/đỏ trên lá)</span> <i class="fa-solid fa-chevron-right text-[10px] text-slate-400"></i></button>
                    <button onclick="diagnose('thối')" class="w-full text-left p-2.5 rounded-xl border border-slate-200 hover:border-pink-400 hover:bg-pink-50/50 text-xs font-bold transition flex items-center justify-between"><span>💧 Bệnh thối mềm củ rễ (Soft Rot)</span> <i class="fa-solid fa-chevron-right text-[10px] text-slate-400"></i></button>
                    <button onclick="diagnose('hoa')" class="w-full text-left p-2.5 rounded-xl border border-slate-200 hover:border-pink-400 hover:bg-pink-50/50 text-xs font-bold transition flex items-center justify-between"><span>🌸 Cây phát triển tốt nhưng không ra hoa</span> <i class="fa-solid fa-chevron-right text-[10px] text-slate-400"></i></button>
                </div>
            </div>
            <div class="glass-card p-6 rounded-2xl h-full flex flex-col">
                <h3 class="text-xs font-bold text-slate-800 uppercase tracking-wider mb-3 flex items-center gap-2"><i class="fa-solid fa-clipboard-medical text-pink-500"></i> Kết quả Chẩn đoán</h3>
                <div id="diagnosisResult" class="text-xs text-slate-600 leading-relaxed overflow-y-auto whitespace-pre-line bg-slate-50 p-4 rounded-xl border border-slate-200 font-medium flex-grow min-h-[280px]">Chọn hoặc nhập triệu chứng bên trái.</div>
            </div>
        </div>

        <div id="tab-logs" class="tab-content grid-cols-1 gap-6">
            <div class="glass-card p-6 rounded-2xl">
                <div class="flex justify-between items-center mb-4">
                    <h3 class="text-xs font-extrabold text-slate-900 uppercase tracking-wider flex items-center gap-2"><i class="fa-solid fa-database text-pink-500"></i> Lịch sử dự đoán (Ghi nhận Model Kernel)</h3>
                    <button onclick="loadLogs()" class="text-xs text-pink-600 font-bold hover:underline"><i class="fa-solid fa-rotate-right"></i> Làm mới</button>
                </div>
                <div class="overflow-x-auto max-h-[400px]">
                    <table class="w-full text-left text-xs">
                        <thead><tr class="border-b border-slate-200 text-slate-400 font-bold uppercase sticky top-0 bg-white"><th class="pb-2.5">Thời gian</th><th class="pb-2.5">Model SVM</th><th class="pb-2.5">Thông số</th><th class="pb-2.5">Dự đoán</th><th class="pb-2.5">Độ tin cậy</th></tr></thead>
                        <tbody id="logsTableBody" class="divide-y divide-slate-100 font-semibold text-slate-700"><tr><td colspan="5" class="py-4 text-center text-slate-400">Đang tải...</td></tr></tbody>
                    </table>
                </div>
            </div>
        </div>
    </main>

    <footer class="text-center py-4 text-xs text-slate-400 font-medium border-t border-slate-200 bg-white">Iris Botanical Enterprise Suite &bull; FastAPI & Scikit-Learn SVM Multi-Kernel</footer>

    <script>
        let selectedKernel = 'linear';
        let allLogs = [];
        const ctxProb = document.getElementById('probChart').getContext('2d');
        const probChart = new Chart(ctxProb, {
            type: 'bar',
            data: { labels: ['Setosa', 'Versicolor', 'Virginica'], datasets: [{ data: [100, 0, 0], backgroundColor: ['#ec4899', '#a855f7', '#3b82f6'], borderRadius: 6 }] },
            options: { plugins: { legend: { display: false } }, scales: { y: { max: 100, ticks: { display: false }, grid: { display: false } }, x: { grid: { display: false }, ticks: { font: { size: 11, weight: 'bold' } } } } }
        });

        function setKernel(kernel) {
            selectedKernel = kernel;
            ['linear', 'rbf', 'poly', 'sigmoid'].forEach(k => {
                const btn = document.getElementById('btn-kernel-' + k);
                if(k === kernel) {
                    btn.className = "py-2 text-xs font-extrabold rounded-xl border transition bg-pink-500 text-white border-pink-500 shadow-sm";
                } else {
                    btn.className = "py-2 text-xs font-bold rounded-xl border transition bg-slate-50 text-slate-700 border-slate-200 hover:bg-slate-100";
                }
            });
            document.getElementById('activeModelBadge').innerHTML = `<span class="w-1.5 h-1.5 rounded-full bg-pink-500"></span> SVM Model: ${kernel.toUpperCase()}`;
        }

        function switchTab(tabId) {
            document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
            document.getElementById('tab-' + tabId).classList.add('active');
            ['dashboard', 'config', 'diagnosis', 'logs'].forEach(id => {
                const btn = document.getElementById('nav-' + id);
                btn.className = (id === tabId) ? "px-3.5 py-1.5 rounded-xl text-xs font-extrabold transition bg-white text-pink-600 shadow-sm" : "px-3.5 py-1.5 rounded-xl text-xs font-bold transition text-slate-600 hover:text-slate-900";
            });
            if(tabId === 'logs') loadLogs();
        }

        function toggleDropdown() { document.getElementById('exportDropdown').classList.toggle('hidden'); }

        function loadSample(type) {
            const samples = { 'setosa': [5.1, 3.5, 1.4, 0.2], 'versicolor': [6.0, 2.9, 4.5, 1.5], 'virginica': [6.5, 3.0, 5.8, 2.2] };
            const val = samples[type];
            document.getElementById('sepal_length').value = val[0];
            document.getElementById('sepal_width').value = val[1];
            document.getElementById('petal_length').value = val[2];
            document.getElementById('petal_width').value = val[3];
            document.getElementById('predictionForm').dispatchEvent(new Event('submit'));
        }

        const sampleImages = { 'Setosa': 'https://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg', 'Versicolor': 'https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg', 'Virginica': 'https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg' };

        async function loadLogs() {
            try {
                let res = await fetch('/logs');
                allLogs = await res.json();
                let tbody = document.getElementById('logsTableBody');
                if(!Array.isArray(allLogs) || allLogs.length === 0) { tbody.innerHTML = '<tr><td colspan="5" class="py-4 text-center text-slate-400">Chưa có bản ghi.</td></tr>'; return; }
                tbody.innerHTML = allLogs.map(item => `
                    <tr class="hover:bg-slate-50 transition">
                        <td class="py-2.5 text-slate-500 font-normal">${item.timestamp}</td>
                        <td class="py-2.5 font-bold text-pink-600">${item.model_used}</td>
                        <td class="py-2.5 font-mono">${item.sepal_length} / ${item.sepal_width} / ${item.petal_length} / ${item.petal_width}</td>
                        <td class="py-2.5 text-slate-900 font-extrabold">${item.prediction}</td>
                        <td class="py-2.5 text-emerald-600 font-bold">${(item.confidence * 100).toFixed(1)}%</td>
                    </tr>
                `).join('');
            } catch(e) { console.error(e); }
        }

        async function diagnose(symptom) {
            document.getElementById('diagnosisResult').innerText = "⏳ Đang phân tích...";
            let res = await fetch('/diagnose', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ symptom }) });
            let data = await res.json();
            document.getElementById('diagnosisResult').innerHTML = `<strong class="text-pink-600 text-sm block mb-2">🔍 ${data.disease}</strong><p class="font-semibold">${data.solution}</p>`;
        }

        function submitCustomDiagnosis() {
            let val = document.getElementById('customSymptomInput').value.trim();
            if(val) diagnose(val);
        }

        function updateUI(result) {
            document.getElementById('predClass').innerText = result.prediction;
            document.getElementById('predConf').innerText = (result.confidence * 100).toFixed(1) + '%';
            document.getElementById('reportModelBadge').innerText = result.model_used;
            if(sampleImages[result.prediction]) document.getElementById('resultImage').src = sampleImages[result.prediction];
            if(result.probabilities) { probChart.data.datasets[0].data = result.probabilities; probChart.update(); }
            if(result.ai_report) document.getElementById('aiReportContent').innerText = result.ai_report;
        }

        document.getElementById('predictionForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            const data = {
                sepal_length: parseFloat(document.getElementById('sepal_length').value),
                sepal_width: parseFloat(document.getElementById('sepal_width').value),
                petal_length: parseFloat(document.getElementById('petal_length').value),
                petal_width: parseFloat(document.getElementById('petal_width').value),
                kernel: selectedKernel
            };
            let res = await fetch('/predict', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(data) });
            let result = await res.json();
            updateUI(result);
            switchTab('dashboard');
        });

        async function handleFileUpload(event) {
            const file = event.target.files[0];
            if(!file) return;
            const formData = new FormData();
            formData.append("file", file);
            let res = await fetch('/analyze-file', { method: 'POST', body: formData });
            let data = await res.json();
            if(data.status === "SUCCESS") {
                updateUI(data.prediction_result);
                switchTab('dashboard');
            }
        }
    </script>
</body>
</html>
    """
