import os
import sqlite3
import csv
import io
import json
import hashlib
from datetime import datetime
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException, Response, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from sklearn.datasets import load_iris
from sklearn.svm import SVC

app = FastAPI(title="Iris Botanica — Personal Studio", version="21.0")

DB_FILE = "iris_system.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            username TEXT,
            model_used TEXT,
            sepal_length REAL,
            sepal_width REAL,
            petal_length REAL,
            petal_width REAL,
            prediction TEXT,
            confidence REAL,
            is_anomaly INTEGER
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT
        )
    ''')
    # migrate old table if needed
    try:
        cursor.execute("ALTER TABLE predictions ADD COLUMN username TEXT")
    except:
        pass
    conn.commit()
    conn.close()

init_db()

AVAILABLE_KERNELS = ['linear', 'rbf', 'poly', 'sigmoid']

def get_svm_model(kernel_name: str):
    model_file = f"svm_{kernel_name}_model.pkl"
    if not os.path.exists(model_file):
        iris = load_iris()
        model = SVC(kernel=kernel_name, probability=True, random_state=42)
        model.fit(iris.data, iris.target)
        joblib.dump(model, model_file)
    return joblib.load(model_file)

target_names = ['Setosa', 'Versicolor', 'Virginica']
latest_predicted_species = "Setosa"

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    kernel: str = "linear"
    username: str = "guest"

class DiagnosisInput(BaseModel):
    symptom: str

class UserAuthInput(BaseModel):
    username: str
    password: str

def hash_pw(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

@app.post("/api/register")
def register_user(data: UserAuthInput):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO users (username, password) VALUES (?, ?)",
            (data.username.strip(), hash_pw(data.password))
        )
        conn.commit()
        conn.close()
        return {"status": "SUCCESS", "message": "Đăng ký thành công!"}
    except sqlite3.IntegrityError:
        conn.close()
        raise HTTPException(status_code=400, detail="Tên đăng nhập đã tồn tại!")

@app.post("/api/login")
def login_user(data: UserAuthInput):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM users WHERE username = ? AND password = ?",
        (data.username.strip(), hash_pw(data.password))
    )
    user = cursor.fetchone()
    conn.close()
    if user:
        return {"status": "SUCCESS", "message": "Đăng nhập thành công!", "username": user[1]}
    raise HTTPException(status_code=400, detail="Sai tên đăng nhập hoặc mật khẩu!")

@app.post("/predict")
def predict_iris(data: IrisInput):
    global latest_predicted_species
    kernel = data.kernel if data.kernel in AVAILABLE_KERNELS else "linear"
    model = get_svm_model(kernel)
    features = np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])

    probs = model.predict_proba(features)[0]
    pred_idx = np.argmax(probs)
    prediction = target_names[pred_idx]
    confidence = float(probs[pred_idx])

    latest_predicted_species = prediction
    probabilities = [round(float(p) * 100, 1) for p in probs]
    is_anomaly = 1 if (data.petal_length < 1.0 or data.petal_length > 7.5 or data.sepal_length < 4.0) else 0

    species_reports = {
        'Setosa': f"""• Đặc điểm nổi bật: Hoa nhỏ, cánh mỏng, màu tím-xanh nhẹ nhàng. Chịu lạnh rất tốt.
• Đất & pH: Đất thịt nhẹ, giàu mùn, hơi chua (pH 6.0–6.5).
• Chăm sóc: Giữ ẩm bề mặt đều, tránh úng. Tỉa lá già định kỳ.
• Ánh sáng: Nắng nhẹ buổi sáng, tránh nắng gắt trưa hè.""",
        'Versicolor': f"""• Đặc điểm nổi bật: Hoa lớn hơn Setosa, cánh có vân rõ, màu tím-xanh đậm.
• Đất & pH: Đất sét bùn, nhiều hữu cơ (pH 5.5–7.0).
• Chăm sóc: Ưa nước, chịu ngập nhẹ. Bón phân hữu cơ 2 lần/năm.
• Ánh sáng: Nắng gián tiếp hoặc nửa ngày.""",
        'Virginica': f"""• Đặc điểm nổi bật: Hoa to, cánh dày, màu tím đậm hoặc trắng-tím. Rất khỏe.
• Đất & pH: Đất phù sa, thoát nước tốt (pH 6.5–7.5).
• Chăm sóc: Ưa nắng toàn phần 6–8 giờ/ngày. Tưới sâu, để khô nhẹ giữa các lần.
• Ánh sáng: Nắng đầy đủ là điều kiện ra hoa tốt nhất."""
    }
    ai_report = species_reports.get(prediction, "Chăm sóc theo tiêu chuẩn sinh học chung.")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO predictions (timestamp, username, model_used, sepal_length, sepal_width, petal_length, petal_width, prediction, confidence, is_anomaly)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (timestamp, data.username, f"SVM ({kernel})", data.sepal_length, data.sepal_width,
          data.petal_length, data.petal_width, prediction, confidence, is_anomaly))
    conn.commit()
    conn.close()

    return {
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": probabilities,
        "is_anomaly": bool(is_anomaly),
        "ai_report": ai_report,
        "model_used": f"SVM ({kernel})"
    }

@app.post("/compare")
def compare_kernels(data: IrisInput):
    results = []
    features = np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
    for k in AVAILABLE_KERNELS:
        model = get_svm_model(k)
        probs = model.predict_proba(features)[0]
        pred_idx = np.argmax(probs)
        results.append({
            "kernel": k.upper(),
            "prediction": target_names[pred_idx],
            "confidence": round(float(probs[pred_idx]) * 100, 1),
            "probabilities": [round(float(p) * 100, 1) for p in probs]
        })
    return {"results": results}

@app.post("/diagnose")
def diagnose_plant(data: DiagnosisInput):
    global latest_predicted_species
    query = data.symptom.lower().strip()

    knowledge_base = {
        "sâu": {"name": "Sâu hại ăn lá & sâu đục thân", "solution": "Bắt thủ công vào buổi sáng. Phun Bt hoặc dung dịch tỏi-ớt (1 củ tỏi + 3 quả ớt xay, ngâm 24h)."},
        "vàng": {"name": "Vàng lá do úng / thối rễ", "solution": f"Ngừng tưới ngay. Kiểm tra lỗ thoát nước cho giống {latest_predicted_species}. Bổ sung Trichoderma."},
        "đốm": {"name": "Đốm lá vi khuẩn / nấm", "solution": "Cắt bỏ lá bệnh, hạn chế tưới phun ban đêm. Phun thuốc gốc đồng hoặc Mancozeb."},
        "gỉ": {"name": "Bệnh gỉ sắt", "solution": "Tiêu hủy lá nặng. Phun thuốc chứa Mancozeb hoặc Propiconazole."},
        "thối": {"name": "Thối mềm củ rễ (Soft Rot)", "solution": "Đào củ, cắt phần nhũn, sát trùng bằng vôi bột hoặc dung dịch đồng."},
        "hoa": {"name": "Không ra hoa", "solution": f"Giống {latest_predicted_species} cần đủ nắng (6–8h). Tăng lân + kali, giảm đạm."}
    }
    matched = [info for kw, info in knowledge_base.items() if kw in query]
    if not matched:
        return {
            "disease": f"Phân tích cho giống {latest_predicted_species}",
            "solution": f"Triệu chứng “{data.symptom}”: Kiểm tra độ ẩm đất, ánh sáng và dinh dưỡng phù hợp với giống {latest_predicted_species}."
        }
    return {
        "disease": f"Phát hiện {len(matched)} vấn đề cho giống {latest_predicted_species}: " + " + ".join([d["name"] for d in matched]),
        "solution": "\n\n".join([f"• {d['name']}:\n{d['solution']}" for d in matched])
    }

@app.post("/analyze-file")
async def analyze_file(file: UploadFile = File(...)):
    filename = file.filename.lower()
    contents = await file.read()
    extracted_metrics = {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2, "kernel": "linear"}
    image_preview_url = None
    try:
        if filename.endswith((".png", ".jpg", ".jpeg", ".webp")):
            import base64
            encoded = base64.b64encode(contents).decode('utf-8')
            image_preview_url = f"data:image/jpeg;base64,{encoded}"
            hash_val = sum(contents) % 3
            if hash_val == 1:
                extracted_metrics = {"sepal_length": 6.0, "sepal_width": 2.9, "petal_length": 4.5, "petal_width": 1.5, "kernel": "linear"}
            elif hash_val == 2:
                extracted_metrics = {"sepal_length": 6.9, "sepal_width": 3.1, "petal_length": 5.4, "petal_width": 2.1, "kernel": "linear"}
        res = predict_iris(IrisInput(**extracted_metrics, username="guest"))
        return {"status": "SUCCESS", "extracted_metrics": extracted_metrics, "image_preview": image_preview_url, "prediction_result": res}
    except Exception as e:
        return {"status": "ERROR", "message": str(e)}

@app.get("/logs")
def get_logs(username: str = None):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    if username and username != "guest":
        cursor.execute("""
            SELECT id, timestamp, model_used, sepal_length, sepal_width, petal_length, petal_width, prediction, confidence, is_anomaly
            FROM predictions WHERE username = ? ORDER BY id DESC
        """, (username,))
    else:
        cursor.execute("""
            SELECT id, timestamp, model_used, sepal_length, sepal_width, petal_length, petal_width, prediction, confidence, is_anomaly
            FROM predictions ORDER BY id DESC LIMIT 100
        """)
    rows = cursor.fetchall()
    conn.close()
    return [{
        "id": r[0], "timestamp": r[1], "model_used": r[2],
        "sepal_length": r[3], "sepal_width": r[4], "petal_length": r[5], "petal_width": r[6],
        "prediction": r[7], "confidence": r[8], "is_anomaly": bool(r[9])
    } for r in rows]

@app.get("/stats")
def get_stats(username: str = None):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    if username and username != "guest":
        cursor.execute("SELECT prediction, confidence, is_anomaly FROM predictions WHERE username = ?", (username,))
    else:
        cursor.execute("SELECT prediction, confidence, is_anomaly FROM predictions")
    rows = cursor.fetchall()
    conn.close()
    if not rows:
        return {"total": 0, "by_species": {}, "avg_confidence": 0, "anomaly_rate": 0}
    total = len(rows)
    by_species = {}
    conf_sum = 0
    anomaly = 0
    for pred, conf, is_a in rows:
        by_species[pred] = by_species.get(pred, 0) + 1
        conf_sum += conf
        if is_a:
            anomaly += 1
    return {
        "total": total,
        "by_species": by_species,
        "avg_confidence": round(conf_sum / total * 100, 1),
        "anomaly_rate": round(anomaly / total * 100, 1)
    }

@app.get("/export/{format_type}")
def export_data(format_type: str, username: str = None):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    if username and username != "guest":
        cursor.execute("SELECT * FROM predictions WHERE username = ?", (username,))
    else:
        cursor.execute("SELECT * FROM predictions")
    rows = cursor.fetchall()
    conn.close()
    if format_type == "json":
        data = [{"id": r[0], "timestamp": r[1], "user": r[2], "model": r[3], "prediction": r[8]} for r in rows]
        return JSONResponse(content=data)
    elif format_type == "csv":
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["ID", "Timestamp", "User", "Model", "Sepal L", "Sepal W", "Petal L", "Petal W", "Prediction", "Confidence", "Anomaly"])
        writer.writerows(rows)
        return Response(output.getvalue(), media_type="text/csv",
                        headers={"Content-Disposition": "attachment; filename=iris_export.csv"})
    raise HTTPException(status_code=400, detail="Không hỗ trợ.")

@app.get("/", response_class=HTMLResponse)
def get_dashboard():
    return """
<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Iris Botanica — Personal Studio</title>
<script src="https://cdn.tailwindcss.com"></script>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=Playfair+Display:wght@600;700&display=swap');
  :root {
    --bg: #f7f4ef;
    --card: #ffffff;
    --ink: #1c1917;
    --muted: #78716c;
    --accent: #3f6212;
    --accent-soft: #ecfccb;
    --border: #e7e5e4;
    --danger: #b91c1c;
  }
  .dark {
    --bg: #1c1917;
    --card: #292524;
    --ink: #fafaf9;
    --muted: #a8a29e;
    --accent: #a3e635;
    --accent-soft: #365314;
    --border: #44403c;
  }
  body {
    font-family: 'DM Sans', system-ui, sans-serif;
    background: var(--bg);
    color: var(--ink);
    min-height: 100vh;
    transition: background .3s, color .3s;
  }
  .font-display { font-family: 'Playfair Display', serif; }
  .card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 1.25rem;
    box-shadow: 0 1px 3px rgba(0,0,0,.04);
  }
  .tab-content { display: none; }
  .tab-content.active { display: grid; }
  .nav-btn.active {
    background: var(--accent-soft);
    color: var(--accent);
    font-weight: 600;
  }
  .kernel-btn.active {
    background: var(--accent);
    color: white;
    border-color: var(--accent);
  }
  .toast {
    position: fixed; bottom: 1.5rem; right: 1.5rem;
    background: var(--ink); color: var(--bg);
    padding: .75rem 1.25rem; border-radius: .75rem;
    font-size: .8rem; font-weight: 600;
    z-index: 100; transform: translateY(120%);
    transition: transform .35s cubic-bezier(.2,.8,.2,1);
  }
  .toast.show { transform: translateY(0); }
  input, select, button { transition: all .15s; }
  .species-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
</style>
</head>
<body class="flex flex-col">
  <!-- AUTH MODAL -->
  <div id="authModal" class="fixed inset-0 z-50 bg-stone-900/50 backdrop-blur-sm flex items-center justify-center p-4">
    <div class="card max-w-md w-full p-8 space-y-6">
      <div class="text-center space-y-2">
        <div class="w-14 h-14 rounded-2xl bg-[var(--accent)] text-white flex items-center justify-center mx-auto text-xl">
          <i class="fa-solid fa-leaf"></i>
        </div>
        <h2 id="authTitle" class="font-display text-2xl font-bold">Chào mừng trở lại</h2>
        <p class="text-sm text-[var(--muted)]">Iris Botanica — Personal Studio</p>
      </div>
      <div id="authError" class="hidden bg-red-50 dark:bg-red-900/30 border border-red-200 text-red-700 text-xs font-medium p-3 rounded-xl text-center"></div>
      <form id="authForm" onsubmit="handleAuth(event)" class="space-y-4">
        <div>
          <label class="block text-[11px] font-semibold text-[var(--muted)] uppercase tracking-wider mb-1.5">Tên đăng nhập</label>
          <input type="text" id="authUsername" required class="w-full bg-[var(--bg)] border border-[var(--border)] rounded-xl px-4 py-3 text-sm font-medium focus:outline-none focus:ring-2 focus:ring-[var(--accent)]">
        </div>
        <div>
          <label class="block text-[11px] font-semibold text-[var(--muted)] uppercase tracking-wider mb-1.5">Mật khẩu</label>
          <input type="password" id="authPassword" required class="w-full bg-[var(--bg)] border border-[var(--border)] rounded-xl px-4 py-3 text-sm font-medium focus:outline-none focus:ring-2 focus:ring-[var(--accent)]">
        </div>
        <button type="submit" id="authSubmitBtn" class="w-full py-3.5 bg-[var(--accent)] text-white font-semibold rounded-xl text-sm hover:opacity-90">Đăng nhập</button>
      </form>
      <div class="text-center">
        <button type="button" onclick="toggleAuthMode()" id="authSwitchBtn" class="text-sm font-medium text-[var(--accent)] hover:underline">Chưa có tài khoản? Đăng ký</button>
      </div>
    </div>
  </div>

  <!-- HEADER -->
  <header class="sticky top-0 z-40 border-b border-[var(--border)] bg-[var(--card)]/90 backdrop-blur-md px-5 py-3">
    <div class="max-w-6xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-3">
      <div class="flex items-center gap-3">
        <div class="w-10 h-10 rounded-xl bg-[var(--accent)] text-white flex items-center justify-center">
          <i class="fa-solid fa-leaf text-sm"></i>
        </div>
        <div>
          <h1 class="font-display font-bold text-lg leading-tight">Iris Botanica</h1>
          <p class="text-[11px] text-[var(--muted)]">Personal Studio · <span id="userGreeting">Guest</span></p>
        </div>
      </div>
      <nav class="flex items-center bg-[var(--bg)] p-1 rounded-2xl gap-0.5 flex-wrap justify-center">
        <button onclick="switchTab('dashboard')" id="nav-dashboard" class="nav-btn active px-3.5 py-1.5 rounded-xl text-xs font-medium transition">Dashboard</button>
        <button onclick="switchTab('config')" id="nav-config" class="nav-btn px-3.5 py-1.5 rounded-xl text-xs font-medium transition text-[var(--muted)]">Phân tích</button>
        <button onclick="switchTab('compare')" id="nav-compare" class="nav-btn px-3.5 py-1.5 rounded-xl text-xs font-medium transition text-[var(--muted)]">So sánh</button>
        <button onclick="switchTab('diagnosis')" id="nav-diagnosis" class="nav-btn px-3.5 py-1.5 rounded-xl text-xs font-medium transition text-[var(--muted)]">Chẩn đoán</button>
        <button onclick="switchTab('logs')" id="nav-logs" class="nav-btn px-3.5 py-1.5 rounded-xl text-xs font-medium transition text-[var(--muted)]">Lịch sử</button>
        <button onclick="switchTab('stats')" id="nav-stats" class="nav-btn px-3.5 py-1.5 rounded-xl text-xs font-medium transition text-[var(--muted)]">Thống kê</button>
      </nav>
      <div class="flex items-center gap-2">
        <button onclick="toggleTheme()" class="w-9 h-9 rounded-xl border border-[var(--border)] flex items-center justify-center text-[var(--muted)] hover:text-[var(--ink)]" title="Đổi giao diện">
          <i class="fa-solid fa-moon text-sm" id="themeIcon"></i>
        </button>
        <button onclick="logout()" class="px-3 py-2 text-xs font-medium text-red-600 hover:bg-red-50 dark:hover:bg-red-900/20 rounded-xl transition">Đăng xuất</button>
      </div>
    </div>
  </header>

  <main class="max-w-6xl mx-auto px-5 py-8 flex-grow w-full">
    <!-- DASHBOARD -->
    <div id="tab-dashboard" class="tab-content active grid-cols-1 lg:grid-cols-5 gap-6 items-start">
      <div class="lg:col-span-2 space-y-5">
        <div class="card p-6 flex flex-col items-center text-center">
          <div class="w-28 h-28 rounded-2xl overflow-hidden border border-[var(--border)] mb-4 shadow-sm">
            <img id="resultImage" src="https://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg" class="w-full h-full object-cover">
          </div>
          <span id="anomalyBadge" class="text-[10px] font-semibold px-2.5 py-0.5 rounded-full bg-emerald-100 text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-300 mb-2">Bình thường</span>
          <h2 id="predClass" class="font-display text-3xl font-bold tracking-tight">Setosa</h2>
          <p class="text-sm text-[var(--muted)] mt-1">Độ tin cậy <span id="predConf" class="font-semibold text-[var(--accent)]">—</span></p>
          <p class="text-[11px] text-[var(--muted)] mt-2" id="reportModelBadge">SVM (linear)</p>
        </div>
        <div class="card p-5">
          <h3 class="text-xs font-semibold uppercase tracking-wider text-[var(--muted)] mb-3">Phân bố xác suất</h3>
          <div class="h-40"><canvas id="probChart"></canvas></div>
        </div>
      </div>
      <div class="lg:col-span-3">
        <div class="card p-6 h-full flex flex-col">
          <div class="flex items-center justify-between mb-4">
            <h3 class="text-xs font-semibold uppercase tracking-wider text-[var(--muted)]">Báo cáo chăm sóc</h3>
            <button onclick="window.print()" class="text-xs font-medium text-[var(--accent)] hover:underline">In báo cáo</button>
          </div>
          <div id="aiReportContent" class="text-sm leading-relaxed whitespace-pre-line text-[var(--ink)] flex-grow min-h-[260px] bg-[var(--bg)] rounded-xl p-5 border border-[var(--border)]">
            Hệ thống sẵn sàng. Chọn mẫu hoa hoặc nhập thông số để bắt đầu.
          </div>
        </div>
      </div>
    </div>

    <!-- CONFIG / PHÂN TÍCH -->
    <div id="tab-config" class="tab-content grid-cols-1 md:grid-cols-2 gap-6 items-start">
      <div class="card p-6 space-y-5">
        <h3 class="font-display text-lg font-semibold">Thông số & Kernel</h3>
        <div>
          <label class="block text-[11px] font-semibold text-[var(--muted)] uppercase mb-2">Kernel SVM</label>
          <div class="grid grid-cols-4 gap-1.5">
            <button type="button" onclick="setKernel('linear')" id="btn-kernel-linear" class="kernel-btn active py-2 text-xs font-semibold rounded-xl border border-[var(--border)]">Linear</button>
            <button type="button" onclick="setKernel('rbf')" id="btn-kernel-rbf" class="kernel-btn py-2 text-xs font-medium rounded-xl border border-[var(--border)]">RBF</button>
            <button type="button" onclick="setKernel('poly')" id="btn-kernel-poly" class="kernel-btn py-2 text-xs font-medium rounded-xl border border-[var(--border)]">Poly</button>
            <button type="button" onclick="setKernel('sigmoid')" id="btn-kernel-sigmoid" class="kernel-btn py-2 text-xs font-medium rounded-xl border border-[var(--border)]">Sigmoid</button>
          </div>
        </div>
        <div class="grid grid-cols-3 gap-2">
          <button type="button" onclick="loadSample('setosa')" class="text-xs py-2.5 rounded-xl border border-[var(--border)] hover:bg-[var(--accent-soft)] font-medium">Setosa</button>
          <button type="button" onclick="loadSample('versicolor')" class="text-xs py-2.5 rounded-xl border border-[var(--border)] hover:bg-[var(--accent-soft)] font-medium">Versicolor</button>
          <button type="button" onclick="loadSample('virginica')" class="text-xs py-2.5 rounded-xl border border-[var(--border)] hover:bg-[var(--accent-soft)] font-medium">Virginica</button>
        </div>
        <form id="predictionForm" class="space-y-3">
          <div class="grid grid-cols-2 gap-3">
            <div class="bg-[var(--bg)] p-3 rounded-xl border border-[var(--border)]">
              <label class="block text-[10px] font-semibold text-[var(--muted)] uppercase mb-1">Sepal Length</label>
              <input type="number" step="0.1" id="sepal_length" value="5.1" class="w-full bg-transparent font-semibold text-sm focus:outline-none">
            </div>
            <div class="bg-[var(--bg)] p-3 rounded-xl border border-[var(--border)]">
              <label class="block text-[10px] font-semibold text-[var(--muted)] uppercase mb-1">Sepal Width</label>
              <input type="number" step="0.1" id="sepal_width" value="3.5" class="w-full bg-transparent font-semibold text-sm focus:outline-none">
            </div>
            <div class="bg-[var(--bg)] p-3 rounded-xl border border-[var(--border)]">
              <label class="block text-[10px] font-semibold text-[var(--muted)] uppercase mb-1">Petal Length</label>
              <input type="number" step="0.1" id="petal_length" value="1.4" class="w-full bg-transparent font-semibold text-sm focus:outline-none">
            </div>
            <div class="bg-[var(--bg)] p-3 rounded-xl border border-[var(--border)]">
              <label class="block text-[10px] font-semibold text-[var(--muted)] uppercase mb-1">Petal Width</label>
              <input type="number" step="0.1" id="petal_width" value="0.2" class="w-full bg-transparent font-semibold text-sm focus:outline-none">
            </div>
          </div>
          <button type="submit" class="w-full py-3.5 bg-[var(--accent)] text-white font-semibold rounded-xl text-sm hover:opacity-90">Chạy phân tích</button>
        </form>
      </div>
      <div class="card p-6 space-y-4">
        <h3 class="font-display text-lg font-semibold">Tải ảnh / file</h3>
        <p class="text-sm text-[var(--muted)]">Hệ thống sẽ ước lượng thông số từ ảnh (demo) hoặc đọc file CSV/JSON.</p>
        <div class="border-2 border-dashed border-[var(--border)] hover:border-[var(--accent)] rounded-2xl p-10 text-center cursor-pointer transition bg-[var(--bg)]" onclick="document.getElementById('fileInput').click()">
          <i class="fa-solid fa-cloud-arrow-up text-2xl text-[var(--muted)] mb-2"></i>
          <p class="text-sm font-medium" id="uploadStatusText">Chọn ảnh hoa hoặc file</p>
          <input type="file" id="fileInput" accept=".csv,.json,image/*" class="hidden" onchange="handleFileUpload(event)">
        </div>
      </div>
    </div>

    <!-- SO SÁNH KERNEL -->
    <div id="tab-compare" class="tab-content grid-cols-1 gap-6">
      <div class="card p-6">
        <div class="flex flex-col sm:flex-row sm:items-end justify-between gap-4 mb-6">
          <div>
            <h3 class="font-display text-lg font-semibold">So sánh 4 Kernel</h3>
            <p class="text-sm text-[var(--muted)] mt-1">Cùng một bộ thông số, xem sự khác biệt giữa các kernel</p>
          </div>
          <button onclick="runCompare()" class="px-5 py-2.5 bg-[var(--accent)] text-white text-sm font-semibold rounded-xl hover:opacity-90">Chạy so sánh</button>
        </div>
        <div id="compareResult" class="overflow-x-auto">
          <p class="text-sm text-[var(--muted)]">Nhấn “Chạy so sánh” để xem kết quả.</p>
        </div>
      </div>
    </div>

    <!-- CHẨN ĐOÁN -->
    <div id="tab-diagnosis" class="tab-content grid-cols-1 md:grid-cols-2 gap-6 items-start">
      <div class="card p-6 space-y-4">
        <h3 class="font-display text-lg font-semibold">Trợ lý chẩn đoán</h3>
        <div class="flex gap-2">
          <input type="text" id="customSymptomInput" placeholder="Mô tả triệu chứng..." class="flex-grow bg-[var(--bg)] border border-[var(--border)] rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-[var(--accent)]" onkeypress="if(event.key==='Enter') submitCustomDiagnosis()">
          <button onclick="submitCustomDiagnosis()" class="px-4 py-2.5 bg-[var(--accent)] text-white text-sm font-semibold rounded-xl">Hỏi</button>
        </div>
        <div class="space-y-1.5 max-h-[280px] overflow-y-auto">
          <button onclick="diagnose('vàng')" class="w-full text-left p-3 rounded-xl border border-[var(--border)] hover:bg-[var(--accent-soft)] text-sm font-medium transition">Lá úa vàng / thối gốc</button>
          <button onclick="diagnose('đốm')" class="w-full text-left p-3 rounded-xl border border-[var(--border)] hover:bg-[var(--accent-soft)] text-sm font-medium transition">Đốm nâu / đen trên lá</button>
          <button onclick="diagnose('sâu')" class="w-full text-left p-3 rounded-xl border border-[var(--border)] hover:bg-[var(--accent-soft)] text-sm font-medium transition">Sâu ăn lá / đục thân</button>
          <button onclick="diagnose('gỉ')" class="w-full text-left p-3 rounded-xl border border-[var(--border)] hover:bg-[var(--accent-soft)] text-sm font-medium transition">Gỉ sắt (mụn cam/đỏ)</button>
          <button onclick="diagnose('thối')" class="w-full text-left p-3 rounded-xl border border-[var(--border)] hover:bg-[var(--accent-soft)] text-sm font-medium transition">Thối mềm củ rễ</button>
          <button onclick="diagnose('hoa')" class="w-full text-left p-3 rounded-xl border border-[var(--border)] hover:bg-[var(--accent-soft)] text-sm font-medium transition">Không ra hoa</button>
        </div>
      </div>
      <div class="card p-6 h-full flex flex-col">
        <h3 class="text-xs font-semibold uppercase tracking-wider text-[var(--muted)] mb-3">Kết quả</h3>
        <div id="diagnosisResult" class="text-sm leading-relaxed whitespace-pre-line bg-[var(--bg)] p-5 rounded-xl border border-[var(--border)] flex-grow min-h-[260px]">
          Chọn hoặc nhập triệu chứng bên trái.
        </div>
      </div>
    </div>

    <!-- LỊCH SỬ -->
    <div id="tab-logs" class="tab-content grid-cols-1 gap-6">
      <div class="card p-6">
        <div class="flex justify-between items-center mb-5">
          <h3 class="font-display text-lg font-semibold">Lịch sử dự đoán</h3>
          <div class="flex gap-3">
            <a href="#" onclick="exportData('csv'); return false;" class="text-xs font-medium text-[var(--accent)] hover:underline">Xuất CSV</a>
            <button onclick="loadLogs()" class="text-xs font-medium text-[var(--muted)] hover:text-[var(--ink)]"><i class="fa-solid fa-rotate-right"></i></button>
          </div>
        </div>
        <div class="overflow-x-auto max-h-[420px]">
          <table class="w-full text-left text-sm">
            <thead>
              <tr class="border-b border-[var(--border)] text-[var(--muted)] text-xs uppercase tracking-wider">
                <th class="pb-3 font-semibold">Thời gian</th>
                <th class="pb-3 font-semibold">Model</th>
                <th class="pb-3 font-semibold">Thông số</th>
                <th class="pb-3 font-semibold">Dự đoán</th>
                <th class="pb-3 font-semibold">Tin cậy</th>
              </tr>
            </thead>
            <tbody id="logsTableBody" class="divide-y divide-[var(--border)]">
              <tr><td colspan="5" class="py-8 text-center text-[var(--muted)]">Đang tải...</td></tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- THỐNG KÊ -->
    <div id="tab-stats" class="tab-content grid-cols-1 md:grid-cols-3 gap-5">
      <div class="card p-6">
        <p class="text-xs font-semibold text-[var(--muted)] uppercase tracking-wider">Tổng dự đoán</p>
        <p id="statTotal" class="font-display text-4xl font-bold mt-2">0</p>
      </div>
      <div class="card p-6">
        <p class="text-xs font-semibold text-[var(--muted)] uppercase tracking-wider">Độ tin cậy TB</p>
        <p id="statConf" class="font-display text-4xl font-bold mt-2">—%</p>
      </div>
      <div class="card p-6">
        <p class="text-xs font-semibold text-[var(--muted)] uppercase tracking-wider">Tỷ lệ anomaly</p>
        <p id="statAnomaly" class="font-display text-4xl font-bold mt-2">—%</p>
      </div>
      <div class="card p-6 md:col-span-3">
        <h3 class="text-xs font-semibold uppercase tracking-wider text-[var(--muted)] mb-4">Phân bố theo loài</h3>
        <div class="h-56"><canvas id="speciesChart"></canvas></div>
      </div>
    </div>
  </main>

  <footer class="text-center py-5 text-xs text-[var(--muted)] border-t border-[var(--border)]">
    Iris Botanica · Personal Studio · FastAPI + Scikit-Learn
  </footer>

  <div id="toast" class="toast"></div>

<script>
  let isRegistering = false;
  let selectedKernel = 'linear';
  let currentUser = localStorage.getItem('iris_user') || null;
  let speciesChart = null;

  // Theme
  function toggleTheme() {
    document.documentElement.classList.toggle('dark');
    const isDark = document.documentElement.classList.contains('dark');
    localStorage.setItem('iris_theme', isDark ? 'dark' : 'light');
    document.getElementById('themeIcon').className = isDark ? 'fa-solid fa-sun text-sm' : 'fa-solid fa-moon text-sm';
  }
  if (localStorage.getItem('iris_theme') === 'dark') {
    document.documentElement.classList.add('dark');
    document.getElementById('themeIcon').className = 'fa-solid fa-sun text-sm';
  }

  // Auth
  window.addEventListener('DOMContentLoaded', () => {
    if (currentUser) {
      document.getElementById('authModal').classList.add('hidden');
      document.getElementById('userGreeting').innerText = currentUser;
    }
  });

  function toggleAuthMode() {
    isRegistering = !isRegistering;
    document.getElementById('authTitle').innerText = isRegistering ? 'Tạo tài khoản mới' : 'Chào mừng trở lại';
    document.getElementById('authSubmitBtn').innerText = isRegistering ? 'Đăng ký' : 'Đăng nhập';
    document.getElementById('authSwitchBtn').innerText = isRegistering ? 'Đã có tài khoản? Đăng nhập' : 'Chưa có tài khoản? Đăng ký';
    document.getElementById('authError').classList.add('hidden');
  }

  async function handleAuth(e) {
    e.preventDefault();
    const username = document.getElementById('authUsername').value.trim();
    const password = document.getElementById('authPassword').value;
    const endpoint = isRegistering ? '/api/register' : '/api/login';
    const errDiv = document.getElementById('authError');
    errDiv.classList.add('hidden');
    try {
      const res = await fetch(endpoint, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ username, password })
      });
      const data = await res.json();
      if (res.ok) {
        if (isRegistering) {
          showToast('Đăng ký thành công! Hãy đăng nhập.');
          toggleAuthMode();
        } else {
          localStorage.setItem('iris_user', data.username);
          currentUser = data.username;
          document.getElementById('authModal').classList.add('hidden');
          document.getElementById('userGreeting').innerText = currentUser;
          showToast('Xin chào, ' + currentUser);
        }
      } else {
        errDiv.innerText = data.detail || 'Có lỗi xảy ra';
        errDiv.classList.remove('hidden');
      }
    } catch (err) {
      errDiv.innerText = 'Không kết nối được server';
      errDiv.classList.remove('hidden');
    }
  }

  function logout() {
    localStorage.removeItem('iris_user');
    location.reload();
  }

  function showToast(msg) {
    const t = document.getElementById('toast');
    t.innerText = msg;
    t.classList.add('show');
    setTimeout(() => t.classList.remove('show'), 2800);
  }

  // Charts
  const ctxProb = document.getElementById('probChart').getContext('2d');
  const probChart = new Chart(ctxProb, {
    type: 'bar',
    data: {
      labels: ['Setosa', 'Versicolor', 'Virginica'],
      datasets: [{
        data: [100, 0, 0],
        backgroundColor: ['#65a30d', '#a855f7', '#0ea5e9'],
        borderRadius: 6,
        borderSkipped: false
      }]
    },
    options: {
      plugins: { legend: { display: false } },
      scales: {
        y: { max: 100, ticks: { display: false }, grid: { display: false } },
        x: { grid: { display: false }, ticks: { font: { size: 11, weight: '600' } } }
      }
    }
  });

  function setKernel(k) {
    selectedKernel = k;
    ['linear','rbf','poly','sigmoid'].forEach(name => {
      const btn = document.getElementById('btn-kernel-' + name);
      btn.classList.toggle('active', name === k);
    });
  }

  function switchTab(id) {
    document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
    document.getElementById('tab-' + id).classList.add('active');
    document.querySelectorAll('.nav-btn').forEach(btn => {
      btn.classList.remove('active');
      btn.classList.add('text-[var(--muted)]');
    });
    const activeBtn = document.getElementById('nav-' + id);
    activeBtn.classList.add('active');
    activeBtn.classList.remove('text-[var(--muted)]');
    if (id === 'logs') loadLogs();
    if (id === 'stats') loadStats();
  }

  function loadSample(type) {
    const samples = {
      setosa: [5.1, 3.5, 1.4, 0.2],
      versicolor: [6.0, 2.9, 4.5, 1.5],
      virginica: [6.5, 3.0, 5.8, 2.2]
    };
    const v = samples[type];
    document.getElementById('sepal_length').value = v[0];
    document.getElementById('sepal_width').value = v[1];
    document.getElementById('petal_length').value = v[2];
    document.getElementById('petal_width').value = v[3];
    document.getElementById('predictionForm').dispatchEvent(new Event('submit'));
  }

  const sampleImages = {
    Setosa: 'https://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg',
    Versicolor: 'https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg',
    Virginica: 'https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg'
  };

  function updateUI(result) {
    document.getElementById('predClass').innerText = result.prediction;
    document.getElementById('predConf').innerText = (result.confidence * 100).toFixed(1) + '%';
    document.getElementById('reportModelBadge').innerText = result.model_used;
    if (sampleImages[result.prediction]) document.getElementById('resultImage').src = sampleImages[result.prediction];
    const badge = document.getElementById('anomalyBadge');
    if (result.is_anomaly) {
      badge.innerText = 'Cảnh báo anomaly';
      badge.className = 'text-[10px] font-semibold px-2.5 py-0.5 rounded-full bg-amber-100 text-amber-800 dark:bg-amber-900/40 dark:text-amber-300 mb-2';
    } else {
      badge.innerText = 'Bình thường';
      badge.className = 'text-[10px] font-semibold px-2.5 py-0.5 rounded-full bg-emerald-100 text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-300 mb-2';
    }
    if (result.probabilities) {
      probChart.data.datasets[0].data = result.probabilities;
      probChart.update();
    }
    if (result.ai_report) document.getElementById('aiReportContent').innerText = result.ai_report;
  }

  document.getElementById('predictionForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    const data = {
      sepal_length: parseFloat(document.getElementById('sepal_length').value),
      sepal_width: parseFloat(document.getElementById('sepal_width').value),
      petal_length: parseFloat(document.getElementById('petal_length').value),
      petal_width: parseFloat(document.getElementById('petal_width').value),
      kernel: selectedKernel,
      username: currentUser || 'guest'
    };
    const res = await fetch('/predict', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(data)
    });
    const result = await res.json();
    updateUI(result);
    switchTab('dashboard');
    showToast('Phân tích xong · ' + result.prediction);
  });

  async function runCompare() {
    const data = {
      sepal_length: parseFloat(document.getElementById('sepal_length').value) || 5.1,
      sepal_width: parseFloat(document.getElementById('sepal_width').value) || 3.5,
      petal_length: parseFloat(document.getElementById('petal_length').value) || 1.4,
      petal_width: parseFloat(document.getElementById('petal_width').value) || 0.2,
      username: currentUser || 'guest'
    };
    const res = await fetch('/compare', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(data)
    });
    const json = await res.json();
    let html = `<table class="w-full text-sm">
      <thead><tr class="border-b border-[var(--border)] text-left text-xs text-[var(--muted)] uppercase">
        <th class="pb-3">Kernel</th><th class="pb-3">Dự đoán</th><th class="pb-3">Tin cậy</th>
        <th class="pb-3">Setosa</th><th class="pb-3">Versicolor</th><th class="pb-3">Virginica</th>
      </tr></thead><tbody class="divide-y divide-[var(--border)]">`;
    json.results.forEach(r => {
      html += `<tr>
        <td class="py-3 font-semibold">${r.kernel}</td>
        <td class="py-3 font-medium">${r.prediction}</td>
        <td class="py-3 text-[var(--accent)] font-semibold">${r.confidence}%</td>
        <td class="py-3">${r.probabilities[0]}%</td>
        <td class="py-3">${r.probabilities[1]}%</td>
        <td class="py-3">${r.probabilities[2]}%</td>
      </tr>`;
    });
    html += '</tbody></table>';
    document.getElementById('compareResult').innerHTML = html;
  }

  async function diagnose(symptom) {
    document.getElementById('diagnosisResult').innerText = 'Đang phân tích...';
    const res = await fetch('/diagnose', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ symptom })
    });
    const data = await res.json();
    document.getElementById('diagnosisResult').innerHTML = `<strong class="text-[var(--accent)] block mb-2">${data.disease}</strong><p>${data.solution.replace(/\\n/g, '<br>')}</p>`;
  }

  function submitCustomDiagnosis() {
    const val = document.getElementById('customSymptomInput').value.trim();
    if (val) diagnose(val);
  }

  async function loadLogs() {
    const url = currentUser ? `/logs?username=${encodeURIComponent(currentUser)}` : '/logs';
    const res = await fetch(url);
    const allLogs = await res.json();
    const tbody = document.getElementById('logsTableBody');
    if (!allLogs.length) {
      tbody.innerHTML = '<tr><td colspan="5" class="py-8 text-center text-[var(--muted)]">Chưa có bản ghi</td></tr>';
      return;
    }
    tbody.innerHTML = allLogs.map(item => `
      <tr class="hover:bg-[var(--bg)] transition">
        <td class="py-3 text-[var(--muted)] text-xs">${item.timestamp}</td>
        <td class="py-3 font-medium text-[var(--accent)] text-xs">${item.model_used}</td>
        <td class="py-3 font-mono text-xs">${item.sepal_length} / ${item.sepal_width} / ${item.petal_length} / ${item.petal_width}</td>
        <td class="py-3 font-semibold">${item.prediction}</td>
        <td class="py-3 text-emerald-600 dark:text-emerald-400 font-medium">${(item.confidence * 100).toFixed(1)}%</td>
      </tr>
    `).join('');
  }

  async function loadStats() {
    const url = currentUser ? `/stats?username=${encodeURIComponent(currentUser)}` : '/stats';
    const res = await fetch(url);
    const s = await res.json();
    document.getElementById('statTotal').innerText = s.total;
    document.getElementById('statConf').innerText = s.avg_confidence + '%';
    document.getElementById('statAnomaly').innerText = s.anomaly_rate + '%';

    const labels = Object.keys(s.by_species || {});
    const values = Object.values(s.by_species || {});
    const ctx = document.getElementById('speciesChart').getContext('2d');
    if (speciesChart) speciesChart.destroy();
    speciesChart = new Chart(ctx, {
      type: 'doughnut',
      data: {
        labels,
        datasets: [{
          data: values,
          backgroundColor: ['#65a30d', '#a855f7', '#0ea5e9'],
          borderWidth: 0
        }]
      },
      options: {
        plugins: { legend: { position: 'bottom', labels: { font: { size: 12 } } } },
        cutout: '62%'
      }
    });
  }

  function exportData(fmt) {
    const q = currentUser ? `?username=${encodeURIComponent(currentUser)}` : '';
    window.open(`/export/${fmt}${q}`, '_blank');
  }

  async function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    const formData = new FormData();
    formData.append('file', file);
    document.getElementById('uploadStatusText').innerText = 'Đang xử lý...';
    const res = await fetch('/analyze-file', { method: 'POST', body: formData });
    const data = await res.json();
    if (data.status === 'SUCCESS') {
      updateUI(data.prediction_result);
      switchTab('dashboard');
      showToast('Đã phân tích từ file/ảnh');
    }
    document.getElementById('uploadStatusText').innerText = 'Chọn ảnh hoa hoặc file';
  }
</script>
</body>
</html>
"""
