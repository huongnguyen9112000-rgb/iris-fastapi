import os
import sqlite3
import csv
import io
import json
from datetime import datetime, timedelta
from typing import Optional

import bcrypt
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Response, UploadFile, File, Depends, Header
from fastapi.responses import HTMLResponse, JSONResponse
from jose import JWTError, jwt
from pydantic import BaseModel, EmailStr, Field
from sklearn.datasets import load_iris
from sklearn.ensemble import IsolationForest
from sklearn.svm import SVC

# ==================== CẤU HÌNH ====================
SECRET_KEY = os.getenv("SECRET_KEY", "iris-botanica-secret-key-change-me-in-production-2026")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 giờ

DB_FILE = "iris_system.db"
AVAILABLE_KERNELS = ["linear", "rbf", "poly", "sigmoid"]
TARGET_NAMES = ["Setosa", "Versicolor", "Virginica"]

app = FastAPI(title="Iris AI Analytics Platform", version="30.0")

# ==================== DATABASE ====================
def get_conn():
    return sqlite3.connect(DB_FILE)

def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            user_id INTEGER,
            username TEXT,
            model_used TEXT,
            sepal_length REAL,
            sepal_width REAL,
            petal_length REAL,
            petal_width REAL,
            prediction TEXT,
            confidence REAL,
            is_anomaly INTEGER,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ==================== ML MODELS ====================
def get_svm_model(kernel: str):
    path = f"svm_{kernel}_model.pkl"
    if not os.path.exists(path):
        iris = load_iris()
        model = SVC(kernel=kernel, probability=True, random_state=42)
        model.fit(iris.data, iris.target)
        joblib.dump(model, path)
    return joblib.load(path)

# Isolation Forest cho anomaly
_iris = load_iris()
anomaly_detector = IsolationForest(contamination=0.05, random_state=42)
anomaly_detector.fit(_iris.data)

# ==================== AUTH UTILS ====================
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Chưa đăng nhập hoặc token không hợp lệ")
    token = authorization.split(" ")[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Token không hợp lệ")
    except JWTError:
        raise HTTPException(status_code=401, detail="Token hết hạn hoặc không hợp lệ")

    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, username, email FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=401, detail="Người dùng không tồn tại")
    return {"id": row[0], "username": row[1], "email": row[2]}

# ==================== SCHEMAS ====================
class RegisterInput(BaseModel):
    username: str = Field(..., min_length=3, max_length=30)
    email: EmailStr
    password: str = Field(..., min_length=8)

class LoginInput(BaseModel):
    username: str
    password: str

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    kernel: str = "rbf"

class DiagnosisInput(BaseModel):
    symptom: str

# ==================== AUTH API ====================
@app.post("/api/register")
def register(data: RegisterInput):
    if len(data.password) < 8:
        raise HTTPException(status_code=400, detail="Mật khẩu phải có ít nhất 8 ký tự")
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO users (username, email, password, created_at) VALUES (?, ?, ?, ?)",
            (data.username.strip(), data.email.lower().strip(), hash_password(data.password),
             datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
        conn.commit()
        return {"status": "SUCCESS", "message": "Đăng ký thành công. Vui lòng đăng nhập."}
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Username hoặc Email đã tồn tại")
    finally:
        conn.close()

@app.post("/api/login")
def login(data: LoginInput):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, username, password FROM users WHERE username = ?", (data.username.strip(),))
    row = cur.fetchone()
    conn.close()
    if not row or not verify_password(data.password, row[2]):
        raise HTTPException(status_code=400, detail="Sai tên đăng nhập hoặc mật khẩu")
    token = create_access_token({"sub": row[1]})
    return {
        "status": "SUCCESS",
        "access_token": token,
        "token_type": "bearer",
        "username": row[1]
    }

@app.get("/api/me")
def me(user=Depends(get_current_user)):
    return user

# ==================== PREDICTION ====================
@app.post("/predict")
def predict(data: IrisInput, user=Depends(get_current_user)):
    kernel = data.kernel if data.kernel in AVAILABLE_KERNELS else "rbf"
    model = get_svm_model(kernel)
    features = np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])

    probs = model.predict_proba(features)[0]
    pred_idx = int(np.argmax(probs))
    prediction = TARGET_NAMES[pred_idx]
    confidence = float(probs[pred_idx])
    probabilities = [round(float(p) * 100, 1) for p in probs]

    # Anomaly bằng Isolation Forest
    anomaly_score = anomaly_detector.predict(features)
    is_anomaly = bool(anomaly_score[0] == -1)

    reports = {
        "Setosa": {
            "title": "Iris setosa",
            "desc": "Hoa nhỏ, cánh mỏng, màu tím-xanh nhẹ. Chịu lạnh tốt, phù hợp khí hậu mát.",
            "soil": "Đất thịt nhẹ, giàu mùn, hơi chua (pH 6.0–6.5)",
            "care": "Giữ ẩm bề mặt đều, tránh úng. Tỉa lá già định kỳ.",
            "light": "Nắng nhẹ buổi sáng, tránh nắng gắt trưa hè."
        },
        "Versicolor": {
            "title": "Iris versicolor",
            "desc": "Hoa lớn hơn Setosa, cánh có vân rõ, màu tím-xanh đậm. Ưa ẩm.",
            "soil": "Đất sét bùn, nhiều hữu cơ (pH 5.5–7.0)",
            "care": "Ưa nước, chịu ngập nhẹ. Bón phân hữu cơ 2 lần/năm.",
            "light": "Nắng gián tiếp hoặc nửa ngày."
        },
        "Virginica": {
            "title": "Iris virginica",
            "desc": "Hoa to, cánh dày, màu tím đậm hoặc trắng-tím. Rất khỏe và ưa nắng.",
            "soil": "Đất phù sa, thoát nước tốt (pH 6.5–7.5)",
            "care": "Tưới sâu, để khô nhẹ giữa các lần. Tăng lân + kali khi chuẩn bị ra hoa.",
            "light": "Nắng toàn phần 6–8 giờ/ngày."
        }
    }

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO predictions
        (timestamp, user_id, username, model_used, sepal_length, sepal_width,
         petal_length, petal_width, prediction, confidence, is_anomaly)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (timestamp, user["id"], user["username"], f"SVM ({kernel})",
          data.sepal_length, data.sepal_width, data.petal_length, data.petal_width,
          prediction, confidence, int(is_anomaly)))
    conn.commit()
    conn.close()

    return {
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": probabilities,
        "is_anomaly": is_anomaly,
        "report": reports[prediction],
        "model_used": f"SVM ({kernel})",
        "input_features": {
            "sepal_length": data.sepal_length,
            "sepal_width": data.sepal_width,
            "petal_length": data.petal_length,
            "petal_width": data.petal_width
        }
    }

@app.post("/compare")
def compare_kernels(data: IrisInput, user=Depends(get_current_user)):
    features = np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
    results = []
    for k in AVAILABLE_KERNELS:
        model = get_svm_model(k)
        probs = model.predict_proba(features)[0]
        pred_idx = int(np.argmax(probs))
        results.append({
            "kernel": k.upper(),
            "prediction": TARGET_NAMES[pred_idx],
            "confidence": round(float(probs[pred_idx]) * 100, 1),
            "probabilities": [round(float(p) * 100, 1) for p in probs]
        })
    return {"results": results}

# ==================== DIAGNOSIS (không dùng global) ====================
@app.post("/diagnose")
def diagnose(data: DiagnosisInput, user=Depends(get_current_user)):
    # Lấy loài gần nhất của user này
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT prediction FROM predictions
        WHERE user_id = ? ORDER BY id DESC LIMIT 1
    """, (user["id"],))
    row = cur.fetchone()
    conn.close()
    species = row[0] if row else "Setosa"

    query = data.symptom.lower().strip()
    kb = {
        "sâu": {"name": "Sâu hại ăn lá & sâu đục thân",
                "solution": "Bắt thủ công vào buổi sáng. Phun Bt hoặc dung dịch tỏi-ớt."},
        "vàng": {"name": "Vàng lá do úng / thối rễ",
                 "solution": f"Ngừng tưới. Kiểm tra thoát nước cho giống {species}. Bổ sung Trichoderma."},
        "đốm": {"name": "Đốm lá vi khuẩn / nấm",
                "solution": "Cắt bỏ lá bệnh, hạn chế tưới phun ban đêm. Phun thuốc gốc đồng."},
        "gỉ": {"name": "Bệnh gỉ sắt",
               "solution": "Tiêu hủy lá nặng. Phun Mancozeb hoặc Propiconazole."},
        "thối": {"name": "Thối mềm củ rễ (Soft Rot)",
                 "solution": "Đào củ, cắt phần nhũn, sát trùng bằng vôi bột."},
        "hoa": {"name": "Không ra hoa",
                "solution": f"Giống {species} cần đủ nắng 6–8h. Tăng lân + kali, giảm đạm."}
    }
    matched = [v for k, v in kb.items() if k in query]
    if not matched:
        return {
            "disease": f"Phân tích cho giống {species}",
            "solution": f"Triệu chứng “{data.symptom}”: Kiểm tra độ ẩm, ánh sáng và dinh dưỡng phù hợp với {species}."
        }
    return {
        "disease": f"Phát hiện {len(matched)} vấn đề cho giống {species}",
        "solution": "\n\n".join([f"• {d['name']}:\n{d['solution']}" for d in matched])
    }

# ==================== FILE (không còn fake hash) ====================
@app.post("/analyze-file")
async def analyze_file(file: UploadFile = File(...), user=Depends(get_current_user)):
    filename = (file.filename or "").lower()
    if filename.endswith((".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp")):
        return {
            "status": "ERROR",
            "message": "Phiên bản hiện tại chỉ hỗ trợ nhập thông số đo đạc của Iris. Phân tích ảnh chưa được huấn luyện (cần CNN + dataset)."
        }
    try:
        contents = await file.read()
        text = contents.decode("utf-8", errors="ignore")
        if filename.endswith(".json"):
            data = json.loads(text)
            metrics = {
                "sepal_length": float(data.get("sepal_length", data.get("sepalLength", 5.1))),
                "sepal_width": float(data.get("sepal_width", data.get("sepalWidth", 3.5))),
                "petal_length": float(data.get("petal_length", data.get("petalLength", 1.4))),
                "petal_width": float(data.get("petal_width", data.get("petalWidth", 0.2))),
                "kernel": data.get("kernel", "rbf")
            }
        else:
            lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
            vals = lines[-1].split(",")
            metrics = {
                "sepal_length": float(vals[0]),
                "sepal_width": float(vals[1]),
                "petal_length": float(vals[2]),
                "petal_width": float(vals[3]),
                "kernel": "rbf"
            }
        # Gọi lại predict logic
        result = predict(IrisInput(**metrics), user)
        return {"status": "SUCCESS", "extracted_metrics": metrics, "prediction_result": result}
    except Exception as e:
        return {"status": "ERROR", "message": f"Không đọc được file: {str(e)}"}

# ==================== LOGS & STATS ====================
@app.get("/logs")
def get_logs(user=Depends(get_current_user)):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, timestamp, model_used, sepal_length, sepal_width,
               petal_length, petal_width, prediction, confidence, is_anomaly
        FROM predictions WHERE user_id = ? ORDER BY id DESC LIMIT 200
    """, (user["id"],))
    rows = cur.fetchall()
    conn.close()
    return [{
        "id": r[0], "timestamp": r[1], "model_used": r[2],
        "sepal_length": r[3], "sepal_width": r[4],
        "petal_length": r[5], "petal_width": r[6],
        "prediction": r[7], "confidence": r[8], "is_anomaly": bool(r[9])
    } for r in rows]

@app.get("/stats")
def get_stats(user=Depends(get_current_user)):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT prediction, confidence, is_anomaly FROM predictions WHERE user_id = ?", (user["id"],))
    rows = cur.fetchall()
    conn.close()
    if not rows:
        return {"total": 0, "by_species": {}, "avg_confidence": 0, "anomaly_rate": 0}
    total = len(rows)
    by_species = {}
    conf_sum = 0.0
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

@app.get("/export/{fmt}")
def export_data(fmt: str, user=Depends(get_current_user)):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, timestamp, username, model_used, sepal_length, sepal_width,
               petal_length, petal_width, prediction, confidence, is_anomaly
        FROM predictions WHERE user_id = ?
    """, (user["id"],))
    rows = cur.fetchall()
    conn.close()
    if fmt == "json":
        data = [{"id": r[0], "timestamp": r[1], "user": r[2], "model": r[3],
                 "prediction": r[8], "confidence": r[9]} for r in rows]
        return JSONResponse(content=data)
    if fmt == "csv":
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["ID", "Timestamp", "User", "Model", "SepalL", "SepalW",
                         "PetalL", "PetalW", "Prediction", "Confidence", "Anomaly"])
        writer.writerows(rows)
        return Response(output.getvalue(), media_type="text/csv",
                        headers={"Content-Disposition": "attachment; filename=iris_export.csv"})
    raise HTTPException(status_code=400, detail="Chỉ hỗ trợ json hoặc csv")

# ==================== FRONTEND ====================
@app.get("/", response_class=HTMLResponse)
def dashboard():
    return r'''
<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Iris AI Analytics Platform</title>
<script src="https://cdn.tailwindcss.com"></script>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
  body { font-family: 'Inter', system-ui, sans-serif; }
  .card { background: white; border: 1px solid #e2e8f0; border-radius: 12px; box-shadow: 0 1px 2px rgba(0,0,0,.04); }
  .nav-item { padding: 0.5rem 1rem; border-radius: 8px; font-size: 0.875rem; font-weight: 500; color: #64748b; transition: all .15s; }
  .nav-item:hover { background: #f1f5f9; color: #0f172a; }
  .nav-item.active { background: #1e40af; color: white; }
  .tab { display: none; }
  .tab.active { display: block; }
  .kpi { background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 1.25rem; }
  .toast { position: fixed; bottom: 1.5rem; right: 1.5rem; background: #0f172a; color: white; padding: 0.75rem 1.25rem; border-radius: 8px; font-size: 0.875rem; font-weight: 500; z-index: 100; transform: translateY(120%); transition: transform .3s; }
  .toast.show { transform: translateY(0); }
</style>
</head>
<body class="bg-slate-50 text-slate-900 min-h-screen flex flex-col">

<!-- AUTH MODAL -->
<div id="authModal" class="fixed inset-0 z-50 bg-slate-900/60 flex items-center justify-center p-4">
  <div class="bg-white rounded-xl shadow-xl w-full max-w-md p-8">
    <div class="text-center mb-6">
      <div class="w-12 h-12 bg-blue-700 text-white rounded-xl flex items-center justify-center mx-auto mb-3">
        <i class="fa-solid fa-leaf"></i>
      </div>
      <h2 id="authTitle" class="text-xl font-bold">Đăng nhập</h2>
      <p class="text-sm text-slate-500 mt-1">Iris AI Analytics Platform</p>
    </div>
    <div id="authError" class="hidden bg-red-50 text-red-700 text-sm p-3 rounded-lg mb-4 text-center"></div>

    <form id="authForm" onsubmit="handleAuth(event)" class="space-y-4">
      <div id="emailField" class="hidden">
        <label class="block text-xs font-semibold text-slate-500 mb-1">Email</label>
        <input type="email" id="authEmail" class="w-full border border-slate-200 rounded-lg px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-600">
      </div>
      <div>
        <label class="block text-xs font-semibold text-slate-500 mb-1">Tên đăng nhập</label>
        <input type="text" id="authUsername" required class="w-full border border-slate-200 rounded-lg px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-600">
      </div>
      <div>
        <label class="block text-xs font-semibold text-slate-500 mb-1">Mật khẩu</label>
        <input type="password" id="authPassword" required class="w-full border border-slate-200 rounded-lg px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-600">
      </div>
      <div id="confirmField" class="hidden">
        <label class="block text-xs font-semibold text-slate-500 mb-1">Xác nhận mật khẩu</label>
        <input type="password" id="authConfirm" class="w-full border border-slate-200 rounded-lg px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-600">
      </div>
      <button type="submit" id="authBtn" class="w-full bg-blue-700 hover:bg-blue-800 text-white font-semibold py-2.5 rounded-lg text-sm">Đăng nhập</button>
    </form>
    <p class="text-center text-sm text-slate-500 mt-4">
      <button type="button" onclick="toggleAuth()" id="authSwitch" class="text-blue-700 font-medium hover:underline">Chưa có tài khoản? Đăng ký</button>
    </p>
  </div>
</div>

<!-- HEADER -->
<header class="bg-white border-b border-slate-200 sticky top-0 z-40">
  <div class="max-w-7xl mx-auto px-4 sm:px-6 h-14 flex items-center justify-between">
    <div class="flex items-center gap-3">
      <div class="w-8 h-8 bg-blue-700 text-white rounded-lg flex items-center justify-center text-sm">
        <i class="fa-solid fa-leaf"></i>
      </div>
      <div>
        <div class="font-semibold text-sm leading-tight">Iris AI Analytics</div>
        <div class="text-[10px] text-slate-400">Platform</div>
      </div>
    </div>
    <nav class="hidden md:flex items-center gap-1">
      <button onclick="switchTab('dashboard')" id="nav-dashboard" class="nav-item active">Dashboard</button>
      <button onclick="switchTab('predict')" id="nav-predict" class="nav-item">Phân tích</button>
      <button onclick="switchTab('compare')" id="nav-compare" class="nav-item">So sánh</button>
      <button onclick="switchTab('diagnosis')" id="nav-diagnosis" class="nav-item">Chẩn đoán</button>
      <button onclick="switchTab('logs')" id="nav-logs" class="nav-item">Lịch sử</button>
      <button onclick="switchTab('stats')" id="nav-stats" class="nav-item">Thống kê</button>
    </nav>
    <div class="flex items-center gap-3">
      <div class="flex items-center gap-2">
        <div id="userAvatar" class="w-8 h-8 rounded-full bg-blue-100 text-blue-700 flex items-center justify-center text-xs font-bold">?</div>
        <div class="hidden sm:block">
          <div id="userName" class="text-sm font-medium leading-tight">—</div>
          <div class="text-[10px] text-slate-400">Online</div>
        </div>
      </div>
      <button onclick="logout()" class="text-xs text-slate-500 hover:text-red-600 font-medium">Đăng xuất</button>
    </div>
  </div>
</header>

<main class="max-w-7xl mx-auto px-4 sm:px-6 py-6 flex-1 w-full">

  <!-- DASHBOARD -->
  <div id="tab-dashboard" class="tab active space-y-6">
    <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
      <div class="kpi">
        <div class="text-xs font-medium text-slate-500 uppercase tracking-wide">Tổng dự đoán</div>
        <div id="kpiTotal" class="text-2xl font-bold mt-1">0</div>
      </div>
      <div class="kpi">
        <div class="text-xs font-medium text-slate-500 uppercase tracking-wide">Độ tin cậy TB</div>
        <div id="kpiConf" class="text-2xl font-bold mt-1">—%</div>
      </div>
      <div class="kpi">
        <div class="text-xs font-medium text-slate-500 uppercase tracking-wide">Anomaly</div>
        <div id="kpiAnomaly" class="text-2xl font-bold mt-1">—%</div>
      </div>
      <div class="kpi">
        <div class="text-xs font-medium text-slate-500 uppercase tracking-wide">Model mặc định</div>
        <div class="text-2xl font-bold mt-1">SVM-RBF</div>
      </div>
    </div>

    <div class="grid lg:grid-cols-12 gap-6">
      <div class="lg:col-span-4 space-y-4">
        <div class="card p-5 text-center">
          <img id="resultImage" src="https://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg"
               class="w-28 h-28 object-cover rounded-xl mx-auto border border-slate-200 mb-3">
          <div id="anomalyBadge" class="inline-block text-[10px] font-semibold px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-700 mb-2">Bình thường</div>
          <h2 id="predClass" class="text-2xl font-bold">—</h2>
          <p class="text-sm text-slate-500 mt-1">Độ tin cậy <span id="predConf" class="font-semibold text-blue-700">—</span></p>
          <p class="text-xs text-slate-400 mt-1" id="reportModel">SVM (rbf)</p>
          <div id="inputFeatures" class="flex flex-wrap justify-center gap-1.5 mt-3 text-[11px] text-slate-500"></div>
        </div>
        <div class="card p-4">
          <div class="text-xs font-semibold text-slate-500 uppercase mb-2">Phân bố xác suất</div>
          <div class="h-36"><canvas id="probChart"></canvas></div>
        </div>
      </div>
      <div class="lg:col-span-8">
        <div class="card p-5 h-full">
          <div class="flex items-center justify-between mb-4">
            <h3 class="font-semibold">Báo cáo chăm sóc</h3>
            <button onclick="window.print()" class="text-xs text-blue-700 font-medium hover:underline">In báo cáo</button>
          </div>
          <div id="reportBox" class="text-sm text-slate-600 leading-relaxed">
            Chưa có kết quả. Hãy chạy phân tích ở tab “Phân tích”.
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- PREDICT -->
  <div id="tab-predict" class="tab">
    <div class="grid md:grid-cols-2 gap-6">
      <div class="card p-5 space-y-4">
        <h3 class="font-semibold">Thông số & Kernel</h3>
        <div>
          <label class="text-xs font-semibold text-slate-500 uppercase">Kernel</label>
          <div class="grid grid-cols-4 gap-2 mt-1">
            <button onclick="setKernel('linear')" id="k-linear" class="kernel-btn py-2 text-xs border rounded-lg">Linear</button>
            <button onclick="setKernel('rbf')" id="k-rbf" class="kernel-btn py-2 text-xs border rounded-lg bg-blue-700 text-white border-blue-700">RBF</button>
            <button onclick="setKernel('poly')" id="k-poly" class="kernel-btn py-2 text-xs border rounded-lg">Poly</button>
            <button onclick="setKernel('sigmoid')" id="k-sigmoid" class="kernel-btn py-2 text-xs border rounded-lg">Sigmoid</button>
          </div>
        </div>
        <div class="grid grid-cols-3 gap-2">
          <button onclick="loadSample('setosa')" class="text-xs py-2 border rounded-lg hover:bg-slate-50">Setosa</button>
          <button onclick="loadSample('versicolor')" class="text-xs py-2 border rounded-lg hover:bg-slate-50">Versicolor</button>
          <button onclick="loadSample('virginica')" class="text-xs py-2 border rounded-lg hover:bg-slate-50">Virginica</button>
        </div>
        <form id="predForm" class="space-y-3">
          <div class="grid grid-cols-2 gap-3">
            <div>
              <label class="text-[11px] font-medium text-slate-500">Sepal Length</label>
              <input type="number" step="0.1" id="sl" value="5.1" class="w-full border rounded-lg px-3 py-2 text-sm mt-0.5">
            </div>
            <div>
              <label class="text-[11px] font-medium text-slate-500">Sepal Width</label>
              <input type="number" step="0.1" id="sw" value="3.5" class="w-full border rounded-lg px-3 py-2 text-sm mt-0.5">
            </div>
            <div>
              <label class="text-[11px] font-medium text-slate-500">Petal Length</label>
              <input type="number" step="0.1" id="pl" value="1.4" class="w-full border rounded-lg px-3 py-2 text-sm mt-0.5">
            </div>
            <div>
              <label class="text-[11px] font-medium text-slate-500">Petal Width</label>
              <input type="number" step="0.1" id="pw" value="0.2" class="w-full border rounded-lg px-3 py-2 text-sm mt-0.5">
            </div>
          </div>
          <button type="submit" class="w-full bg-blue-700 hover:bg-blue-800 text-white font-semibold py-2.5 rounded-lg text-sm">Chạy phân tích</button>
        </form>
      </div>
      <div class="card p-5">
        <h3 class="font-semibold mb-2">Tải file CSV / JSON</h3>
        <p class="text-sm text-slate-500 mb-4">Không hỗ trợ nhận diện ảnh. Chỉ nhận file chứa 4 thông số đo.</p>
        <div class="border-2 border-dashed border-slate-200 rounded-xl p-8 text-center cursor-pointer hover:border-blue-400" onclick="document.getElementById('fileIn').click()">
          <i class="fa-solid fa-file-arrow-up text-2xl text-slate-400 mb-2"></i>
          <p class="text-sm font-medium" id="fileStatus">Chọn file CSV hoặc JSON</p>
          <input type="file" id="fileIn" accept=".csv,.json" class="hidden" onchange="handleFile(event)">
        </div>
      </div>
    </div>
  </div>

  <!-- COMPARE -->
  <div id="tab-compare" class="tab">
    <div class="card p-5">
      <div class="flex items-center justify-between mb-4">
        <div>
          <h3 class="font-semibold">So sánh 4 Kernel</h3>
          <p class="text-sm text-slate-500">Cùng một bộ thông số</p>
        </div>
        <button onclick="runCompare()" class="bg-blue-700 text-white text-sm font-medium px-4 py-2 rounded-lg">Chạy so sánh</button>
      </div>
      <div id="compareBox" class="overflow-x-auto text-sm text-slate-500">Nhấn nút để xem kết quả.</div>
    </div>
  </div>

  <!-- DIAGNOSIS -->
  <div id="tab-diagnosis" class="tab">
    <div class="grid md:grid-cols-2 gap-6">
      <div class="card p-5 space-y-3">
        <h3 class="font-semibold">Chẩn đoán</h3>
        <div class="flex gap-2">
          <input id="symptomIn" placeholder="Nhập triệu chứng..." class="flex-1 border rounded-lg px-3 py-2 text-sm" onkeypress="if(event.key==='Enter') doDiagnose()">
          <button onclick="doDiagnose()" class="bg-blue-700 text-white px-4 py-2 rounded-lg text-sm font-medium">Hỏi</button>
        </div>
        <div class="space-y-1.5">
          <button onclick="doDiagnose('vàng')" class="w-full text-left text-sm p-2.5 border rounded-lg hover:bg-slate-50">Lá úa vàng / thối gốc</button>
          <button onclick="doDiagnose('đốm')" class="w-full text-left text-sm p-2.5 border rounded-lg hover:bg-slate-50">Đốm nâu / đen trên lá</button>
          <button onclick="doDiagnose('sâu')" class="w-full text-left text-sm p-2.5 border rounded-lg hover:bg-slate-50">Sâu ăn lá / đục thân</button>
          <button onclick="doDiagnose('gỉ')" class="w-full text-left text-sm p-2.5 border rounded-lg hover:bg-slate-50">Gỉ sắt</button>
          <button onclick="doDiagnose('thối')" class="w-full text-left text-sm p-2.5 border rounded-lg hover:bg-slate-50">Thối mềm củ rễ</button>
          <button onclick="doDiagnose('hoa')" class="w-full text-left text-sm p-2.5 border rounded-lg hover:bg-slate-50">Không ra hoa</button>
        </div>
      </div>
      <div class="card p-5">
        <h3 class="font-semibold mb-3">Kết quả</h3>
        <div id="diagResult" class="text-sm text-slate-600 leading-relaxed min-h-[200px]">Chọn hoặc nhập triệu chứng.</div>
      </div>
    </div>
  </div>

  <!-- LOGS -->
  <div id="tab-logs" class="tab">
    <div class="card p-5">
      <div class="flex justify-between items-center mb-4">
        <h3 class="font-semibold">Lịch sử dự đoán</h3>
        <div class="flex gap-3">
          <button onclick="exportData('csv')" class="text-xs text-blue-700 font-medium">Xuất CSV</button>
          <button onclick="loadLogs()" class="text-xs text-slate-500"><i class="fa-solid fa-rotate"></i></button>
        </div>
      </div>
      <div class="overflow-x-auto max-h-96">
        <table class="w-full text-sm">
          <thead>
            <tr class="text-left text-xs text-slate-500 border-b">
              <th class="pb-2 font-medium">Thời gian</th>
              <th class="pb-2 font-medium">Model</th>
              <th class="pb-2 font-medium">Thông số</th>
              <th class="pb-2 font-medium">Dự đoán</th>
              <th class="pb-2 font-medium">Tin cậy</th>
            </tr>
          </thead>
          <tbody id="logsBody" class="divide-y"></tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- STATS -->
  <div id="tab-stats" class="tab">
    <div class="grid md:grid-cols-3 gap-4 mb-6">
      <div class="kpi"><div class="text-xs text-slate-500 uppercase">Tổng</div><div id="sTotal" class="text-2xl font-bold mt-1">0</div></div>
      <div class="kpi"><div class="text-xs text-slate-500 uppercase">Tin cậy TB</div><div id="sConf" class="text-2xl font-bold mt-1">—%</div></div>
      <div class="kpi"><div class="text-xs text-slate-500 uppercase">Anomaly</div><div id="sAnomaly" class="text-2xl font-bold mt-1">—%</div></div>
    </div>
    <div class="card p-5">
      <h3 class="font-semibold mb-4">Phân bố loài</h3>
      <div class="h-56"><canvas id="speciesChart"></canvas></div>
    </div>
  </div>
</main>

<footer class="border-t border-slate-200 py-4 text-center text-xs text-slate-400">
  Iris AI Analytics Platform · FastAPI + Scikit-Learn · JWT + bcrypt + IsolationForest
</footer>

<div id="toast" class="toast"></div>

<script>
let token = localStorage.getItem('iris_token');
let currentUser = null;
let selectedKernel = 'rbf';
let speciesChart = null;
let isRegister = false;

const images = {
  Setosa: 'https://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg',
  Versicolor: 'https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg',
  Virginica: 'https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg'
};

const probChart = new Chart(document.getElementById('probChart'), {
  type: 'bar',
  data: { labels: ['Setosa','Versicolor','Virginica'], datasets: [{ data: [0,0,0], backgroundColor: ['#2563eb','#7c3aed','#0ea5e9'], borderRadius: 4 }] },
  options: { plugins:{legend:{display:false}}, scales:{ y:{max:100, display:false}, x:{grid:{display:false}} } }
});

function authHeaders() {
  return { 'Content-Type': 'application/json', 'Authorization': 'Bearer ' + token };
}

function showToast(msg) {
  const t = document.getElementById('toast');
  t.innerText = msg;
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 2800);
}

function toggleAuth() {
  isRegister = !isRegister;
  document.getElementById('authTitle').innerText = isRegister ? 'Đăng ký tài khoản' : 'Đăng nhập';
  document.getElementById('authBtn').innerText = isRegister ? 'Đăng ký' : 'Đăng nhập';
  document.getElementById('authSwitch').innerText = isRegister ? 'Đã có tài khoản? Đăng nhập' : 'Chưa có tài khoản? Đăng ký';
  document.getElementById('emailField').classList.toggle('hidden', !isRegister);
  document.getElementById('confirmField').classList.toggle('hidden', !isRegister);
  document.getElementById('authError').classList.add('hidden');
}

async function handleAuth(e) {
  e.preventDefault();
  const username = document.getElementById('authUsername').value.trim();
  const password = document.getElementById('authPassword').value;
  const err = document.getElementById('authError');
  err.classList.add('hidden');

  if (isRegister) {
    const email = document.getElementById('authEmail').value.trim();
    const confirm = document.getElementById('authConfirm').value;
    if (password.length < 8) { err.innerText = 'Mật khẩu tối thiểu 8 ký tự'; err.classList.remove('hidden'); return; }
    if (password !== confirm) { err.innerText = 'Mật khẩu xác nhận không khớp'; err.classList.remove('hidden'); return; }
    const res = await fetch('/api/register', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ username, email, password })
    });
    const data = await res.json();
    if (!res.ok) { err.innerText = data.detail || 'Lỗi đăng ký'; err.classList.remove('hidden'); return; }
    showToast('Đăng ký thành công. Hãy đăng nhập.');
    toggleAuth();
    return;
  }

  const res = await fetch('/api/login', {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({ username, password })
  });
  const data = await res.json();
  if (!res.ok) { err.innerText = data.detail || 'Sai thông tin'; err.classList.remove('hidden'); return; }

  token = data.access_token;
  localStorage.setItem('iris_token', token);
  currentUser = data.username;
  document.getElementById('authModal').classList.add('hidden');
  document.getElementById('userName').innerText = currentUser;
  document.getElementById('userAvatar').innerText = currentUser[0].toUpperCase();
  showToast('Xin chào, ' + currentUser);
  loadStats();
}

function logout() {
  localStorage.removeItem('iris_token');
  token = null;
  location.reload();
}

async function checkAuth() {
  if (!token) return;
  try {
    const res = await fetch('/api/me', { headers: authHeaders() });
    if (!res.ok) throw new Error();
    const u = await res.json();
    currentUser = u.username;
    document.getElementById('authModal').classList.add('hidden');
    document.getElementById('userName').innerText = currentUser;
    document.getElementById('userAvatar').innerText = currentUser[0].toUpperCase();
    loadStats();
  } catch {
    localStorage.removeItem('iris_token');
    token = null;
  }
}

function switchTab(id) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.getElementById('tab-' + id).classList.add('active');
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  document.getElementById('nav-' + id).classList.add('active');
  if (id === 'logs') loadLogs();
  if (id === 'stats' || id === 'dashboard') loadStats();
}

function setKernel(k) {
  selectedKernel = k;
  ['linear','rbf','poly','sigmoid'].forEach(name => {
    const btn = document.getElementById('k-' + name);
    if (name === k) {
      btn.className = 'kernel-btn py-2 text-xs border rounded-lg bg-blue-700 text-white border-blue-700';
    } else {
      btn.className = 'kernel-btn py-2 text-xs border rounded-lg';
    }
  });
}

function loadSample(type) {
  const s = { setosa:[5.1,3.5,1.4,0.2], versicolor:[6.0,2.9,4.5,1.5], virginica:[6.5,3.0,5.8,2.2] };
  const v = s[type];
  document.getElementById('sl').value = v[0];
  document.getElementById('sw').value = v[1];
  document.getElementById('pl').value = v[2];
  document.getElementById('pw').value = v[3];
  document.getElementById('predForm').dispatchEvent(new Event('submit'));
}

function updateUI(r) {
  document.getElementById('predClass').innerText = r.prediction;
  document.getElementById('predConf').innerText = (r.confidence*100).toFixed(1) + '%';
  document.getElementById('reportModel').innerText = r.model_used;
  if (images[r.prediction]) document.getElementById('resultImage').src = images[r.prediction];

  const badge = document.getElementById('anomalyBadge');
  if (r.is_anomaly) {
    badge.innerText = 'Anomaly';
    badge.className = 'inline-block text-[10px] font-semibold px-2 py-0.5 rounded-full bg-amber-50 text-amber-700 mb-2';
  } else {
    badge.innerText = 'Bình thường';
    badge.className = 'inline-block text-[10px] font-semibold px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-700 mb-2';
  }

  if (r.probabilities) {
    probChart.data.datasets[0].data = r.probabilities;
    probChart.update();
  }

  if (r.input_features) {
    const f = r.input_features;
    document.getElementById('inputFeatures').innerHTML =
      `<span class="bg-slate-100 px-2 py-0.5 rounded">SL ${f.sepal_length}</span>
       <span class="bg-slate-100 px-2 py-0.5 rounded">SW ${f.sepal_width}</span>
       <span class="bg-slate-100 px-2 py-0.5 rounded">PL ${f.petal_length}</span>
       <span class="bg-slate-100 px-2 py-0.5 rounded">PW ${f.petal_width}</span>`;
  }

  if (r.report) {
    const rp = r.report;
    document.getElementById('reportBox').innerHTML = `
      <div class="space-y-3">
        <div><div class="text-xs font-semibold text-slate-400 uppercase">Đặc điểm</div><p>${rp.desc}</p></div>
        <div><div class="text-xs font-semibold text-slate-400 uppercase">Đất & pH</div><p>${rp.soil}</p></div>
        <div><div class="text-xs font-semibold text-slate-400 uppercase">Chăm sóc</div><p>${rp.care}</p></div>
        <div><div class="text-xs font-semibold text-slate-400 uppercase">Ánh sáng</div><p>${rp.light}</p></div>
      </div>`;
  }
}

document.getElementById('predForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  if (!token) return showToast('Vui lòng đăng nhập');
  const body = {
    sepal_length: parseFloat(document.getElementById('sl').value),
    sepal_width: parseFloat(document.getElementById('sw').value),
    petal_length: parseFloat(document.getElementById('pl').value),
    petal_width: parseFloat(document.getElementById('pw').value),
    kernel: selectedKernel
  };
  const res = await fetch('/predict', { method:'POST', headers: authHeaders(), body: JSON.stringify(body) });
  if (!res.ok) { showToast('Lỗi phân tích'); return; }
  const data = await res.json();
  updateUI(data);
  switchTab('dashboard');
  showToast('Phân tích xong · ' + data.prediction);
  loadStats();
});

async function runCompare() {
  if (!token) return showToast('Vui lòng đăng nhập');
  const body = {
    sepal_length: parseFloat(document.getElementById('sl').value)||5.1,
    sepal_width: parseFloat(document.getElementById('sw').value)||3.5,
    petal_length: parseFloat(document.getElementById('pl').value)||1.4,
    petal_width: parseFloat(document.getElementById('pw').value)||0.2
  };
  const res = await fetch('/compare', { method:'POST', headers: authHeaders(), body: JSON.stringify(body) });
  const json = await res.json();
  let html = `<table class="w-full text-sm"><thead><tr class="text-left text-xs text-slate-500 border-b">
    <th class="pb-2">Kernel</th><th class="pb-2">Dự đoán</th><th class="pb-2">Tin cậy</th>
    <th class="pb-2">Setosa</th><th class="pb-2">Versicolor</th><th class="pb-2">Virginica</th></tr></thead><tbody>`;
  json.results.forEach(r => {
    html += `<tr class="border-b border-slate-100">
      <td class="py-2 font-medium">${r.kernel}</td>
      <td class="py-2">${r.prediction}</td>
      <td class="py-2 text-blue-700 font-semibold">${r.confidence}%</td>
      <td class="py-2">${r.probabilities[0]}%</td>
      <td class="py-2">${r.probabilities[1]}%</td>
      <td class="py-2">${r.probabilities[2]}%</td></tr>`;
  });
  html += '</tbody></table>';
  document.getElementById('compareBox').innerHTML = html;
}

async function doDiagnose(symptom) {
  if (!token) return showToast('Vui lòng đăng nhập');
  const s = symptom || document.getElementById('symptomIn').value.trim();
  if (!s) return;
  document.getElementById('diagResult').innerText = 'Đang phân tích...';
  const res = await fetch('/diagnose', { method:'POST', headers: authHeaders(), body: JSON.stringify({ symptom: s }) });
  const data = await res.json();
  document.getElementById('diagResult').innerHTML = `<div class="font-semibold text-blue-700 mb-2">${data.disease}</div><div class="whitespace-pre-line">${data.solution}</div>`;
}

async function loadLogs() {
  if (!token) return;
  const res = await fetch('/logs', { headers: authHeaders() });
  const rows = await res.json();
  const body = document.getElementById('logsBody');
  if (!rows.length) { body.innerHTML = '<tr><td colspan="5" class="py-6 text-center text-slate-400">Chưa có dữ liệu</td></tr>'; return; }
  body.innerHTML = rows.map(r => `
    <tr>
      <td class="py-2 text-xs text-slate-500">${r.timestamp}</td>
      <td class="py-2 text-xs text-blue-700 font-medium">${r.model_used}</td>
      <td class="py-2 font-mono text-xs">${r.sepal_length}/${r.sepal_width}/${r.petal_length}/${r.petal_width}</td>
      <td class="py-2 font-medium">${r.prediction}</td>
      <td class="py-2 text-emerald-600">${(r.confidence*100).toFixed(1)}%</td>
    </tr>`).join('');
}

async function loadStats() {
  if (!token) return;
  const res = await fetch('/stats', { headers: authHeaders() });
  const s = await res.json();
  document.getElementById('kpiTotal').innerText = s.total;
  document.getElementById('kpiConf').innerText = s.avg_confidence + '%';
  document.getElementById('kpiAnomaly').innerText = s.anomaly_rate + '%';
  document.getElementById('sTotal').innerText = s.total;
  document.getElementById('sConf').innerText = s.avg_confidence + '%';
  document.getElementById('sAnomaly').innerText = s.anomaly_rate + '%';

  const labels = Object.keys(s.by_species || {});
  const values = Object.values(s.by_species || {});
  const ctx = document.getElementById('speciesChart');
  if (speciesChart) speciesChart.destroy();
  speciesChart = new Chart(ctx, {
    type: 'doughnut',
    data: { labels, datasets: [{ data: values, backgroundColor: ['#2563eb','#7c3aed','#0ea5e9'], borderWidth: 0 }] },
    options: { plugins: { legend: { position: 'bottom' } }, cutout: '60%' }
  });
}

function exportData(fmt) {
  if (!token) return;
  // Vì export cần token, ta dùng fetch + blob
  fetch('/export/' + fmt, { headers: authHeaders() })
    .then(r => r.blob())
    .then(blob => {
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'iris_export.' + fmt;
      a.click();
    });
}

async function handleFile(e) {
  const file = e.target.files[0];
  if (!file || !token) return;
  document.getElementById('fileStatus').innerText = 'Đang xử lý...';
  const fd = new FormData();
  fd.append('file', file);
  const res = await fetch('/analyze-file', { method: 'POST', headers: { 'Authorization': 'Bearer ' + token }, body: fd });
  const data = await res.json();
  if (data.status === 'SUCCESS') {
    updateUI(data.prediction_result);
    switchTab('dashboard');
    showToast('Đã phân tích từ file');
  } else {
    showToast(data.message || 'Lỗi');
  }
  document.getElementById('fileStatus').innerText = 'Chọn file CSV hoặc JSON';
}

// Init
checkAuth();
</script>
</body>
</html>
'''
