# main.py

import sqlite3
from datetime import datetime, timedelta

import bcrypt
import joblib
import numpy as np

from fastapi import (
    FastAPI,
    HTTPException,
    Depends
)

from fastapi.security import (
    OAuth2PasswordBearer
)

from jose import jwt, JWTError

from pydantic import BaseModel

from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.ensemble import IsolationForest

# ==================================================
# CONFIG
# ==================================================

app = FastAPI(
    title="Iris AI Analytics Platform",
    version="20.0"
)

DB_FILE = "iris_system.db"

SECRET_KEY = "CHANGE_THIS_SECRET_KEY"
ALGORITHM = "HS256"

# ==================================================
# DATABASE
# ==================================================

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        email TEXT UNIQUE,
        password_hash TEXT,
        created_at TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS predictions(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        timestamp TEXT,

        model_used TEXT,

        sepal_length REAL,
        sepal_width REAL,
        petal_length REAL,
        petal_width REAL,

        prediction TEXT,
        confidence REAL,

        is_anomaly INTEGER,

        FOREIGN KEY(user_id)
        REFERENCES users(id)
    )
    """)

    conn.commit()
    conn.close()

init_db()

# ==================================================
# MODEL
# ==================================================

AVAILABLE_KERNELS = [
    "linear",
    "rbf",
    "poly",
    "sigmoid"
]

TARGET_NAMES = [
    "Setosa",
    "Versicolor",
    "Virginica"
]

iris = load_iris()

ANOMALY_MODEL = IsolationForest(
    contamination=0.05,
    random_state=42
)

ANOMALY_MODEL.fit(iris.data)


def load_model(kernel):

    file_name = f"svm_{kernel}.pkl"

    try:
        return joblib.load(file_name)

    except:

        model = SVC(
            kernel=kernel,
            probability=True,
            random_state=42
        )

        model.fit(
            iris.data,
            iris.target
        )

        joblib.dump(
            model,
            file_name
        )

        return model

# ==================================================
# SCHEMAS
# ==================================================

class RegisterInput(BaseModel):
    username: str
    email: str
    password: str


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

# ==================================================
# JWT
# ==================================================

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="login"
)


def create_token(user_id):

    payload = {
        "sub": str(user_id),
        "exp": datetime.utcnow() + timedelta(days=7)
    }

    return jwt.encode(
        payload,
        SECRET_KEY,
        algorithm=ALGORITHM
    )


def get_current_user(
        token: str = Depends(oauth2_scheme)
):

    try:

        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM]
        )

        user_id = int(payload["sub"])

    except JWTError:
        raise HTTPException(
            401,
            "Token không hợp lệ"
        )

    conn = sqlite3.connect(DB_FILE)

    cur = conn.cursor()

    cur.execute(
        """
        SELECT id,username,email
        FROM users
        WHERE id=?
        """,
        (user_id,)
    )

    user = cur.fetchone()

    conn.close()

    if not user:
        raise HTTPException(
            401,
            "User không tồn tại"
        )

    return {
        "id": user[0],
        "username": user[1],
        "email": user[2]
    }

# ==================================================
# AUTH
# ==================================================

@app.post("/auth/register")
def register(data: RegisterInput):

    if len(data.password) < 8:

        raise HTTPException(
            400,
            "Mật khẩu tối thiểu 8 ký tự"
        )

    password_hash = bcrypt.hashpw(
        data.password.encode(),
        bcrypt.gensalt()
    ).decode()

    conn = sqlite3.connect(DB_FILE)

    cur = conn.cursor()

    try:

        cur.execute("""
        INSERT INTO users(
            username,
            email,
            password_hash,
            created_at
        )
        VALUES(?,?,?,?)
        """, (
            data.username,
            data.email,
            password_hash,
            datetime.now().isoformat()
        ))

        conn.commit()

    except sqlite3.IntegrityError:

        raise HTTPException(
            400,
            "Username hoặc Email đã tồn tại"
        )

    finally:
        conn.close()

    return {
        "message": "Đăng ký thành công"
    }


@app.post("/auth/login")
def login(data: LoginInput):

    conn = sqlite3.connect(DB_FILE)

    cur = conn.cursor()

    cur.execute("""
    SELECT id,password_hash
    FROM users
    WHERE username=?
    """,(data.username,))

    row = cur.fetchone()

    conn.close()

    if not row:

        raise HTTPException(
            401,
            "Sai thông tin đăng nhập"
        )

    user_id = row[0]
    password_hash = row[1]

    if not bcrypt.checkpw(
        data.password.encode(),
        password_hash.encode()
    ):

        raise HTTPException(
            401,
            "Sai thông tin đăng nhập"
        )

    token = create_token(user_id)

    return {
        "access_token": token,
        "token_type": "bearer"
    }

# ==================================================
# PROFILE
# ==================================================

@app.get("/me")
def me(user=Depends(get_current_user)):
    return user

# ==================================================
# PREDICTION
# ==================================================

@app.post("/predict")
def predict(
    data: IrisInput,
    user=Depends(get_current_user)
):

    kernel = data.kernel

    if kernel not in AVAILABLE_KERNELS:
        kernel = "rbf"

    model = load_model(kernel)

    features = np.array([[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]])

    probs = model.predict_proba(
        features
    )[0]

    index = np.argmax(probs)

    prediction = TARGET_NAMES[index]

    confidence = float(
        probs[index]
    )

    anomaly = ANOMALY_MODEL.predict(
        features
    )[0]

    is_anomaly = anomaly == -1

    conn = sqlite3.connect(DB_FILE)

    cur = conn.cursor()

    cur.execute("""
    INSERT INTO predictions(
        user_id,
        timestamp,
        model_used,

        sepal_length,
        sepal_width,
        petal_length,
        petal_width,

        prediction,
        confidence,

        is_anomaly
    )
    VALUES(
        ?,?,?,?,?,?,?,?,?,?
    )
    """,
    (
        user["id"],
        datetime.now().isoformat(),

        f"SVM-{kernel}",

        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width,

        prediction,
        confidence,

        int(is_anomaly)
    ))

    conn.commit()
    conn.close()

    return {

        "species": prediction,

        "confidence":
        round(confidence * 100, 2),

        "probabilities": {

            "Setosa":
            round(float(probs[0]) * 100, 2),

            "Versicolor":
            round(float(probs[1]) * 100, 2),

            "Virginica":
            round(float(probs[2]) * 100, 2)
        },

        "anomaly": is_anomaly,

        "model": kernel
    }

# ==================================================
# MY LOGS
# ==================================================

@app.get("/predictions")
def my_predictions(
    user=Depends(get_current_user)
):

    conn = sqlite3.connect(DB_FILE)

    cur = conn.cursor()

    cur.execute("""
    SELECT
        timestamp,
        prediction,
        confidence,
        model_used,
        is_anomaly
    FROM predictions
    WHERE user_id=?
    ORDER BY id DESC
    """,
    (user["id"],)
    )

    data = cur.fetchall()

    conn.close()

    return [
        {
            "timestamp": r[0],
            "prediction": r[1],
            "confidence": r[2],
            "model": r[3],
            "anomaly": bool(r[4])
        }
        for r in data
    ]


from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
async def dashboard():

    return """
<!DOCTYPE html>
<html lang="vi">

<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">

<title>Iris AI Analytics Platform</title>

<script src="https://cdn.tailwindcss.com"></script>

<style>
body{
    background:#0f172a;
}
.glass{
    background:rgba(255,255,255,.06);
    backdrop-filter:blur(16px);
    border:1px solid rgba(255,255,255,.08);
}
</style>

</head>

<body class="min-h-screen text-white">

<!-- NAVBAR -->

<nav class="border-b border-slate-800 bg-slate-950">

<div class="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">

<div>
<h1 class="font-black text-2xl">
🌸 Iris AI Analytics
</h1>
</div>

<div class="flex gap-3">

<button onclick="showLogin()"
class="px-5 py-2 rounded-xl bg-slate-800 hover:bg-slate-700">
Đăng nhập
</button>

<button onclick="showRegister()"
class="px-5 py-2 rounded-xl bg-blue-600 hover:bg-blue-700">
Đăng ký
</button>

</div>

</div>

</nav>

<!-- HERO -->

<section class="max-w-7xl mx-auto px-6 py-16">

<div class="grid lg:grid-cols-2 gap-12 items-center">

<div>

<h2 class="text-6xl font-black leading-tight">
Plant Intelligence
Powered By AI
</h2>

<p class="mt-6 text-slate-400 text-lg">
Nền tảng phân tích và nhận dạng Iris sử dụng
Machine Learning, SVM Multi Kernel,
Anomaly Detection và hệ thống báo cáo thông minh.
</p>

<div class="flex gap-4 mt-8">

<button onclick="showLogin()"
class="bg-blue-600 hover:bg-blue-700 px-6 py-3 rounded-xl font-bold">
Bắt đầu
</button>

<a href="/docs"
class="bg-slate-800 hover:bg-slate-700 px-6 py-3 rounded-xl font-bold">
Swagger API
</a>

</div>

</div>

<div>

<div class="glass rounded-3xl p-8">

<h3 class="font-bold text-xl mb-6">
Hệ thống AI
</h3>

<div class="grid grid-cols-2 gap-4">

<div class="bg-slate-900 rounded-2xl p-5">
<div class="text-3xl font-black text-blue-400">
4
</div>
<div class="text-slate-400 text-sm mt-1">
SVM Kernels
</div>
</div>

<div class="bg-slate-900 rounded-2xl p-5">
<div class="text-3xl font-black text-green-400">
99%
</div>
<div class="text-slate-400 text-sm mt-1">
Accuracy
</div>
</div>

<div class="bg-slate-900 rounded-2xl p-5">
<div class="text-3xl font-black text-purple-400">
JWT
</div>
<div class="text-slate-400 text-sm mt-1">
Authentication
</div>
</div>

<div class="bg-slate-900 rounded-2xl p-5">
<div class="text-3xl font-black text-pink-400">
AI
</div>
<div class="text-slate-400 text-sm mt-1">
Diagnosis
</div>
</div>

</div>

</div>

</div>

</div>

</section>

<!-- LOGIN MODAL -->

<div id="loginModal"
class="hidden fixed inset-0 bg-black/70 flex justify-center items-center p-4">

<div class="bg-slate-900 w-full max-w-md rounded-3xl p-8">

<h2 class="text-2xl font-black mb-6">
Đăng nhập
</h2>

<input
id="loginUsername"
placeholder="Tên đăng nhập"
class="w-full mb-4 bg-slate-800 rounded-xl p-3">

<input
id="loginPassword"
type="password"
placeholder="Mật khẩu"
class="w-full mb-4 bg-slate-800 rounded-xl p-3">

<button
onclick="login()"
class="w-full bg-blue-600 py-3 rounded-xl font-bold">
Đăng nhập
</button>

<button
onclick="closeModal()"
class="w-full mt-3 bg-slate-700 py-3 rounded-xl">
Đóng
</button>

</div>

</div>

<!-- REGISTER MODAL -->

<div id="registerModal"
class="hidden fixed inset-0 bg-black/70 flex justify-center items-center p-4">

<div class="bg-slate-900 w-full max-w-md rounded-3xl p-8">

<h2 class="text-2xl font-black mb-6">
Tạo tài khoản
</h2>

<input
id="registerUsername"
placeholder="Tên đăng nhập"
class="w-full mb-4 bg-slate-800 rounded-xl p-3">

<input
id="registerEmail"
placeholder="Email"
class="w-full mb-4 bg-slate-800 rounded-xl p-3">

<input
id="registerPassword"
type="password"
placeholder="Mật khẩu"
class="w-full mb-4 bg-slate-800 rounded-xl p-3">

<button
onclick="register()"
class="w-full bg-green-600 py-3 rounded-xl font-bold">
Đăng ký
</button>

<button
onclick="closeModal()"
class="w-full mt-3 bg-slate-700 py-3 rounded-xl">
Đóng
</button>

</div>

</div>

<script>

function showLogin(){
document.getElementById("loginModal").classList.remove("hidden");
}

function showRegister(){
document.getElementById("registerModal").classList.remove("hidden");
}

function closeModal(){
document.getElementById("loginModal").classList.add("hidden");
document.getElementById("registerModal").classList.add("hidden");
}

async function register(){

const data = {
username:
document.getElementById("registerUsername").value,

email:
document.getElementById("registerEmail").value,

password:
document.getElementById("registerPassword").value
};

let res = await fetch('/auth/register',{
method:'POST',
headers:{
'Content-Type':'application/json'
},
body:JSON.stringify(data)
});

let result = await res.json();

if(res.ok){
alert("Đăng ký thành công");
closeModal();
}else{
alert(result.detail);
}
}

async function login(){

const data = {
username:
document.getElementById("loginUsername").value,

password:
document.getElementById("loginPassword").value
};

let res = await fetch('/auth/login',{
method:'POST',
headers:{
'Content-Type':'application/json'
},
body:JSON.stringify(data)
});

let result = await res.json();

if(res.ok){

localStorage.setItem(
"token",
result.access_token
);

alert("Đăng nhập thành công");

window.location="/dashboard";

}else{
alert(result.detail);
}
}

</script>

</body>
</html>
"""
