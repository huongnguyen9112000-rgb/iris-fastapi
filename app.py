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
