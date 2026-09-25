from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from database import init_db
from routers import predict_router

app = FastAPI(title="Iris Botanical Enterprise Suite", version="18.0")

# Khởi tạo cơ sở dữ liệu khi khởi động
init_db()

# Gắn router dự đoán vào app chính
app.include_router(predict_router.router)

@app.get("/", response_class=HTMLResponse)
def get_dashboard():
    return "<h1>Iris Botanical Enterprise Suite is Running!</h1>"
