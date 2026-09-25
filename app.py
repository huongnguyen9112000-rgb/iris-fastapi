from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from database import init_db
from routers import predict_router

app = FastAPI(title="Iris Botanical Enterprise Suite", version="18.0")

# Khởi tạo cơ sở dữ liệu khi khởi động
init_db()

# Gắn router dự đoán vào app chính
app.include_router(predict_router.router)

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
    return """
    <!DOCTYPE html>
    <html lang="vi">
    <head>
        <meta charset="UTF-8">
        <title>Iris Botanical Enterprise Suite</title>
        <style>
            body { font-family: Arial, sans-serif; background-color: #f4f6f9; margin: 0; padding: 20px; }
            .container { max-width: 600px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 10px rgba(0,0,0,0.1); }
            h2 { color: #2c3e50; text-align: center; }
            .form-group { margin-bottom: 15px; }
            label { display: block; margin-bottom: 5px; font-weight: bold; color: #34495e; }
            input, select { width: 100%; padding: 10px; border: 1px solid #ccc; border-radius: 5px; box-sizing: border-box; }
            button { background-color: #27ae60; color: white; padding: 12px; border: none; width: 100%; border-radius: 5px; font-size: 16px; cursor: pointer; }
            button:hover { background-color: #219653; }
            #result { margin-top: 20px; padding: 15px; background: #e8f8f5; border-left: 5px solid #27ae60; display: none; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>🌸 Iris Botanical Enterprise Suite</h2>
            <div class="form-group">
                <label>Sepal Length (Chiều dài đài hoa):</label>
                <input type="number" id="sepal_length" step="0.1" value="5.1">
            </div>
            <div class="form-group">
                <label>Sepal Width (Chiều rộng đài hoa):</label>
                <input type="number" id="sepal_width" step="0.1" value="3.5">
            </div>
            <div class="form-group">
                <label>Petal Length (Chiều dài cánh hoa):</label>
                <input type="number" id="petal_length" step="0.1" value="1.4">
            </div>
            <div class="form-group">
                <label>Petal Width (Chiều rộng cánh hoa):</label>
                <input type="number" id="petal_width" step="0.1" value="0.2">
            </div>
            <div class="form-group">
                <label>Chọn Kernel Model:</label>
                <select id="kernel">
                    <option value="linear">Linear</option>
                    <option value="rbf">RBF</option>
                    <option value="poly">Poly</option>
                    <option value="sigmoid">Sigmoid</option>
                </select>
            </div>
            <button onclick="predict()">Dự đoán loài hoa</button>
            <div id="result"></div>
        </div>

        <script>
            async function predict() {
                const data = {
                    sepal_length: parseFloat(document.getElementById('sepal_length').value),
                    sepal_width: parseFloat(document.getElementById('sepal_width').value),
                    petal_length: parseFloat(document.getElementById('petal_length').value),
                    petal_width: parseFloat(document.getElementById('petal_width').value),
                    kernel: document.getElementById('kernel').value
                };

                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });

                const result = await response.json();
                const resDiv = document.getElementById('result');
                resDiv.style.display = 'block';
                resDiv.innerHTML = `<strong>Kết quả dự đoán:</strong> ${result.prediction} <br>` +
                                   `<strong>Độ tin cậy:</strong> ${(result.confidence * 100).toFixed(1)}% <br>` +
                                   `<strong>Model sử dụng:</strong> ${result.model_used}`;
            }
        </script>
    </body>
    </html>
    """
