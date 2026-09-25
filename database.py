import sqlite3

DB_FILE = "iris_system.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
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
    conn.commit()
    conn.close()
