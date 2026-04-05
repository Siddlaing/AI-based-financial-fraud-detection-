import os
import json
import sqlite3
from datetime import datetime

import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# --- Load Model & Config ---
print("Loading AI model...")
try:
    model = joblib.load('fraud_model.pkl')
    print("Model loaded successfully!")
except FileNotFoundError:
    raise SystemExit(
        "ERROR: 'fraud_model.pkl' not found.\n"
        "Run train_model.py first to generate it."
    )

# Load the optimal threshold saved by train_model.py (default 0.5 if missing)
THRESHOLD = 0.5
FEATURE_NAMES = list(model.feature_names_in_)
try:
    with open('model_config.json') as f:
        config = json.load(f)
        THRESHOLD = config.get('threshold', 0.5)
        print(f"Using decision threshold: {THRESHOLD}")
except FileNotFoundError:
    print("model_config.json not found — using default threshold 0.5")


# --- Database Setup ---
def get_db():
    """Returns a DB connection using context manager — no connection leaks."""
    conn = sqlite3.connect('transactions.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS transaction_history (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp  TEXT    NOT NULL,
                amount     REAL    NOT NULL,
                prediction TEXT    NOT NULL,
                probability REAL   NOT NULL
            )
        ''')
        conn.commit()

init_db()


# --- Input Validation ---
def validate_transaction(data: dict):
    """
    Checks that all 30 required features are present and numeric.
    Returns (True, None) on success or (False, error_message) on failure.
    """
    required = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
    missing = [f for f in required if f not in data]
    if missing:
        return False, f"Missing required fields: {missing}"

    non_numeric = []
    for field in required:
        try:
            float(data[field])
        except (TypeError, ValueError):
            non_numeric.append(field)
    if non_numeric:
        return False, f"Non-numeric values in fields: {non_numeric}"

    return True, None


# --- Routes ---

@app.route('/predict', methods=['POST'])
def predict_fraud():
    try:
        data = request.get_json(force=True)
        if data is None:
            return jsonify({'status': 'error', 'message': 'No JSON body provided'}), 400

        # FIX 1: Validate input before touching the model
        valid, err = validate_transaction(data)
        if not valid:
            return jsonify({'status': 'error', 'message': err}), 400

        df = pd.DataFrame([data])[FEATURE_NAMES]
        fraud_probability = float(model.predict_proba(df)[0][1])

        # FIX 2: Use optimal threshold instead of hardcoded 0.5
        prediction = "Fraud" if fraud_probability >= THRESHOLD else "Genuine"

        # FIX 3: Use context manager — connection always closes, even on error
        with get_db() as conn:
            conn.execute(
                'INSERT INTO transaction_history (timestamp, amount, prediction, probability) '
                'VALUES (?, ?, ?, ?)',
                (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                 float(data.get('Amount', 0)),
                 prediction,
                 round(fraud_probability, 4))
            )
            conn.commit()

        return jsonify({
            'status': 'success',
            'prediction': prediction,
            'fraud_probability': round(fraud_probability, 4),
            'threshold_used': THRESHOLD
        })

    except Exception as e:
        # FIX 4: Return 500 so the frontend knows something went wrong
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/transactions', methods=['GET'])
def get_transactions():
    try:
        # FIX 5: Support ?limit=N query param (default 20, max 100)
        limit = min(int(request.args.get('limit', 20)), 100)

        with get_db() as conn:
            rows = conn.execute(
                'SELECT * FROM transaction_history ORDER BY id DESC LIMIT ?',
                (limit,)
            ).fetchall()

        return jsonify({
            'status': 'success',
            'transactions': [dict(row) for row in rows]
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


# FIX 6: New /stats endpoint — gives the frontend real numbers for KPI cards
@app.route('/stats', methods=['GET'])
def get_stats():
    try:
        with get_db() as conn:
            total = conn.execute(
                'SELECT COUNT(*) FROM transaction_history'
            ).fetchone()[0]

            fraud_count = conn.execute(
                "SELECT COUNT(*) FROM transaction_history WHERE prediction='Fraud'"
            ).fetchone()[0]

            fraud_amount = conn.execute(
                "SELECT COALESCE(SUM(amount), 0) FROM transaction_history "
                "WHERE prediction='Fraud'"
            ).fetchone()[0]

            genuine_count = total - fraud_count

        fraud_rate = round((fraud_count / total * 100), 4) if total > 0 else 0

        return jsonify({
            'status': 'success',
            'total_transactions': total,
            'genuine_count': genuine_count,
            'fraud_count': fraud_count,
            'fraud_rate_percent': fraud_rate,
            'total_fraud_amount': round(fraud_amount, 2)
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


if __name__ == '__main__':
    # FIX 7: debug mode controlled by environment variable, not hardcoded
    debug_mode = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    app.run(debug=debug_mode, port=5000)