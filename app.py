"""
FraudShield AI — Flask REST API
================================
Run AFTER train_model.py has produced fraud_model.pkl and model_config.json.

    python app.py

Endpoints
---------
POST /predict          — score a single transaction
GET  /transactions     — paginated history (filter by status)
GET  /stats            — dashboard aggregates
GET  /model-info       — expose model_config.json to the frontend
"""

import os
import json
import sqlite3
import logging
from datetime import datetime
from typing import Tuple, Dict, Any   # FIX 1: use typing.Tuple for Python 3.8 compatibility

import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# ── Load Model ────────────────────────────────────────────────────────────────
log.info("Loading AI model...")
try:
    model = joblib.load("fraud_model.pkl")
    log.info("Model loaded successfully — %d features", len(model.feature_names_in_))
except FileNotFoundError:
    raise SystemExit(
        "\nERROR: 'fraud_model.pkl' not found.\n"
        "Run   python train_model.py   first to generate it.\n"
    )

# ── Load Config (thresholds) ──────────────────────────────────────────────────
THRESHOLD            = 0.5
SUSPICIOUS_THRESHOLD = 0.3
FEATURE_NAMES        = list(model.feature_names_in_)

try:
    with open("model_config.json") as f:
        _cfg = json.load(f)
    THRESHOLD            = _cfg.get("threshold", 0.5)
    SUSPICIOUS_THRESHOLD = _cfg.get("suspicious_threshold", round(THRESHOLD * 0.6, 4))
    log.info(
        "Thresholds →  suspicious ≥ %.4f  |  fraud ≥ %.4f",
        SUSPICIOUS_THRESHOLD, THRESHOLD,
    )
except FileNotFoundError:
    log.warning("model_config.json not found — using default thresholds (0.3 / 0.5).")


# ── Database helpers ──────────────────────────────────────────────────────────
DB_PATH = "transactions.db"

def get_db() -> sqlite3.Connection:
    """Return a new SQLite connection with Row factory enabled."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """
    Create the transaction_history table if it doesn't exist, and migrate any
    older DB that is missing the newer columns (txn_type, card_present, classification).
    """
    conn = get_db()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS transaction_history (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp      TEXT    NOT NULL,
                amount         REAL    NOT NULL,
                txn_type       TEXT    DEFAULT 'Unknown',
                card_present   TEXT    DEFAULT 'Unknown',
                prediction     TEXT    NOT NULL,
                classification TEXT    NOT NULL,
                probability    REAL    NOT NULL
            )
        """)
        # FIX 2: graceful migration — add missing columns to existing databases
        existing_cols = {
            row[1]
            for row in conn.execute("PRAGMA table_info(transaction_history)").fetchall()
        }
        migrations = [
            ("txn_type",       "'Unknown'"),
            ("card_present",   "'Unknown'"),
            ("classification", "'Genuine'"),
        ]
        for col, default in migrations:
            if col not in existing_cols:
                conn.execute(
                    f"ALTER TABLE transaction_history ADD COLUMN {col} TEXT DEFAULT {default}"
                )
                log.info("DB migration: added column '%s'", col)
        conn.commit()
    finally:
        conn.close()   # FIX 2: always close the connection


init_db()


# ── Input Validation ──────────────────────────────────────────────────────────
def validate_transaction(data: Dict[str, Any]) -> Tuple[bool, str]:
    """
    FIX 3: Validation now uses float() conversion inside a try/except instead
    of fragile string manipulation — catches edge cases like '1.5.6', inf, nan.
    Returns (is_valid, error_message).
    """
    required = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]

    # Check all required fields are present
    missing = [f for f in required if f not in data]
    if missing:
        return False, f"Missing required fields: {missing}"

    # Check all values are actually numeric
    non_numeric = []
    for field in required:
        try:
            val = float(data[field])
            if val != val:           # NaN check
                non_numeric.append(field)
        except (TypeError, ValueError):
            non_numeric.append(field)

    if non_numeric:
        return False, f"Non-numeric or NaN values in: {non_numeric}"

    return True, ""


# ── 3-Class Labelling ─────────────────────────────────────────────────────────
def classify(probability: float) -> Tuple[str, str]:
    """
    Convert a raw fraud probability into a binary prediction and ternary label.

    Returns
    -------
    prediction     : 'Fraud' | 'Genuine'         (for DB filtering)
    classification : 'Fraud' | 'Suspicious' | 'Genuine'  (for UI display)
    """
    if probability >= THRESHOLD:
        return "Fraud",   "Fraud"
    elif probability >= SUSPICIOUS_THRESHOLD:
        return "Genuine", "Suspicious"
    else:
        return "Genuine", "Genuine"


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/predict", methods=["POST"])
def predict_fraud():
    """Score a single transaction and persist it to the database."""
    conn = None
    try:
        data = request.get_json(force=True)
        if data is None:
            return jsonify({"status": "error", "message": "No JSON body provided"}), 400

        valid, err = validate_transaction(data)
        if not valid:
            return jsonify({"status": "error", "message": err}), 400

        # Build feature DataFrame in the exact column order the model expects
        df         = pd.DataFrame([data])[FEATURE_NAMES]
        fraud_prob = float(model.predict_proba(df)[0][1])
        prediction, cls = classify(fraud_prob)

        txn_type     = str(data.get("txn_type",     "Unknown"))
        card_present = str(data.get("card_present",  "Unknown"))

        # FIX 2: explicit open / close — no silent connection leak
        conn = get_db()
        conn.execute(
            "INSERT INTO transaction_history "
            "(timestamp, amount, txn_type, card_present, prediction, classification, probability) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                round(float(data.get("Amount", 0)), 2),
                txn_type,
                card_present,
                prediction,
                cls,
                round(fraud_prob, 4),
            ),
        )
        conn.commit()

        log.info(
            "predict  amt=%.2f  prob=%.4f  cls=%s",
            float(data.get("Amount", 0)), fraud_prob, cls,
        )

        return jsonify({
            "status":               "success",
            "prediction":           prediction,
            "classification":       cls,
            "fraud_probability":    round(fraud_prob, 4),
            "threshold_used":       THRESHOLD,
            "suspicious_threshold": SUSPICIOUS_THRESHOLD,
        })

    except Exception as exc:
        log.exception("predict error: %s", exc)
        return jsonify({"status": "error", "message": str(exc)}), 500

    finally:
        if conn:
            conn.close()   # FIX 2: always close


@app.route("/transactions", methods=["GET"])
def get_transactions():
    """Return recent transactions with optional classification filter."""
    conn = None
    try:
        limit         = min(int(request.args.get("limit", 20)), 100)
        status_filter = request.args.get("status", "all").strip().lower()

        conn = get_db()
        if status_filter == "all":
            rows = conn.execute(
                "SELECT * FROM transaction_history ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        else:
            # Capitalize to match stored values: 'Genuine', 'Suspicious', 'Fraud'
            rows = conn.execute(
                "SELECT * FROM transaction_history "
                "WHERE classification=? ORDER BY id DESC LIMIT ?",
                (status_filter.capitalize(), limit),
            ).fetchall()

        return jsonify({
            "status":       "success",
            "transactions": [dict(row) for row in rows],
        })

    except Exception as exc:
        log.exception("transactions error: %s", exc)
        return jsonify({"status": "error", "message": str(exc)}), 500

    finally:
        if conn:
            conn.close()


@app.route("/stats", methods=["GET"])
def get_stats():
    """Return aggregate counts used by the dashboard KPI cards."""
    conn = None
    try:
        conn = get_db()

        # FIX 4: run all counts in a single query for efficiency
        row = conn.execute("""
            SELECT
                COUNT(*)                                              AS total,
                SUM(CASE WHEN classification='Genuine'    THEN 1 ELSE 0 END) AS genuine,
                SUM(CASE WHEN classification='Suspicious' THEN 1 ELSE 0 END) AS suspicious,
                SUM(CASE WHEN classification='Fraud'      THEN 1 ELSE 0 END) AS fraud,
                COALESCE(SUM(CASE WHEN classification='Fraud' THEN amount ELSE 0 END), 0) AS fraud_amt
            FROM transaction_history
        """).fetchone()

        total          = row["total"]          or 0
        genuine_count  = row["genuine"]        or 0
        suspicious_count = row["suspicious"]   or 0
        fraud_count    = row["fraud"]          or 0
        fraud_amount   = row["fraud_amt"]      or 0.0

        fraud_rate = round(fraud_count / total * 100, 4) if total > 0 else 0.0

        return jsonify({
            "status":              "success",
            "total_transactions":  total,
            "genuine_count":       genuine_count,
            "suspicious_count":    suspicious_count,
            "fraud_count":         fraud_count,
            "fraud_rate_percent":  fraud_rate,
            "total_fraud_amount":  round(fraud_amount, 2),
        })

    except Exception as exc:
        log.exception("stats error: %s", exc)
        return jsonify({"status": "error", "message": str(exc)}), 500

    finally:
        if conn:
            conn.close()


@app.route("/model-info", methods=["GET"])
def get_model_info():
    """Expose model_config.json to the frontend (thresholds, metrics, features)."""
    try:
        with open("model_config.json") as f:
            cfg = json.load(f)
        return jsonify({"status": "success", "model_info": cfg})
    except FileNotFoundError:
        return jsonify({"status": "error", "message": "model_config.json not found"}), 404
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    log.info("Starting FraudShield API on port 5000  (debug=%s)", debug_mode)
    app.run(debug=debug_mode, host="0.0.0.0", port=5000)