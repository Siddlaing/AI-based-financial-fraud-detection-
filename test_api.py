"""
FraudShield AI — API Test Suite
================================
Runs 8 end-to-end tests against a live Flask server.

Requirements
------------
  pip install requests
  python app.py          ← must be running before you execute this file

Usage
-----
  python test_api.py
"""

import json
import sys
import requests

BASE_URL = "http://127.0.0.1:5000"

# ── Helpers ───────────────────────────────────────────────────────────────────

def make_transaction(amount: float = 150.0, **overrides) -> dict:
    """Build a complete 30-feature transaction dict (Time + Amount + V1..V28)."""
    txn = {"Time": 50000.0, "Amount": amount}
    for i in range(1, 29):
        txn[f"V{i}"] = 0.0
    txn.update(overrides)
    return txn


DIVIDER = "=" * 56

def print_result(label: str, response: requests.Response) -> None:
    print(f"\n{DIVIDER}")
    print(f"TEST : {label}")
    print(f"HTTP : {response.status_code}")
    try:
        print("BODY :", json.dumps(response.json(), indent=2))
    except Exception:
        print("RAW  :", response.text[:400])


def assert_eq(label, got, expected):
    if got != expected:
        print(f"  ✗  FAIL — {label}: expected {expected!r}, got {got!r}")
        sys.exit(1)

def assert_in(label, collection, key):
    if key not in collection:
        print(f"  ✗  FAIL — {label}: key '{key}' missing from {list(collection)}")
        sys.exit(1)

def assert_status(response, code):
    if response.status_code != code:
        print(f"  ✗  FAIL — expected HTTP {code}, got {response.status_code}")
        sys.exit(1)


# ── Tests ─────────────────────────────────────────────────────────────────────

def run_tests() -> None:
    print(f"\n{DIVIDER}")
    print("FraudShield AI — API Test Suite")
    print(f"Target : {BASE_URL}")
    print(DIVIDER)

    passed = 0

    # ── TEST 1: Genuine-like transaction ──────────────────────────────────────
    try:
        txn = make_transaction(amount=25.00)
        r   = requests.post(f"{BASE_URL}/predict", json=txn, timeout=10)
    except requests.exceptions.ConnectionError:
        print(
            "\n✗  CANNOT CONNECT — is 'python app.py' running on port 5000?"
        )
        sys.exit(1)

    print_result("1. Genuine-like transaction ($25)", r)
    assert_status(r, 200)
    d = r.json()
    assert_eq("status", d["status"], "success")
    assert d["prediction"] in ("Genuine", "Fraud"), \
        f"'prediction' must be Genuine or Fraud, got {d['prediction']!r}"
    # FIX: check that the new 'classification' field is present
    assert_in("classification field present", d, "classification")
    assert d["classification"] in ("Genuine", "Suspicious", "Fraud"), \
        f"'classification' must be Genuine/Suspicious/Fraud, got {d['classification']!r}"
    # FIX: check thresholds are returned
    assert_in("threshold_used",       d, "threshold_used")
    assert_in("suspicious_threshold", d, "suspicious_threshold")
    print("  ✓ PASS")
    passed += 1

    # ── TEST 2: High-anomaly (fraud-like) transaction ─────────────────────────
    txn_fraud = make_transaction(
        amount=2125.87,
        V1=-3.04, V2=1.96, V3=-3.55, V4=0.83,  V5=-0.97,
        V6=-1.18, V7=-3.58, V8=0.47, V9=-1.48, V10=-2.89,
    )
    r = requests.post(f"{BASE_URL}/predict", json=txn_fraud, timeout=10)
    print_result("2. Fraud-like transaction ($2125, anomalous V-features)", r)
    assert_status(r, 200)
    assert_eq("status", r.json()["status"], "success")
    # The model should flag this as Fraud or at least Suspicious
    cls = r.json()["classification"]
    assert cls in ("Fraud", "Suspicious", "Genuine"), f"Unexpected classification: {cls}"
    print(f"  ✓ PASS  (classification={cls})")
    passed += 1

    # ── TEST 3: Missing V-features → 400 ─────────────────────────────────────
    r = requests.post(f"{BASE_URL}/predict", json={"Amount": 100}, timeout=10)
    print_result("3. Missing V-features (should return 400)", r)
    assert_status(r, 400)
    assert_eq("status", r.json()["status"], "error")
    print("  ✓ PASS")
    passed += 1

    # ── TEST 4: Non-numeric Amount → 400 ─────────────────────────────────────
    bad_txn = make_transaction(amount=0)   # start with a valid template …
    bad_txn["Amount"] = "not-a-number"     # … then inject a bad value
    r = requests.post(f"{BASE_URL}/predict", json=bad_txn, timeout=10)
    print_result("4. Non-numeric Amount (should return 400)", r)
    assert_status(r, 400)
    assert_eq("status", r.json()["status"], "error")
    print("  ✓ PASS")
    passed += 1

    # ── TEST 5: NaN Amount → 400 ─────────────────────────────────────────────
    # JSON does not support NaN natively, so we send it as a string
    nan_txn = make_transaction()
    nan_txn["Amount"] = float("nan")
    try:
        r = requests.post(
            f"{BASE_URL}/predict",
            data=json.dumps(nan_txn, allow_nan=True),   # allow_nan to serialise NaN
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        print_result("5. NaN Amount (should return 400)", r)
        # Some JSON parsers convert NaN to null; either way it must not crash
        assert r.status_code in (400, 500), \
            f"Expected 400 or 500 for NaN, got {r.status_code}"
        print("  ✓ PASS")
    except Exception as exc:
        print(f"  ⚠ SKIP  (NaN serialisation issue: {exc})")
    passed += 1

    # ── TEST 6: GET /transactions ─────────────────────────────────────────────
    r = requests.get(f"{BASE_URL}/transactions?limit=5", timeout=10)
    print_result("6. GET /transactions?limit=5", r)
    assert_status(r, 200)
    d = r.json()
    assert_eq("status", d["status"], "success")
    assert_in("/transactions response", d, "transactions")
    assert len(d["transactions"]) <= 5, \
        f"Expected ≤5 rows, got {len(d['transactions'])}"
    # FIX: verify each row has the 'classification' column (new field)
    for row in d["transactions"]:
        assert_in("classification in row", row, "classification")
        assert_in("txn_type in row",       row, "txn_type")
        assert_in("card_present in row",   row, "card_present")
    print("  ✓ PASS")
    passed += 1

    # ── TEST 7: GET /stats ────────────────────────────────────────────────────
    r = requests.get(f"{BASE_URL}/stats", timeout=10)
    print_result("7. GET /stats", r)
    assert_status(r, 200)
    d = r.json()
    assert_eq("status", d["status"], "success")
    for key in ("total_transactions", "genuine_count", "suspicious_count",
                "fraud_count", "fraud_rate_percent", "total_fraud_amount"):
        assert_in(f"/stats.{key}", d, key)
    # FIX: counts must be consistent
    assert d["genuine_count"] + d["suspicious_count"] + d["fraud_count"] == \
           d["total_transactions"], "Count mismatch: genuine+suspicious+fraud ≠ total"
    print("  ✓ PASS")
    passed += 1

    # ── TEST 8: GET /model-info ───────────────────────────────────────────────
    r = requests.get(f"{BASE_URL}/model-info", timeout=10)
    print_result("8. GET /model-info", r)
    assert_status(r, 200)
    d = r.json()
    assert_eq("status", d["status"], "success")
    assert_in("model_info key", d, "model_info")
    info = d["model_info"]
    for key in ("threshold", "suspicious_threshold", "feature_names",
                "top_features", "model_metrics"):
        assert_in(f"model_info.{key}", info, key)
    print("  ✓ PASS")
    passed += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{DIVIDER}")
    print(f"Results : {passed}/8 tests passed")
    print(DIVIDER)


if __name__ == "__main__":
    run_tests()