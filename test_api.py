import requests
import json

BASE_URL = 'http://127.0.0.1:5000'

def make_transaction(amount=150.0, **overrides):
    """Helper: builds a full 30-feature transaction dict."""
    txn = {"Time": 50000.0, "Amount": amount}
    for i in range(1, 29):
        txn[f"V{i}"] = 0.0
    txn.update(overrides)
    return txn


def print_result(label, response):
    print(f"\n{'='*50}")
    print(f"TEST: {label}")
    print(f"Status Code : {response.status_code}")
    try:
        print(f"Response    : {json.dumps(response.json(), indent=2)}")
    except Exception:
        print(f"Raw         : {response.text}")


def run_tests():
    print("Starting API tests...\n")

    # ----------------------------------------------------------------
    # TEST 1: Normal (likely genuine) transaction
    # ----------------------------------------------------------------
    try:
        txn = make_transaction(amount=25.00)
        r = requests.post(f"{BASE_URL}/predict", json=txn)
        print_result("Genuine-like transaction ($25)", r)
        data = r.json()
        assert r.status_code == 200,          "Expected HTTP 200"
        assert data['status'] == 'success',   "Expected status=success"
        assert data['prediction'] in ('Genuine', 'Fraud'), "Invalid prediction value"
        print("PASS")
    except requests.exceptions.ConnectionError:
        print("FAIL: Cannot connect to Flask. Is app.py running?")
        return

    # ----------------------------------------------------------------
    # TEST 2: High-anomaly transaction (likely fraud)
    # V features below are from a known fraud sample in the Kaggle dataset
    # ----------------------------------------------------------------
    txn_fraud = make_transaction(
        amount=2125.87,
        V1=-3.04, V2=1.96, V3=-3.55, V4=0.83, V5=-0.97,
        V6=-1.18, V7=-3.58, V8=0.47, V9=-1.48, V10=-2.89
    )
    r = requests.post(f"{BASE_URL}/predict", json=txn_fraud)
    print_result("Fraud-like transaction ($2125, anomalous V-features)", r)
    assert r.status_code == 200
    assert r.json()['status'] == 'success'
    print("PASS")

    # ----------------------------------------------------------------
    # TEST 3: Missing fields — should return 400
    # ----------------------------------------------------------------
    r = requests.post(f"{BASE_URL}/predict", json={"Amount": 100})
    print_result("Missing V-features (should return 400)", r)
    assert r.status_code == 400,              "Expected HTTP 400 for bad input"
    assert r.json()['status'] == 'error',     "Expected status=error"
    print("PASS")

    # ----------------------------------------------------------------
    # TEST 4: Non-numeric value — should return 400
    # ----------------------------------------------------------------
    bad_txn = make_transaction(amount="not-a-number")
    r = requests.post(f"{BASE_URL}/predict", json=bad_txn)
    print_result("Non-numeric Amount (should return 400)", r)
    assert r.status_code == 400
    print("PASS")

    # ----------------------------------------------------------------
    # TEST 5: /transactions endpoint
    # ----------------------------------------------------------------
    r = requests.get(f"{BASE_URL}/transactions?limit=5")
    print_result("/transactions endpoint (limit=5)", r)
    assert r.status_code == 200
    data = r.json()
    assert data['status'] == 'success'
    assert 'transactions' in data
    assert len(data['transactions']) <= 5
    print("PASS")

    # ----------------------------------------------------------------
    # TEST 6: /stats endpoint
    # ----------------------------------------------------------------
    r = requests.get(f"{BASE_URL}/stats")
    print_result("/stats endpoint", r)
    assert r.status_code == 200
    data = r.json()
    assert data['status'] == 'success'
    assert 'total_transactions' in data
    assert 'fraud_count' in data
    print("PASS")

    print(f"\n{'='*50}")
    print("All tests passed!")


if __name__ == '__main__':
    run_tests()