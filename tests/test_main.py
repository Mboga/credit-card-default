import pytest
import numpy as np  # <--- Add this import
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import app.main
from app.main import app as fastapi_app

client = TestClient(fastapi_app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "online"

def test_prediction_flow_mocked():
    # 1. Setup the Mock Model using a REAL numpy array
    mock_model = MagicMock()
    # We use np.array so that [0, 1] indexing actually works!
    mock_model.predict_proba.return_value = np.array([[0.8, 0.2]])
    
    # 2. Setup Mock Metrics
    app.main.model = mock_model
    app.main.PREDICTION_COUNT = MagicMock()
    app.main.CREDIT_LIMIT_TRACKER = MagicMock()

    # 3. Patch extract_features
    with patch("app.main.extract_features", side_effect=lambda x: x):
        payload = {
            "LIMIT_BAL": 50000.0, "SEX": 1, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 30,
            "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
            "BILL_AMT1": 2000.0, "BILL_AMT2": 1500.0, "BILL_AMT3": 1000.0,
            "BILL_AMT4": 500.0, "BILL_AMT5": 200.0, "BILL_AMT6": 0.0,
            "PAY_AMT1": 500.0, "PAY_AMT2": 500.0, "PAY_AMT3": 500.0,
            "PAY_AMT4": 500.0, "PAY_AMT5": 200.0, "PAY_AMT6": 0.0
        }

        response = client.post("/predict", json=payload)
    
    # Debug print if it still fails
    if response.status_code != 200:
        print(f"\nSTILL FAILING: {response.json()}")

    assert response.status_code == 200
    assert response.json()["default_probability"] == 0.2
    assert response.json()["decision"] == "APPROVED"