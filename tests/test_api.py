# HEART-MLOPS/tests/test_api.py

from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_read_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "api_requests_total" in response.text