from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)

def test_read_root():
    """Test the root healthcheck endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["status"] == "healthy"
    assert "api_name" in json_data
    assert json_data["model_loaded"] is True

def test_predict_endpoint_ham():
    """Test the /predict endpoint with ham text."""
    payload = {"text": "Hey mate, hope you are doing good. Let me know when you are free."}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == "ham"
    assert data["ham_probability"] > data["spam_probability"]

def test_predict_endpoint_spam():
    """Test the /predict endpoint with spam text."""
    payload = {"text": "WINNER! You won a cash prize of 5000 pounds. Text CLAIM to 80100 to get it."}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == "spam"
    assert data["spam_probability"] > data["ham_probability"]

def test_predict_endpoint_empty():
    """Test the /predict endpoint error handling for empty text."""
    payload = {"text": "   "}
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    assert "detail" in response.json()
