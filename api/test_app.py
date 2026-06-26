from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "status" in response.json()
    assert response.json()["status"] == "ok"

def test_predict_endpoint_without_model():
    # If the model isn't loaded (which might happen in a fresh CI environment),
    # it should return 503.
    response = client.post("/predict", json={"message": "Free money now!"})
    if response.status_code == 503:
        assert response.json()["detail"] == "Model or vectorizer not fully loaded"
    else:
        assert response.status_code == 200
        assert "is_spam" in response.json()
