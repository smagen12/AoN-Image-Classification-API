from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_model_info():
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "model_type" in data
    assert "framework" in data
    assert "input_shape" in data
    assert "labels" in data

def test_predict():
    with open("tests/test_image.png", "rb") as img:
        response = client.post(
            "/predict",
            files={"file": ("test_image.png", img, "image/png")}
        )
    assert response.status_code == 200
    data = response.json()
    assert "label" in data
    assert "confidence" in data
