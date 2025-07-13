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
    assert "model_type" in response.json()
    assert "framework" in response.json()
    assert "input_shape" in response.json()

def test_predict():
    with open("tests/test_image.png", "rb") as img:
        response = client.post(
            "/predict",
            files={"file": img},
            params={"weights": "resnet_v1.pth"}
        )
    assert response.status_code == 200
    assert "label" in response.json()
    assert "confidence" in response.json()
