import os
import pytest
from fastapi.testclient import TestClient

from src.app.api import app
from src.app.classes import ImagesType

@pytest.fixture(scope="module", autouse=True)
def client():
    with TestClient(app) as client:
        return client

def test_get_labels(client):
    response = client.get("/list_labels")
    assert response.status_code == 200
    data = response.json()
    assert [label.name for label in ImagesType] == data["data"]["classes"]

def test_predict_with_no_jpg(client):
    data = {"file": ("invalid.txt", b"Test1", "text/plain")}
    response = client.post("/prediction", files=data)
    assert response.status_code == 400
    assert "Invalid file, only JPG allowed" in response.text

def test_prediction_with_jpg(client):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, "../data/test/airplane/0000.jpg")
    data = {"file": (image_path, open(image_path, "rb"), "image/jpg")}
    res = client.post("/prediction", files=data)
    assert res.status_code == 200
    data = res.json()
    assert "probabilities" in data["data"]
    assert "predicted_class" in data["data"]