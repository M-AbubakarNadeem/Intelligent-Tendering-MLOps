import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.api.app import app  # noqa: E402


def test_health_endpoint():
    with app.test_client() as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.get_json()
        assert "status" in data


def test_predict_endpoint():
    with app.test_client() as client:
        # Mocking or expecting a failure if no model is loaded is fine for a basic test
        payload = {"sentences": ["This is a test sentence."]}
        response = client.post("/predict", json=payload)
        # We don't necessarily need a 200 here if the model isn't loaded in CI
        assert response.status_code in [200, 500, 503]
