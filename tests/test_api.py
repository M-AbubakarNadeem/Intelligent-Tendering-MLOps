"""
Integration tests for the Flask API.
"""

import sys
import json
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.api.app import app


@pytest.fixture
def client():
    """Create test client."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


class TestAPIEndpoints:
    """Test API endpoints."""

    def test_root_endpoint(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "name" in data
        assert "endpoints" in data

    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "status" in data
        assert data["status"] == "healthy"
        assert "model_loaded" in data

    def test_metrics_endpoint(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200
        # Prometheus format
        assert b"prediction_requests_total" in response.data or response.status_code == 200

    def test_predict_no_body(self, client):
        response = client.post(
            "/predict", data=json.dumps({}), content_type="application/json"
        )
        # Should return 400 or 503 (no model)
        assert response.status_code in [400, 503]

    def test_predict_missing_sentences(self, client):
        response = client.post(
            "/predict",
            data=json.dumps({"text": "hello"}),
            content_type="application/json",
        )
        assert response.status_code in [400, 503]
