"""
tests/test_backend_api.py

Unit and integration tests for FastAPI backend routes.
"""

from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_models_endpoint_returns_installed_models():
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert "default" in data
    assert isinstance(data["models"], list)
    assert len(data["models"]) > 0
    # Embedding models should be excluded
    for m in data["models"]:
        assert "embed" not in m.lower()


def test_documents_endpoint():
    response = client.get("/api/documents")
    assert response.status_code == 200
    assert isinstance(response.json(), dict)


def test_ask_stream_request_validation():
    # Verify AskRequest schema accepts list of filenames
    response = client.post(
        "/api/ask/stream",
        json={
            "question": "What is genetic algorithms?",
            "top_k": 3,
            "filename_filter": ["paper1.pdf", "paper2.pdf"],
        },
    )
    # The endpoint returns a 200 streaming response
    assert response.status_code == 200
