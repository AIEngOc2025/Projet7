import os
import sys
import pytest
from fastapi.testclient import TestClient

# ensure project root is on path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.main import app, rag


@pytest.fixture(autouse=True)
def client():
    return TestClient(app)


def test_home(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "API RAG opérationnelle" in resp.json().get("message", "")


def test_ask_invalid_question(client):
    resp = client.post("/ask", json={"question": "   "})
    assert resp.status_code == 400


def test_ask_empty_db(client, monkeypatch):
    # forcer la vector_db à None pour simuler l'index non construit
    monkeypatch.setattr(rag, "vector_db", None)
    resp = client.post("/ask", json={"question": "quelque chose"})
    assert resp.status_code == 200
    assert "base de données" in resp.json()["answer"]


def test_rebuild_endpoint(client, monkeypatch):
    # monkeypatch de la méthode pour éviter appel réseau
    monkeypatch.setattr(rag, "rebuild_database", lambda: None)
    resp = client.post("/rebuild")
    assert resp.status_code == 200
    assert "Reconstruction" in resp.json().get("message", "")
