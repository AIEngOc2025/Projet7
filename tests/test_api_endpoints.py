import os
import sys
import pytest
from fastapi.testclient import TestClient

# ensure project root is on path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.main import app, get_rag


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
    # obtenir une instance de RAGSystem et forcer la vector_db à None
    rag_instance = get_rag()
    monkeypatch.setattr(rag_instance, "vector_db", None)
    resp = client.post("/ask", json={"question": "quelque chose"})
    assert resp.status_code == 200
    assert "base de données" in resp.json()["answer"]


def test_rebuild_endpoint(client, monkeypatch):
    # obtenir une instance et monkeypatcher sa méthode pour éviter appel réseau
    rag_instance = get_rag()
    monkeypatch.setattr(rag_instance, "rebuild_database", lambda: None)
    resp = client.post("/rebuild")
    assert resp.status_code == 200
    assert "Reconstruction" in resp.json().get("message", "")
