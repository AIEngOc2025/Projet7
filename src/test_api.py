import pytest
from fastapi.testclient import TestClient
from src.main import app  # Ajuste l'import selon ton arborescence

client = TestClient(app)

def test_api():
    # Le TestClient simule l'appel sans avoir besoin de 'uvicorn'
    response = client.get("/ask", params={"query": "C'est quoi Paris ?"})
    assert response.status_code == 200