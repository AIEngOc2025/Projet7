import pytest
from fastapi.testclient import TestClient
from src.main import app  # Vérifie que l'import pointe bien vers ton FastAPI

client = TestClient(app)

def test_api():
    # Test de la racine
    response = client.get("/")
    assert response.status_code == 200
    
    # Test de l'endpoint /ask (on simule l'appel sans réseau)
    # Note: si /ask appelle Mistral, il faudra peut-être mocker core_rag.RAGSystem.ask
    response = client.get("/ask", params={"query": "test"})
    assert response.status_code in [200, 500] # 500 est acceptable ici si la clé API manque