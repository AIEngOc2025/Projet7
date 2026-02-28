import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Import de ton application FastAPI
from src.main import app 

client = TestClient(app)

def test_read_root():
    """Vérifie que l'API répond bien sur la racine."""
    response = client.get("/")
    assert response.status_code == 200
    assert "status" in response.json()

def test_ask_endpoint_mocked():
    """Teste l'endpoint /ask sans appeler l'IA réelle."""
    # On mocke la méthode 'ask' de la classe RAGSystem dans src.main
    with patch("src.main.rag_system.ask") as mock_ask:
        mock_ask.return_value = "Ceci est une réponse simulée pour le test."
        
        response = client.get("/ask", params={"query": "Quelle heure est-il ?"})
        
        assert response.status_code == 200
        assert response.json()["response"] == "Ceci est une réponse simulée pour le test."

def test_rebuild_endpoint_mocked():
    """Teste l'endpoint /rebuild sans reconstruire réellement la DB."""
    with patch("src.main.rag_system.rebuild_database") as mock_rebuild:
        mock_rebuild.return_value = True
        
        response = client.post("/rebuild")
        
        assert response.status_code == 200
        assert "Base de données reconstruite" in response.json()["message"]