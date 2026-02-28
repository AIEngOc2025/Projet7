import pytest
from fastapi.testclient import TestClient
from src.main import app  # Assure-toi que l'import pointe vers ton instance FastAPI

# On utilise TestClient au lieu de requests
client = TestClient(app)

def test_api():
    """
    Teste l'accès à l'API sans avoir besoin de lancer le serveur manuellement.
    """
    # 1. Test de la racine (Healthcheck)
    response = client.get("/")
    assert response.status_code == 200

    # 2. Test de l'endpoint /ask
    # On simule l'envoi du JSON
    payload = {"question": "Quels sont les événements à Paris ?"}
    
    # On utilise client.post (interne) au lieu de requests.post (réseau)
    response = client.post("/ask", json=payload)
    
    # Si ton RAG a besoin d'une clé API pour répondre, 
    # ce test passera car le TestClient utilise les variables d'environnement de la CI.
    assert response.status_code == 200
    assert "response" in response.json() or "answer" in response.json()
