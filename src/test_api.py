from fastapi.testclient import TestClient
from src.main import app  # Adapte l'import selon ta structure

client = TestClient(app)

def test_api():
    # On teste la racine
    response = client.get("/")
    assert response.status_code == 200
    
    # On teste /ask sans que le serveur uvicorn ne soit lancé
    # Note : Si le RAG n'est pas initialisé, l'API renvoie souvent 200 avec un message d'erreur
    response = client.get("/ask", params={"query": "test"})
    assert response.status_code == 200