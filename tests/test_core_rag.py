import os
import sys
import pytest
from unittest.mock import patch, MagicMock

# Configuration du chemin racine
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

def test_clean_html():
    from src.core_rag import clean_html
    assert clean_html("<p>hello</p>") == "hello"
    assert clean_html("<div><b>test</b></div>") == "test"
    assert clean_html("") == ""
    assert clean_html(None) == ""

def test_rag_without_index(tmp_path):
    """Si le répertoire d'index n'existe pas ou est vide, la base reste None."""
    from src.core_rag import RAGSystem
    
    # On initialise avec un dossier temporaire vide
    r = RAGSystem(index_path=str(tmp_path / "empty"))
    assert r.vector_db is None
    assert "Veuillez appeler /rebuild" in r.ask("quoi")
    assert r._get_relevant_docs("test") == ""

def test_rebuild_database_creates_index(tmp_path, monkeypatch):
    """Vérifie la construction de l'index en simulant l'API Paris et Mistral."""
    from src.core_rag import RAGSystem

    # 1. Simulation des données de l'API Paris
    fake_data = {
        "results": [{
            "title_fr": "Événement factice",
            "location_name": "Salle X",
            "location_city": "Paris",
            "firstdate_begin": "2025-06-01",
            "description_fr": "<b>Description</b>"
        }]
    }

    class FakeResponse:
        def raise_for_status(self): pass
        def json(self): return fake_data

    # 2. Simulation de l'appel réseau API Paris
    monkeypatch.setattr("requests.get", lambda url, params=None, timeout=None: FakeResponse())

    # 3. MOCK CRITIQUE : On simule les Embeddings de Mistral
    # Cela évite l'erreur "ValidationError" ou "Connection Error" sur GitHub
    with patch('src.core_rag.MistralAIEmbeddings') as mock_embeddings_class:
        # On crée une instance fictive qui renvoie des vecteurs de nombres
        mock_instance = MagicMock()
        mock_instance.embed_documents.return_value = [[0.1] * 1024] 
        mock_instance.embed_query.return_value = [0.1] * 1024
        mock_embeddings_class.return_value = mock_instance

        # Exécution
        r = RAGSystem(index_path=str(tmp_path / "db"))
        r.rebuild_database()

        # Vérifications
        assert r.vector_db is not None
        docs = r._get_relevant_docs("Événement")
        assert "ÉVÉNEMENT" in docs.upper()