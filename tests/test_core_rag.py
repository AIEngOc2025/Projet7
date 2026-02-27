import os
import sys
import pytest

# ensure the package root is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

def test_clean_html():
    from src.core_rag import clean_html

    assert clean_html("<p>hello</p>") == "hello"
    assert clean_html("<div><b>test</b></div>") == "test"
    assert clean_html("") == ""
    assert clean_html(None) == ""


def test_rag_without_index(tmp_path, monkeypatch):
    """Si le répertoire d'index n'existe pas ou est vide, la base reste None."""
    from src.core_rag import RAGSystem

    # instanciation avec un chemin vide
    r = RAGSystem(index_path=str(tmp_path / "empty"))
    assert r.vector_db is None
    assert r.ask("quoi") == "La base de données est vide. Veuillez appeler /rebuild d'abord."
    # _get_relevant_docs doit retourner une chaîne vide en l'absence de DB
    assert r._get_relevant_docs("test") == ""


def test_rebuild_database_creates_index(tmp_path, monkeypatch):
    """On simule l'API pour vérifier que l'index est construit et queryable."""
    from src.core_rag import RAGSystem

    fake_data = {
        "results": [
            {
                "title_fr": "Événement factice",
                "location_name": "Salle X",
                "location_city": "Paris",
                "firstdate_begin": "2025-06-01",
                "description_fr": "<b>Description</b>"
            }
        ]
    }

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return fake_data

    monkeypatch.setattr("src.core_rag.requests.get", lambda url, params: FakeResponse())

    r = RAGSystem(index_path=str(tmp_path / "db"))
    r.rebuild_database()
    # l'index doit désormais exister
    assert r.vector_db is not None
    docs = r._get_relevant_docs("Événement")
    assert "ÉVÉNEMENT" in docs
