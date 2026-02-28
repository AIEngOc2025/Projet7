import os
from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

def test_vector_db():
    print("- Chargement de la base de données vectorielle...")
    
    # 1. Initialiser le même modèle d'embedding qu'à l'indexation
    embeddings = MistralAIEmbeddings(
        model="mistral-embed",
        mistral_api_key=os.getenv("MISTRAL_API_KEY")
    )

    # 2. Charger l'index (Attention au paramètre allow_dangerous_deserialization)
    try:
        vector_db = FAISS.load_local(
            "./data/vdb_paris", 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        print("Chargement réussi.\n")
    except Exception as e:
        print(f"❌ Erreur lors du chargement : {e}")
        return

    # 3. Effectuer une recherche de test
    query = "Quels sont les détails de l'événement sur la cartographie des startups en IA prévu en mars 2026 ?"
    query = "Exposition ou vernissage sur l'art contemporain à Paris"
    print(f"- Recherche sémantique pour : '{query}'")
    
    # On récupère les 3 résultats les plus proches
    results = vector_db.similarity_search(query, k=3)

    # 4. Affichage des résultats
    if not results:
        print("⚠️ Aucun résultat trouvé.")
    else:
        for i, doc in enumerate(results):
            print(f"\n--- Résultat #{i+1} ---")
            print(f"📍 Titre : {doc.metadata.get('titre', 'N/A')}")
            print(f"📅 Date  : {doc.metadata.get('date', 'N/A')}")
            print(f"📖 Extrait : {doc.page_content[:150]}...")
            

if __name__ == "__main__":
    test_vector_db()