"""
Récupère et indexe les données d'énènements culturels à Paris depuis l'API OpenDataSoft, 
en utilisant le modèle de vectorisation Mistral-Embed.
"""

#%% Récupération des données et indexation avec Mistral-Embed
import os
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Nouveau : Import spécifique pour Mistral
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

load_dotenv()

def fetch_and_vectorize():
    # --- 1. CONFIGURATION & RÉCUPÉRATION ---
    url = "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records"
    
    # Calcul de la date : 1 an d'historique
    one_year_ago = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    params = {
        "limit": 50,
        "where": f"firstdate_begin >= date'{one_year_ago}'",
        "refine": ["location_city:Paris", "location_countrycode:FR"]
    }

    print(f"- Récupération des données depuis le {one_year_ago}...")
    response = requests.get(url, params=params)
    results = response.json().get('results', [])

    if not results:
        print("⚠️ Aucun événement trouvé.")
        return

    # --- 2. STRUCTURATION & NETTOYAGE ---
    documents = []
    for res in results:
        # Nettoyage des valeurs None pour éviter les plantages lors de la vectorisation
        titre = res.get('title_fr') or "Sans titre"
        lieu = res.get('location_name') or "Lieu non précisé"
        date_ev = res.get('firstdate_begin') or "Date non communiquée"
        desc = res.get('description_fr') or "Pas de description disponible."

        content = (
            f"Titre: {titre}\n"
            f"Lieu: {lieu}\n"
            f"Date: {date_ev}\n"
            f"Description: {desc}"
        )
        
        metadata = {
            "uid": res.get('uid'),
            "date": date_ev,
            "city": res.get('location_city', 'Paris')
        }
        documents.append(Document(page_content=content, metadata=metadata))

    # --- 3. VECTORISATION AVEC MISTRAL (Conformément à la consigne) ---
    print("- Conversion vectorielle avec le modèle Mistral-Embed...")
    
    # creation de l'instance d'embeddings MistralAIEmbeddings avec la clé API
    embeddings = MistralAIEmbeddings(
        model="mistral-embed",
        mistral_api_key=os.getenv("MISTRAL_API_KEY")
    )
    
    # --- 4. CRÉATION & SAUVEGARDE DE L'INDEX FAISS ---
    vector_db = FAISS.from_documents(documents, embeddings)
    
    vector_db.save_local("./data/vdb_paris50_chunk0")
    print(f"{len(documents)} événements indexés avec succès dans 'data/vdb_paris50_chunk0'.")

if __name__ == "__main__":
    fetch_and_vectorize()