import os
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

def fetch_and_vectorize():
    # --- 1. RÉCUPÉRATION ---
    url = "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records"
    one_year_ago = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    params = {
        "limit": 100, # Augmenté pour un jeu de données plus riche
        "where": f"updatedat >= date'{one_year_ago}'",
        "refine": ["location_city:Paris"]
    }

    print(f"- Récupération des données OpenAgenda...")
    response = requests.get(url, params=params)
    results = response.json().get('results', [])

    # --- 2. STRUCTURATION INITIALE ---
    # données + métadonnées dans un format brut pour le découpage
    raw_documents = []
    for res in results:
        content = (
            f"Titre: {res.get('title_fr') or 'Sans titre'}\n"
            f"Lieu: {res.get('location_name') or 'Lieu non précisé'}\n"
            f"Date: {res.get('firstdate_begin') or 'Date non précisée'}\n"
            f"Description: {res.get('description_fr') or ''}"
        )
        metadata = {
            "uid": res.get('uid'),
            "date": res.get('firstdate_begin'),
            "titre": res.get('title_fr')
        }
        raw_documents.append(Document(page_content=content, metadata=metadata))

    # --- 3. DÉCOUPAGE EN CHUNKS ---
    # objet text_splitter : On découpe pour s'assurer que chaque vecteur est précis
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, 
        chunk_overlap=80
    )
    documents = text_splitter.split_documents(raw_documents)
    print(f"- Texte découpé en {len(documents)} chunks.")

    # --- 4. VECTORISATION MISTRAL & FAISS ---
    print("- Vectorisation avec Mistral-Embed...")
    embeddings = MistralAIEmbeddings(
        model="mistral-embed",
        mistral_api_key=os.getenv("MISTRAL_API_KEY")
    )
    
    vector_db = FAISS.from_documents(documents, embeddings)
    
    # --- 5. SAUVEGARDE ---
    vector_db.save_local("./data/vdb_paris")
    print(f"✅ BDD vectorielle prête dans 'data/vdb_paris'.")

if __name__ == "__main__":
    fetch_and_vectorize()