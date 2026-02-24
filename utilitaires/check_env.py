import os
from dotenv import load_dotenv

try:
    # 1. Gestion des vecteurs
    import faiss
    from langchain_community.vectorstores import FAISS
    
    # 2. Embeddings (Le chemin moderne via langchain-huggingface)
    from langchain_huggingface import HuggingFaceEmbeddings
    
    # 3. Mistral (Le nouveau SDK unifié)
    from mistralai import Mistral
    
    print("✅ Succès : Tous les modules sont chargés avec les versions actuelles.")
    
except ImportError as e:
    print(f"❌ Erreur d'importation : {e}")
    #print("\nAstuce : Assurez-vous d'avoir installé 'langchain-huggingface' et 'mistralai'.")