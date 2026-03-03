import os
import re
import requests
from datetime import datetime
from dotenv import load_dotenv

from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()


# Récupère la clé (soit du .env local, soit du secret GitHub)
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not MISTRAL_API_KEY:
    print("⚠️ Attention : MISTRAL_API_KEY est introuvable !")

def clean_html(text):
    """Supprime les balises HTML pour un contexte plus propre."""
    if not text: return ""
    clean = re.sub(r'<[^>]+>', '', text)
    return clean.strip()

class RAGSystem:
    def __init__(self, index_path="data/vdb_paris"):
        self.index_path = index_path
        # Utilisation du modèle d'embedding Mistral pour la cohérence vectorielle
        self.embeddings = MistralAIEmbeddings(model="mistral-embed")
        # Température basse (0.1) pour garantir la fidélité aux documents
        self.llm = ChatMistralAI(model="mistral-small-latest", temperature=0.1)
        self.vector_db = None  # Initialize as None, load only if exists
        if os.path.exists(self.index_path):
            self.vector_db = self._load_db()

    def _load_db(self):
        """Charge l'index FAISS local de manière sécurisée."""
        # Check if the FAISS index file actually exists, not just the directory
        index_file = os.path.join(self.index_path, "index.faiss")
        if os.path.exists(index_file):
            return FAISS.load_local(
                self.index_path, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
        return None

    def rebuild_database(self):
        """Action du endpoint /rebuild : Capture, Nettoyage et Indexation."""
        print("🔄 Reconstruction de la base avec les filtres 2025-2026...")
        
        # Utilisation de l'URL et des paramètres validés en test
        url = "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records"
        params = {
            "limit": 100,
            "where": "lastdate_end >= date'2025-03-01' AND location_city = 'Paris'",
            "order_by": "updatedat DESC"
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json().get('results', [])
        except Exception as e:
            print(f"❌ Erreur lors de la récupération API : {e}")
            return

        raw_docs = []
        for r in data:
            # Construction d'un contenu textuel riche avec labels pour faciliter la recherche
            content = (
                f"ÉVÉNEMENT : {r.get('title_fr')}\n"
                f"LIEU : {r.get('location_name')} ({r.get('location_city')})\n"
                f"DATE : {r.get('firstdate_begin')}\n"
                f"DESCRIPTION : {clean_html(r.get('description_fr'))}"
            )
            raw_docs.append(Document(
                page_content=content,
                metadata={
                    "date": r.get('firstdate_begin'), 
                    "titre": r.get('title_fr')
                }
            ))

        # Chunking optimisé pour conserver l'unité d'un événement
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        docs = splitter.split_documents(raw_docs)

        # Création et sauvegarde locale
        self.vector_db = FAISS.from_documents(docs, self.embeddings)
        self.vector_db.save_local(self.index_path)
        print(f"✅ Base de données vectorielle prête : {len(docs)} chunks indexés.")

    def _get_relevant_docs(self, query):
        """Récupération sémantique des documents les plus proches."""
        if not self.vector_db:
            return ""
        
        # On récupère les 5 meilleurs résultats pour laisser un choix au LLM
        docs = self.vector_db.similarity_search(query, k=5)
        
        # On passe le texte brut au prompt sans filtre de date Python 
        # pour laisser l'IA gérer la logique temporelle.
        return "\n\n".join([d.page_content for d in docs])

    def ask(self, question: str):
        """Action du endpoint /ask avec Prompt System final."""
        if not self.vector_db:
            return "La base de données est vide. Veuillez appeler /rebuild d'abord."

        template = """Tu es l'Assistant Officiel des Événements de Paris.\
              Ton rôle est d'aider les utilisateurs à trouver des activités culturelles en utilisant EXCLUSIVEMENT les informations \
                fournies dans le contexte ci-dessous.

### RÈGLES DE CONDUITE :
1. **Priorité au Contexte** : Si l'information n'est pas dans le contexte ou si aucun événement ne correspond ou ne se trouve à moins d'un au dans le passé par rapport à d'aujourd'hui\
    ,réponds que tu ne trouves pas d'information correspondante.
2. **Format de Réponse** : 
   - **Titre de l'événement** (en gras) 
   - 📍 Lieu : [Nom du lieu] 
   - 📅 Date : [Date formatée] 
   - 📝 Résumé : [2-3 phrases max sur l'intérêt de l'événement]
   - revenir à la ligne 
3. **Intelligence Temporelle** : Utilise la "DATE ACTUELLE" pour évaluer si un événement est passé ou futur.
   Ne pas aller au delà d'un an en arrière.
4. **Ton** : Professionnel, clair et accueillant.

### CONSIGNES :
 1. Les évènements doivent dater de moins d'un AN à partir d'aujourd'hui et/ ou  se poursuivre dans le futur. 

### CONTEXTE :
{context}

### DATE ACTUELLE :
{current_date}

QUESTION DE L'UTILISATEUR : {question}

RÉPONSE :"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # Chaîne de traitement LangChain (LCEL)
        chain = (
            {
                "context": lambda x: self._get_relevant_docs(x), 
                "question": RunnablePassthrough(),
                "current_date": lambda _: datetime.now().strftime('%d/%m/%Y')
            }
            | prompt | self.llm | StrOutputParser()
        )
        return chain.invoke(question)