import os
import requests
from datetime import datetime
from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()

class RAGSystem:
    def __init__(self, index_path="data/vdb_paris"):
        self.index_path = index_path
        self.embeddings = MistralAIEmbeddings(model="mistral-embed")
        self.llm = ChatMistralAI(model="mistral-small-latest", temperature=0.2)
        self.vector_db = self._load_db()

    def _load_db(self):
        """Charge l'index FAISS s'il existe."""
        if os.path.exists(self.index_path):
            return FAISS.load_local(
                self.index_path, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
        return None

    def rebuild_database(self):
        """Action du endpoint /rebuild : Fetch API -> Chunking -> Indexation."""
        print("🔄 Début de la reconstruction de la base...")
        
        # 1. Récupération (Exemple simplifié, adapte avec ton URL OpenData)
        url = "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records"
        params = {"limit": 50, "refine": ["location_city:Paris"]}
        response = requests.get(url, params=params)
        data = response.json().get('results', [])

        # 2. Conversion en Documents LangChain
        raw_docs = [
            Document(
                page_content=f"Titre: {r.get('title_fr')}\nDescription: {r.get('description_fr')}\nLieu: {r.get('location_name')}",
                metadata={"date": r.get('firstdate_begin'), "titre": r.get('title_fr')}
            ) for r in data
        ]

        # 3. Chunking
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
        docs = splitter.split_documents(raw_docs)

        # 4. Création et Sauvegarde de la VDB
        self.vector_db = FAISS.from_documents(docs, self.embeddings)
        self.vector_db.save_local(self.index_path)
        print("✅ Base de données vectorielle reconstruite et sauvegardée.")

    def _get_relevant_docs(self, query):
        if not self.vector_db:
            return ""
        docs = self.vector_db.similarity_search(query, k=10)
        today = datetime.now().isoformat()
        future_docs = [d for d in docs if d.metadata.get('date', '') >= today]
        return "\n\n".join([d.page_content for d in future_docs[:4]])

    def ask(self, question: str):
        """Action du endpoint /ask."""
        if not self.vector_db:
            return "La base de données est vide. Veuillez appeler /rebuild d'abord."

        template = """Vous êtes un assistant culturel expert de Paris. 
        Utilisez UNIQUEMENT le CONTEXTE suivant pour répondre. 
        Si ce n'est pas dans le contexte, dites-le poliment.
        
        CONTEXTE : {context}
        QUESTION : {question}
        DATE DU JOUR : {current_date}
        
        RÉPONSE :"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        chain = (
            {
                "context": lambda x: self._get_relevant_docs(x), 
                "question": RunnablePassthrough(),
                "current_date": lambda _: datetime.now().strftime('%d/%m/%Y')
            }
            | prompt | self.llm | StrOutputParser()
        )
        return chain.invoke(question)