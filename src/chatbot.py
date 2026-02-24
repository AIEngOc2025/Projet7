import os
from datetime import datetime
from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# --- 1. INITIALISATION ---
embeddings = MistralAIEmbeddings(model="mistral-embed")
llm = ChatMistralAI(model="mistral-small-latest", temperature=0.2)

# Chargement de la BDD
vector_db = FAISS.load_local(
    "data/vdb_paris", 
    embeddings, 
    allow_dangerous_deserialization=True
)

# --- 2. CONFIGURATION DU PROMPT (SYSTEM MESSAGE) ---
template = """Vous êtes un assistant culturel expert de la ville de Paris. 
Votre but est d'aider l'utilisateur à trouver des événements en fonction de sa demande.

CONSIGNES :
1. Dites Bonjour avant toute recommandation et proposez une formule de politesse à la fin.
2. Utilisez UNIQUEMENT les informations relatives au CONTEXTE .
3. Nous sommes aujourd'hui le {current_date}. Ne proposez QUE des événements dont la date est égale ou postérieure à aujourd'hui.
4. Si aucun événement ne correspond dans le contexte, dites-le poliment.
5. Pour chaque recommandation, précisez le Titre, la Date et le Lieu.


CONTEXTE :
{context}

QUESTION DE L'UTILISATEUR :
{question}

RÉPONSE :"""

prompt = ChatPromptTemplate.from_template(template)

# --- 3. FONCTION DE RÉCUPÉRATION AVEC FILTRE TEMPOREL ---
def get_relevant_docs(query):
    # On récupère large pour pouvoir filtrer
    docs = vector_db.similarity_search(query, k=10)
    today = datetime.now().isoformat()
    # Filtre pour ne garder que le futur
    future_docs = [d for d in docs if d.metadata.get('date', '') >= today]
    return "\n\n".join([d.page_content for d in future_docs[:4]])

# --- 4. LA CHAÎNE RAG ---
rag_chain = (
    {"context": get_relevant_docs, "question": RunnablePassthrough(), "current_date": lambda _: datetime.now().strftime('%d/%m/%Y')}
    | prompt
    | llm
    | StrOutputParser()
)

# --- 5. TEST ---
if __name__ == "__main__":
    question = "Quelles expositions d'art contemporain me conseillez-vous pour le mois prochain ?"
    print(f"🤖 Question : {question}\n")
    reponse = rag_chain.invoke(question)
    print(f"✨ Réponse du Chatbot :\n{reponse}")