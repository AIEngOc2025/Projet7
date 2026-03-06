import os
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics.collections import Faithfulness, AnswerRelevancy
from ragas.run_config import RunConfig

# Imports standardisés selon la philosophie LangChain
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
import src
from src.core_rag import RAGSystem

load_dotenv()

# Configuration de l'observabilité recommandée par la documentation
os.environ["LANGCHAIN_TRACING_V2"] = "true"

def run_langchain_eval():
    print("🚀 Initialisation de l'évaluation via l'interface standard LangChain...")
    
    # Initialisation de votre système
    rag = RAGSystem()
    
    # 1. Configuration des Modèles (Interface standardisée)
    # La doc précise que LangChain permet de swapper les providers sans changer le code.
    eval_llm = ChatMistralAI(model="mistral-small-latest", temperature=0)
    eval_embeddings = MistralAIEmbeddings(model="mistral-embed")

    # 2. Création du dataset (Exemple de questions)
    queries = [
        "Quels événements tech sont prévus à Paris en 2026 ?",
        "Quels évènements à la Cité des Sciences ?"
    ]
    
    data = {"question": [], "answer": [], "contexts": []}
    
    for q in queries:
        # LangChain facilite l'extraction de contexte et la génération en < 10 lignes
        docs = rag.vector_db.similarity_search(q, k=3)
        res = rag.ask(q)
        
        data["question"].append(q)
        data["answer"].append(res)
        data["contexts"].append([d.page_content for d in docs])

    dataset = Dataset.from_dict(data)

    # 3. Évaluation avec RunConfig pour une exécution durable
    # LangChain souligne que ses agents (et donc les évaluateurs) 
    # profitent de la persistance et du streaming.
    config = RunConfig(timeout=120, max_workers=1)

    result = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(llm=eval_llm),
            AnswerRelevancy(llm=eval_llm, embeddings=eval_embeddings)
        ],
        run_config=config
    )

    print("\n📊 Résultats conformes aux standards LangChain :")
    print(result.to_pandas())

if __name__ == "__main__":
    run_langchain_eval()