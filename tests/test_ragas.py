import os
import sys
import pandas as pd
from dotenv import load_dotenv
from datasets import Dataset

# Fix du chemin
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy
from ragas.run_config import RunConfig # Pour gérer les timeouts
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from src.core_rag import RAGSystem

load_dotenv()

def run_eval():
    print("🚀 Initialisation du RAGSystem...")
    rag = RAGSystem()
    
    # 1. Modèles (Augmentation du timeout pour éviter les 'nan')
    eval_llm = ChatMistralAI(model="mistral-small-latest", temperature=0, timeout=120)
    eval_embeddings = MistralAIEmbeddings(model="mistral-embed")

    # 2. Préparation d'un mini-batch de test (plus représentatif qu'une seule question)
    test_queries = [
        "Quels événements tech sont prévus à Paris en 2026 ?",
        "Y a-t-il des concerts gratuits à Paris en 2026 ?",
        "Quels sont les événements culturels majeurs à Paris en 2026 ?"
        "Quels évènements à la Cité des Sciences?",
        "Quels concerts de jazz à Paris en 2026 ?"

    ]
    
    data = {"question": [], "answer": [], "contexts": []}
    
    for query in test_queries:
        docs = rag.vector_db.similarity_search(query, k=3)
        response = rag.ask(query)
        data["question"].append(query)
        data["answer"].append(response)
        data["contexts"].append([d.page_content for d in docs])

    dataset = Dataset.from_dict(data)

    print("📊 Calcul des scores ...")
    
    # Configuration pour éviter les échecs réseau et les 'nan'
    # On force un seul worker (max_workers=1) pour ne pas griller ton quota API
    run_config = RunConfig(timeout=180, max_workers=1)

    result = evaluate(
        dataset=dataset,
        metrics=[Faithfulness(), AnswerRelevancy()],
        llm=eval_llm,
        embeddings=eval_embeddings,
        run_config=run_config
    )

    print("\n🎯 RÉSULTATS FINAUX :")
    df_results = result.to_pandas()
    print(df_results)
    
    # Sauvegarde propre pour ton rapport de projet
    df_results.to_csv("rapport_evaluation_complet.csv", index=False)
    print("\n✅ Rapport sauvegardé : rapport_evaluation_complet.csv")

if __name__ == "__main__":
    run_eval()