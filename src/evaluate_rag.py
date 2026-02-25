from asyncio import sleep
from datetime import datetime
import pandas as pd
import chatbot # Import de ton module chatbot pour accéder à rag_chain et get_relevant_docs
# On importe ta chaine depuis ton fichier (supposons qu'il s'appelle main_rag.py)
# from main_rag import rag_chain, get_relevant_docs 

def evaluate_system():
    # 1. Définition des scénarios de test
    test_cases = [
        {
            "nom": "Recherche Standard",
            "question": "Quels évènements culturels à Paris cette année?",
            "attente": "Doit trouver des événements futurs uniquement."
        },
        {
            "nom": "Hors Contexte (Hallucination)",
            "question": "Comment faire une pizza Margherita ?",
            "attente": "Doit répondre poliment qu'il ne peut pas aider (car pas dans le contexte)."
        },
        {
            "nom": "Filtre Temporel",
            "question": "Donne moi des événements de l'année 2023.",
            "attente": "Doit dire qu'aucun événement n'est disponible (car 2023 est passé)."
        }
        
    ]

test_extremes = [
    {
        "nom": "Requête Vide/Bruit",
        "question": "?...",
        "attente": "Réponse polie d'incapacité."
    },
    {
        "nom": "Spécificité Thématique",
        "question": "Quels événements parlent de 'robot' ou d'intelligence artificielle ?",
        "attente": "Doit trouver le concours 'humanoïdes' s'il est en base."
    },
    {
        "nom": "Hors Zone",
        "question": "Donne moi le programme du festival d'Avignon.",
        "attente": "Doit dire qu'il est limité à Paris."
    }
]

for case in test_extremes:
    print(f"🛠️ Test : {case['nom']}")
    reponse = chatbot.rag_chain.invoke(case['question'])
    print(f"🤖 Réponse : {reponse}\n")

    results = []

 

if __name__ == "__main__":
    evaluate_system()