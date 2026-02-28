from core_rag import RAGSystem

def main():
    print("🚀 Initialisation de la base de données vectorielle...")
    rag = RAGSystem()
    rag.rebuild_database() # Utilise ta méthode qui gère le fetch et l'indexation
    print("✨ Base de données prête pour la soutenance !")

if __name__ == "__main__":
    main()