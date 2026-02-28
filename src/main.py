"""
Ce module contient l'implémentation de l'API REST pour interagir avec le système RAG.
Il utilise FastAPI pour créer les endpoints et gère les requêtes de l'utilisateur.
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from contextlib import asynccontextmanager
from pydantic import BaseModel
import src.core_rag
import uvicorn
import asyncio

# Lazy loading du RAGSystem
rag = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialiser RAGSystem en arrière-plan (non-bloquant)
    print("- Serveur démarré ...")
    asyncio.create_task(init_rag_async())
    yield
    # Shutdown
    print("- Arrêt du serveur ...")

async def init_rag_async():
    """Initialise le RAGSystem en arrière-plan sans bloquer le serveur"""
    global rag
    try:
        print("- Initialisation du RAGSystem en arrière-plan...")
        rag = src.core_rag.RAGSystem()
        print("- RAGSystem initialisé")
    except Exception as e:
        print(f"❌ Erreur lors de l'initialisation: {e}")

def get_rag():
    """Retourne le RAGSystem (attend qu'il soit initialisé si nécessaire)"""
    global rag
    if rag is None:
        print("⚠️ RAGSystem pas encore initialisé, initialisation synchrone...")
        rag = src.core_rag.RAGSystem()
    return rag

# Initialisation de l'API
app = FastAPI(
    title="API Culture Paris - Système RAG",
    description="Interface REST pour interroger les événements parisiens via Mistral AI",
    version="1.1.0",
    lifespan=lifespan
)

# Modèle pour la requête /ask
class Query(BaseModel):
    question: str

# --- ROUTES ---

@app.get("/")
def home():
    return {"message": "API RAG opérationnelle. Rendez-vous sur /docs pour tester."}

@app.post("/ask")
async def ask_question(item: Query):
    """
    Endpoint principal pour poser une question.
    Prend un JSON : {"question": "votre question"}
    """
    if not item.question.strip() or item.question == "???":
        raise HTTPException(status_code=400, detail="La question est invalide ou trop courte.")
    
    try:
        rag_instance = get_rag()
        response = rag_instance.ask(item.question)
        return {"question": item.question, "answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne : {str(e)}")

@app.post("/rebuild")
async def rebuild_vdb(background_tasks: BackgroundTasks):
    """
    Reconstruit la base de données vectorielle à partir de l'API OpenData.
    Utilise BackgroundTasks pour ne pas bloquer l'utilisateur pendant l'indexation.
    """
    try:
        rag_instance = get_rag()
        background_tasks.add_task(rag_instance.rebuild_database)
        return {"message": "Reconstruction de l'index lancée avec succès en arrière-plan."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
