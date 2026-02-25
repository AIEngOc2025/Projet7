from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from src import core_rag
from src.core_rag import RAGSystem # On importe ta classe
import uvicorn



# Initialisation de l'API
app = FastAPI(
    title="API Culture Paris - Système RAG",
    description="Interface REST pour interroger les événements parisiens via Mistral AI",
    version="1.0.0"
)

# On instancie ton système RAG une seule fois au démarrage
rag = core_rag.RAGSystem()

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
    # Protection contre les questions vides (vu dans nos tests précédents !)
    if not item.question.strip() or item.question == "???":
        raise HTTPException(status_code=400, detail="La question est invalide ou trop courte.")
    
    try:
        response = rag.ask(item.question)
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
        # On lance la reconstruction en tâche de fond
        background_tasks.add_task(rag.rebuild_database)
        return {"message": "Reconstruction de l'index lancée avec succès en arrière-plan."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)