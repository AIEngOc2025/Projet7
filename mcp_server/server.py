import os
import sys
import uvicorn
from fastapi import FastAPI
from fastapi_mcp import FastApiMCP

# Correction du chemin pour importer ton module 'src'
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_path not in sys.path:
    sys.path.append(root_path)

from src.core_rag import RAGSystem

app = FastAPI(title="Paris 2026 API")
rag = RAGSystem()

# Ton endpoint qui devient un outil MCP grâce à l'operation_id
@app.get("/rechercher", operation_id="rechercher_evenements_paris")
async def rechercher(query: str):
    """Recherche des événements culturels et tech à Paris en 2026."""
    return {"resultat": rag.ask(query)}
#--

#--

# Configuration du serveur MCP selon la signature de la classe FastApiMCP
mcp = FastApiMCP(
    app,
    name="Paris 2026 MCP",
    description="Serveur d'outils pour l'agenda de Paris 2026",
)

# Montage automatique des routes MCP
mcp.mount()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)