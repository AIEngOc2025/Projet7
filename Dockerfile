# 1. Image de base officielle (légère et stable)
FROM python:3.11-slim

# 2. Dossier de travail dans le conteneur
WORKDIR /app

# 3. Installation des outils système (nécessaires pour FAISS et les bibliothèques C++)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Gestion des dépendances (optimise le cache de build)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copie de l'intégralité du projet
COPY . .

# 6. Configuration des chemins et variables d'environnement
# Permet de trouver les modules dans /app et /app/src simultanément
ENV PYTHONPATH="/app:/app/src"
# Force l'affichage des logs en temps réel (crucial pour la démo)
ENV PYTHONUNBUFFERED=1

# 7. Création du dossier pour la base de données vectorielle
RUN mkdir -p data/vdb_paris

# 8. Port exposé par FastAPI
EXPOSE 8000

# 9. Commande de lancement (flexible pour src.main ou main)
# On lance Uvicorn en pointant sur l'objet 'app' du module 'main'
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]