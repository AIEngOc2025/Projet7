FROM python:3.11-slim

WORKDIR /src/app

# On installe les dépendances d'abord (pour le cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# On copie TOUT le contenu du dossier local dans /app
COPY . .

# Optionnel : Ajouter le répertoire courant au chemin de recherche Python
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]