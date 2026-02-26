import requests

def test_api():
    url = "http://127.0.0.1:8000/ask"
    payload = {"question": "Quels sont les événements qui se déroulent à la 'Cité des sciences' cette année?"}
    
    print("📡 Envoi d'une requête au RAG...")
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        print("✅ Réponse reçue :")
        print(response.json()["answer"])
    else:
        print(f"❌ Erreur {response.status_code}: {response.text}")

if __name__ == "__main__":
    test_api()