"""
ECe script vise l'extraction des données d'évènements culturels à Paris depyis l'API
d'opendataSoft sur une période de moins d'un an et à venir.
"""
#%%
import requests
import pandas as pd

# 1. Requête à l'API
#url = "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records"
url = "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records?limit=20&refine=lastdate_end%3A%222026%22&refine=location_countrycode%3A%22FR%22&refine=location_department%3A%22Paris%22"
params = {
    "limit": 50,
    "where": "updatedat >= date'2025-04-17'", # Un an d'historique dynamique
    "refine": [
        "location_countrycode:FR"
        "location_city:Paris",
        "keywords_fr:culture",
        "firstdate_begin:2025-03-22"
        "updatedat:2025-02-23"
    ]
}

response = requests.get(url, 
                        #params=params
                        )
data = response.json()
#%%
print(data.keys())
#%%
# 2. Structuration avec Pandas
# Dans l'API v2.1, les données sont dans la clé 'results'
df = pd.DataFrame(data['results'])


#%%
# 3. Nettoyage ciblé
# On extrait les colonnes essentielles pour le RAG
cols_to_keep = ['uid', 'title_fr', 'description_fr', 'location_name', 'firstdate_begin']
df_clean = df[cols_to_keep].copy()

# Gestion des données manquantes (Point de vigilance)
df_clean['description_fr'] = df_clean['description_fr'].fillna(df_clean['title_fr'])

# 4. Préparation pour la vectorisation
# Conversion de la colonne date en format lisible (ex: "25 octobre 2026")
df_clean['date_lisible'] = pd.to_datetime(df_clean['firstdate_begin']).dt.strftime('%d %B %Y')

# On injecte explicitement la date dans le bloc de texte qui sera vectorisé
df_clean['text_for_rag'] = (
    "Événement: " + df_clean['title_fr'] + 
    " | Date: " + df_clean['date_lisible'] +
    " | Lieu: " + df_clean['location_name'] + 
    " | Description: " + df_clean['description_fr'] +
    " | firstdate_begin: " + df_clean['firstdate_begin']
)

print(f"{len(df_clean)} événements culturels à Paris récupérés.")

nb_futur = len(df_clean[df_clean['firstdate_begin'] >= '2025-02-23'])
print(f"Nombre d'événements à venir : {nb_futur}")


#%% Sauvegarde du  DataFrame nettoyé pour une utilisation ultérieure
df_clean.to_csv("evenements_culturels_paris.csv", index=False)
print("Données nettoyées et sauvegardées pour l'indexation.")        

#%%
data = pd.read_csv("evenements_culturels_paris.csv")
#%%
data.shape
# %%
data.columns

# %%
data.date_lisible.unique()

# %%
