import os
from huggingface_hub import HfApi

# On récupère le token
token = os.getenv("HF_TOKEN")

if not token:
    print(" Erreur : La variable d'environnement HF_TOKEN est vide !")
else:
    api = HfApi()
    try:
        user_info = api.whoami(token=token)
        print(f"✅ Connecté avec succès en tant que : {user_info['name']}")
    except Exception as e:
        print(f" Erreur d'authentification : {e}")