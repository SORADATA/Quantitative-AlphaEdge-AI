from brvmfinance.tickers import Tickers

# On définit les tickers (BOA Bénin, Mali, Sénégal, Côte d'Ivoire)
mes_actions = Tickers("BOAN BOAM BOAS BOAC")

# Récupération de l'historique
df_global = mes_actions.history(length=7) # On prend un petit historique

# Pour avoir "hier" (la dernière ligne de l'index temporel)
derniere_seance = df_global.index.max()
df_hier = df_global.loc[derniere_seance]

print(f"📅 Données de la séance du : {derniere_seance}")
print(df_hier)