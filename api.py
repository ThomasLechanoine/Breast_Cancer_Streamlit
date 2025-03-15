import requests
import pandas as pd
import matplotlib.pyplot as plt

url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
params = {
    'vs_currency': 'usd',
    'days': '30',  # Nombre de jours de données historiques
    'interval': 'daily'  # Intervalle des données (daily, weekly, etc.)
}

response = requests.get(url, params=params)
data = response.json()

# Convertir en DataFrame pour une manipulation facile et rapide
df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
