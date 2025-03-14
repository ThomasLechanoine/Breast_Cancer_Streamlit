# prep
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder

def load_data(data_path):
    """
    Charge les données depuis un fichier CSV.
    """
    return pd.read_csv(data_path)

def preprocess_data(data):
    """
    Effectue le prétraitement des données :
    - Supprime les colonnes inutiles
    - Encode la variable cible avec LabelEncoder
    - Sépare les features et la cible
    - Applique un scaling avec RobustScaler
    - Divise en train et test
    """
    # Supprimer les colonnes non pertinentes
    X = data.drop(columns=['id', 'diagnosis'])

    # Encodage de la variable cible (Bénin → 0, Malin → 1)
    le = LabelEncoder()
    y = le.fit_transform(data['diagnosis'])

    # Séparation en jeu de test et d'entraînement
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

    # Scaling des données
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, le
