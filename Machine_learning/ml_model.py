import os
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from Machine_learning.ml_preprocess import load_data, preprocess_data

# Détecter automatiquement le bon chemin du fichier CSV
DATA_PATH = "/home/bren/code/ThomasLechanoine/Breast_Cancer_Project/Data/Machine_learning.csv"


def create_model():
    """
    Initialise le modèle de régression logistique.
    """
    return LogisticRegression()

def tune_hyperparameters(X_train, y_train):
    """
    Effectue une recherche d'hyperparamètres avec GridSearchCV.
    """
    model = LogisticRegression()

    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'saga'],
        'penalty': ['l1', 'l2'],
        'max_iter': [10000, 20000]
    }

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    print("Best parameters:", grid_search.best_params_)
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test):
    """
    Évalue le modèle sur l'ensemble de test.
    """
    y_pred = model.predict(X_test)

    print("Model Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    # Charger et prétraiter les données
    data = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test, scaler, le = preprocess_data(data)

    # Entraîner le modèle et rechercher les meilleurs hyperparamètres
    best_model = tune_hyperparameters(X_train, y_train)

    # Sauvegarder le modèle et le scaler
    os.makedirs("models_saved", exist_ok=True)  # ✅ Crée le dossier si nécessaire
    joblib.dump(best_model, "models_saved/ml_best_model.pkl")
    joblib.dump(scaler, "models_saved/ml_scaler.pkl")

    print("✅ Modèle Logistic Regression sauvegardé avec succès !")
