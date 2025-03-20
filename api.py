from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from io import BytesIO
from PIL import Image
import joblib
from pydantic import BaseModel
from params import *


# Initialiser l'API
app = FastAPI()

# Charger le modèle
DL_MODEL_PATH = DL_MODEL_PATH #<------------------------------------------------

print("Chargement du modèle de deep learning...")
model = load_model(DL_MODEL_PATH)
print("Modèle chargé avec succès.")

# Fonction de prétraitement de l'image
def preprocess_image(image: Image.Image):
    img = image.resize((224, 224))  # Adapter à la taille du modèle
    img_array = img_to_array(img) / 255.0  # Normalisation
    img_array = np.expand_dims(img_array, axis=0)  # Ajouter une dimension batch
    return img_array

# Endpoint pour prédire sur une image envoyée
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Reçoit une image, la prétraite et effectue une prédiction avec le modèle de deep learning.
    """
    try:
        # Charger l'image depuis le fichier uploadé
        image = Image.open(BytesIO(await file.read()))

        # Prétraitement de l'image
        img_array = preprocess_image(image)

        # Faire la prédiction
        res = model.predict(img_array)[0][0]
        # Ajout de logs pour comprendre la sortie du modèle
        print(f"Valeur brute de la prédiction : {res}")  # Debugging

        # Interprétation du résultat
        diagnostic = "Positif" if res >= 0.5 else "Négatif"
        prob = res if res >= 0.5 else 1 - res

        return {
            "diagnostic": diagnostic,
            "probability": f"{prob:.2%}"
        }

    except Exception as e:
        return {"error": str(e)}

# ------------------- Machine Learning Prediction -------------------

# Charger le modèle Machine Learning
ML_MODEL_PATH = ML_MODEL_PATH #<------------------------------------------------
SCALER_PATH = ML_SCALER_PATH #<------------------------------------------------

print("Chargement du modèle de Machine Learning...")
ml_model = joblib.load(ML_MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
print("✅ Modèle de Machine Learning chargé avec succès.")

# Définir un schéma pour les données entrantes
class PredictionInput(BaseModel):
    features: list[float]

# Endpoint pour prédire sur des données tabulaires
@app.post("/predict_ml")
async def predict_ml(data: PredictionInput):
    """
    Reçoit une liste de caractéristiques, les prétraite et effectue une prédiction avec le modèle ML.
    """
    try:
        # Convertir les données en tableau numpy et appliquer le scaling
        input_data = np.array(data.features).reshape(1, -1)
        input_scaled = scaler.transform(input_data)

        # Faire la prédiction
        prediction = ml_model.predict(input_scaled)[0]
        diagnostic = "1= Malin (Cancer) " if prediction == 1 else "0= Bénin (Sans Cancer)"

        return {"diagnostic": diagnostic}

    except Exception as e:
        return {"error": str(e)}
