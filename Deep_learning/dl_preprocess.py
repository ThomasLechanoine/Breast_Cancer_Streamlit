# prep

#////////////////////////////////
import zipfile
import os

#////////////////////////////////

def download():
    """
    Extrait un fichier ZIP contenant les données d'entraînement si nécessaire.
    """
    zip_path = "Data/Data_prepros.zip"  # Modifier avec le bon chemin
    extract_path = "Data/Data_Deep_Learning"

    # Vérifier si les données existent déjà
    if not os.path.exists(extract_path):
        print(f"Extraction des données depuis {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print("✅ Extraction terminée !")
    else:
        print("✅ Les données existent déjà, extraction non nécessaire.")

#////////////////////////////////////////////////////////////////

import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def preprocess_image(image_path):
    """
    Charge et prétraite une image pour le modèle de deep learning.
    """
    img = load_img(image_path, target_size=(224, 224))  # Taille adaptée au modèle
    img_array = img_to_array(img) / 255.0  # Normalisation
    img_array = np.expand_dims(img_array, axis=0)  # Ajouter une dimension batch
    return img_array


#////////////////////////////////////////////////////////////////
