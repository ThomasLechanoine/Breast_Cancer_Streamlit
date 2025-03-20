import streamlit as st
import requests
from PIL import Image
import io
import tensorflow as tf
import joblib
import sys
import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from params import *  # Importation des URLs API
import tensorflow as tf
from Deep_learning.dl_model import RandomFixedDense
# Machine Learning Imports
from Machine_learning.ml_preprocess import load_data, preprocess_data
from Machine_learning.ml_model import create_model, tune_hyperparameters, evaluate_model
#----------------------------------------------------------------------------------------------------------

# Add Machine_learning/ to sys.path for module access
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "Machine_learning")))

# ---------------------- CONFIGURATION ----------------------
st.set_page_config(page_title="Application de Détection de Cancer du sein", layout="wide")




# Image à afficher à gauche dans la sidebar
image_path_left = os.path.join("app_img", "2.jpg") #<------------------------------------------------
image = Image.open(image_path_left)

# Afficher l'image sur la barre latérale
st.sidebar.image(image_path_left, use_container_width=True)

# ---------------------- CUSTOM PAGE NAVIGATION ----------------------
st.sidebar.markdown(
    "<div style='text-align: center; font-size: 22px; font-weight: bold;'>Navigation</div>",
    unsafe_allow_html=True
)

# Maintain session state for page navigation
if "page" not in st.session_state:
    st.session_state.page = "Classification Tumeurs (ML)"

# if st.sidebar.button("Graphiques"):
#     st.session_state.page = "Graphiques"
if st.sidebar.button("🔬 Classification Tumeurs \n (Machine Learning)", use_container_width=True):
    st.session_state.page = "Classification Tumeurs (ML)"
if st.sidebar.button("📸 Prédiction Mammographie \n (Deep Learning)", use_container_width=True):
    st.session_state.page = "Prédiction Mammographie (DL)"


page = st.session_state.page

# Image à afficher à gauche dans la sidebar
image_path_left = os.path.join("app_img", "01.png") #<------------------------------------------------
image = Image.open(image_path_left)

# Afficher l'image sur la barre latérale
st.sidebar.image(image_path_left, use_container_width=True)
#------------------------------------------------------------------------------

# Load and display the cover image
# image_path = os.path.join("/home", "bren", "code", "ThomasLechanoine", "Breast_Cancer_Project", "app_img", "01.png")
# st.image(image_path, use_container_width=True)

# # ---------------------- GRAPHICS PAGE ----------------------
# if page == "Graphiques":
#     st.title("Visualisation des Graphiques")
#     st.write("Analyse des données avec des visualisations graphiques.")

#     # Définition du répertoire contenant les graphiques
#     graph_dir = os.path.join("/home", "bren", "code", "ThomasLechanoine", "Breast_Cancer_Project", "app_img")

#     # Liste des nouveaux graphiques avec descriptions
#     graph_data = [
#         {"file": "distribution_age_kde_true.jpg", "title": "Graphique 1", "description": "🔬 Distribution des âges avec courbe KDE."},
#         {"file": "graphique2.jpg", "title": "Graphique 2", "description": "📊 Analyse exploratoire des données."},
#         {"file": "graphique.jpg", "title": "Graphique 3", "description": "📈 Un autre graphique pertinent pour l'analyse."}
#     ]

#     # Affichage des images avec menu déroulant pour description
#     for graph in graph_data:
#         img_path = os.path.join(graph_dir, graph["file"])

#         with st.expander(f"📊 {graph['title']}"):
#             st.image(img_path, use_container_width=True)
#             st.write(graph["description"])

# Ajout de style CSS pour rendre le contour du menu déroulant plus visible
#-------------------------------------------------------------------------------------------------------

st.markdown("""
    <style>
        /* Style pour rendre le contour du menu déroulant plus visible */
        div[data-testid="stExpander"] {
            border: 2px solid #4A90E2 !important; /* Bleu vif */
            border-radius: 10px !important;
            background-color: #E3F2FD !important; /* Bleu pastel */
            padding: 10px !important;
        }

        /* Style du titre dans l'expander */
        div[data-testid="stExpander"] summary {
            font-weight: bold !important;
            font-size: 16px !important;
            color: #1A1A1A !important;
        }
    </style>
""", unsafe_allow_html=True)
#-------------------------------------------------------------------------------------------------------

# ---------------------- LOAD MODELS DL---------------------
@st.cache_resource
def load_dl_model():
    return tf.keras.models.load_model(DL_MODEL_PATH, custom_objects={"RandomFixedDense": RandomFixedDense})
 #//////////

model = load_dl_model()

# ---------------------- LOAD MODELS ML---------------------
# Load the trained model and scaler
@st.cache_resource
def load_model():
    MODEL_PATH = ML_MODEL_PATH
    SCALER_PATH = ML_SCALER_PATH
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    else:
        st.error("❌ Modèle ML non trouvé. Entraînez-le avec `python main.py --train ml`.")
        return None, None

model, scaler = load_model()

# ---------------------- LOAD test data--------------------- a voir si besoin
@st.cache_resource
def load_test_data():
    dataset_path = ML_DATA_PATH #<------------------------------------------------
    data = pd.read_csv(dataset_path)
    X = data.drop(columns=["id", "diagnosis"])  # Drop unnecessary columns
    y = data["diagnosis"].map({"B": 0, "M": 1})  # Encode labels (B:0, M:1)

    # Split into train and test (must match how the model was trained)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    return X_test, y_test

# Load test dataset
X_test, y_test = load_test_data()

# ////////////////////////Page de prédiction via Machine Learning/////////////////////////
#----------------------------------------------
st.markdown("""
    <style>
        /* Fond général en bleu pastel */
        .stApp {
            background-color: #E3F2FD !important;
        }

        /* Modification du header (bande supérieure) */
        header, .st-emotion-cache-18ni7ap {
            background-color: #E3F2FD !important;
        }

        /* Titres en couleur foncée pour contraste */
        h1, h2, h3, h4, h5, h6, p, label {
            color: #1A1A1A !important;
            font-weight: bold !important;
        }

        /* Bouton principal (st.button et st.form_submit_button) */
        div.stButton > button, div[data-testid="stFormSubmitButton"] > button {
            background-color: #FFA69E !important; /* Rouge saumon pastel */
            color: #FFFFFF !important; /* Texte blanc */
            border-radius: 12px !important;
            font-size: 18px !important;
            font-weight: bold !important;
            padding: 12px 24px !important;
            border: none !important; /* Suppression des bordures */
            box-shadow: none !important; /* Suppression de l'ombre */
            transition: all 0.3s ease-in-out !important;
        }

        /* Effet hover sur les boutons */
        div.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {
            background-color: #FF6B6B !important; /* Rouge plus foncé */
            box-shadow: none !important; /* Suppression de l’ombre */
            transform: scale(1.05) !important; /* Effet léger d'agrandissement */
        }

        /* Style des inputs */
        .stNumberInput>div>div {
            width: 140px !important;
            border-radius: 8px !important;
            border: 2px solid #A1C4FD !important; /* Bleu pastel */
            background-color: #FFFFFF !important;
            color: #1A1A1A !important;
            padding: 5px !important;
        }

        /* Texte des inputs */
        div[data-testid="stNumberInput"] input {
            font-size: 14px !important;
            padding: 10px !important;
            text-align: center !important;
            background-color: #FFFFFF !important;
            color: #1A1A1A !important;
            border: none !important;
        }

        /* Style du sidebar */
        .stSidebar {
            background-color: #B2D3FF !important;
        }

        /* Style des résultats de prédiction */
        .stSuccess {
            border-radius: 10px !important;
            padding: 10px !important;
            text-align: center !important;
            font-weight: bold !important;
            color: white !important;
        }
    </style>
""", unsafe_allow_html=True)


# ------------------- Machine Learning Prediction -------------------
# URL de l'API pour la prédiction ML
ML_API_URL = ML_API_URL  # !!! Remplacer cette URL si l'API est hébergée en ligne #<------------------------------------------------


# ------------------- Machine Learning Prediction -------------------
if page == "Classification Tumeurs (ML)":
    st.title("Classification de tumeurs via Machine Learning")

    #  Valeurs par défaut (corrigées)
    default_values_1 = {
        "radius_mean": 17.99, "texture_mean": 10.38, "perimeter_mean": 122.8, "area_mean": 1001.0,
        "smoothness_mean": 0.1184, "compactness_mean": 0.2776, "concavity_mean": 0.3001, "concave points_mean": 0.1471,
        "symmetry_mean": 0.2419, "fractal_dimension_mean": 0.07871, "radius_se": 1.095, "texture_se": 0.9053,
        "perimeter_se": 8.589, "area_se": 153.4, "smoothness_se": 0.006399, "compactness_se": 0.04904,
        "concavity_se": 0.05373, "concave points_se": 0.01587, "symmetry_se": 0.03003, "fractal_dimension_se": 0.006193,
        "radius_worst": 25.38, "texture_worst": 17.33, "perimeter_worst": 184.6, "area_worst": 2019.0,
        "smoothness_worst": 0.1622, "compactness_worst": 0.6656, "concavity_worst": 0.7119, "concave points_worst": 0.2654,
        "symmetry_worst": 0.4601, "fractal_dimension_worst": 0.1189
    }

    default_values_2 = {
        "radius_mean": 13.64, "texture_mean": 16.34, "perimeter_mean": 87.21, "area_mean": 571.8,
        "smoothness_mean": 0.07685, "compactness_mean": 0.06059, "concavity_mean": 0.01857, "concave points_mean": 0.01723,
        "symmetry_mean": 0.1353, "fractal_dimension_mean": 0.05953, "radius_se": 0.1872, "texture_se": 0.9234,
        "perimeter_se": 1.449, "area_se": 14.55, "smoothness_se": 0.004477, "compactness_se": 0.01177,
        "concavity_se": 0.01079, "concave points_se": 0.007956, "symmetry_se": 0.01325, "fractal_dimension_se": 0.002551,
        "radius_worst": 14.67, "texture_worst": 23.19, "perimeter_worst": 96.08, "area_worst": 656.7,
        "smoothness_worst": 0.1089, "compactness_worst": 0.1582, "concavity_worst": 0.105, "concave points_worst": 0.08586,
        "symmetry_worst": 0.2346, "fractal_dimension_worst": 0.08025
    }
    #scénario
    st.write("Scénario 1 : Un médecin reçoit des données issues de l'imagerie médicale → analyse avec le Machine Learning.")
    # Ajout du sous-titre et explication du Machine Learning
    st.subheader("Qu'est-ce que le Machine Learning ?")

    with st.expander("Le Machine Learning en quelques mots"):
        st.write("""

        🧠 Vous vous rappelez de l'exemple de l'enfant qui apprend a reconnaitre les chocolatines grace à des informations décrivant une chocolatine et du parallèle avec le Machine Learning ?

        📸 Application à notre cas :
        Nous avons donné au modèle **des informations de description de tumeur**, pour qu'il apprenne à **reconnaître si la tumeur est maligne ou benigne**.
        """)

    # 🔍 **Comme expliqué dans l'introduction, le Machine Learning (ML)** est une branche de l'intelligence artificielle.
    # Ajout du sous-titre et explication du Machine Learning
    st.subheader("Notre défi?")

    with st.expander("Analyser les tumeurs"):
        st.write("""
        🔍 Notre défi était de **pouvoir Développer un modèle capable de classifier les tumeurs bénignes et malignes à partir de caractéristiques extraites d’images médicales.**.

        🎯 Pour cela, nous avons proposé un Diagnostic Assisté par Machine Learning.
        """)

    # Ajout d'un deuxième sous-titre avant l'input des caractéristiques tumorales
    st.subheader("Outil de prédiction")
    # st.write("Veuillez entrer les mesures de la tumeur pour obtenir une prédiction.")
    st.write("Après avoir entré les paramètres, notre modèle effectue l’analyse et fournit une prédiction.")

 # ------------------- PRÉDICTION 1 -------------------
    st.subheader("🔬 Prédiction 1")
    with st.form(key="prediction_form_1"):
        columns = st.columns(5)
        feature_values_1 = {}

        for i, feature in enumerate(default_values_1.keys()):
            with columns[i % 5]:
                feature_values_1[feature] = st.number_input(
                    feature, min_value=0.0, format="%.4f", value=default_values_1[feature]
                )

        submit_button_1 = st.form_submit_button(label="Lancer la Prédiction 1")


    if submit_button_1:
        input_data_1 = pd.DataFrame([list(feature_values_1.values())], columns=default_values_1.keys())

        if input_data_1.isnull().values.any():
            st.error("⚠️ Certaines valeurs sont vides ou incorrectes ! Veuillez remplir tous les champs.")
        else:
            input_data_json = {"features": input_data_1.values.tolist()[0]}
            response = requests.post(ML_API_URL, json=input_data_json)

            if response.status_code == 200:
                prediction_1 = response.json()["diagnostic"]
            else:
                prediction_1 = "Erreur lors de la prédiction."

            # **Mise en forme du résultat**
            if "Malin" in prediction_1:
                diagnostic_1 = "🔴 Tumeur Maligne"
                color_1 = "#F76C6C"  # Rouge pastel
            else:
                diagnostic_1 = "🔵 Tumeur Bénigne"
                color_1 = "#A1C4FD"  # Bleu pastel

            st.markdown(
                f'<div style="background-color:{color_1}; padding:15px; border-radius:10px; text-align:center; '
                f'font-size:16px; color:white; font-weight:bold;">'
                f'Résultat de la prédiction 1 : {diagnostic_1}'
                '</div>',
                unsafe_allow_html=True
            )
    # ------------------- PRÉDICTION 2 -------------------
    st.subheader("🔬 Prédiction 2")
    with st.form(key="prediction_form_2"):
        columns = st.columns(5)
        feature_values_2 = {}

        for i, feature in enumerate(default_values_2.keys()):
            with columns[i % 5]:
                feature_values_2[feature] = st.number_input(
                    feature, min_value=0.0, format="%.4f", value=default_values_2[feature]
                )

        submit_button_2 = st.form_submit_button(label="Lancer la Prédiction 2")

    if submit_button_2:
        input_data_2 = pd.DataFrame([list(feature_values_2.values())], columns=default_values_2.keys())

        if input_data_2.isnull().values.any():
            st.error("⚠️ Certaines valeurs sont vides ou incorrectes ! Veuillez remplir tous les champs.")
        else:
            input_data_json = {"features": input_data_2.values.tolist()[0]}
            response = requests.post(ML_API_URL, json=input_data_json)

            if response.status_code == 200:
                prediction_2 = response.json()["diagnostic"]
            else:
                prediction_2 = "Erreur lors de la prédiction."

            #  **Mise en forme du résultat**
            if "Malin" in prediction_2:
                diagnostic_2 = "🔴 Malin (Cancer)"
                color_2 = "#F76C6C"  # Rouge pastel
            else:
                diagnostic_2 = "🔵 Bénin"
                color_2 = "#A1C4FD"  # Bleu pastel

            st.markdown(
                f'<div style="background-color:{color_2}; padding:15px; border-radius:10px; text-align:center; '
                f'font-size:16px; color:white; font-weight:bold;">'
                f'Résultat de la prédiction 2 : {diagnostic_2}'
                '</div>',
                unsafe_allow_html=True
            )

#--------------------CONFUSION MATRIX------------------
# ------------------- AFFICHAGE DE LA MATRICE DE CONFUSION SUR LA PAGE ML -------------------
if page == "Classification Tumeurs (ML)":
    st.subheader("📊 Performance de notre Modèle de Machine Learning")

    # 📌 Déterminer dynamiquement le dossier contenant `app.py`
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # 📌 Construire un chemin relatif et généralisable vers l'image de performance
    cm_image_path_ml = os.path.join(BASE_DIR, "app_img", "Performance Machin Learning.png")

    # 📌 Vérifier si le fichier existe avant d'afficher
    if os.path.exists(cm_image_path_ml):
        with st.expander("📊 Afficher l'Évaluation du Modèle"):
            st.image(cm_image_path_ml, caption="Performance du Modèle Machine Learning", use_container_width=True)
    else:
        st.warning(f"⚠️ L'image de performance n'a pas été trouvée : {cm_image_path_ml}")


    # Ajout de la phrase de précision sous la matrice de confusion
    st.markdown(
        """
        🔍 **Fiabilité du modèle Machine Learning :**

        ✅ **98.4 % des tumeurs malignes** sont correctement détectées.

        ✅ **99.1 % des tumeurs bénignes** sont correctement détectées.
        """
    )



# ////////////////////// Page de prédiction DEEP LEARNING /////////////////////////////

if page == "Prédiction Mammographie (DL)":
    # Configuration de la page
    st.title("Prédiction de Tumeurs via Deep Learning")
    #scénario
    st.write("Scénario 2 : Un médecin reçoit des mammographies → utilisation du Deep Learning pour l’analyse d’images.")

    # ---------------------- SECTION EXPLICATION DEEP LEARNING ----------------------
    st.subheader("Qu'est-ce que le Deep Learning ?")

    with st.expander("Le Deep Learning en quelques mots"):
        st.write("""

        🧠 Vous vous rappelez de l'exemple de l'enfant qui apprend a reconnaitre les chocolatines grâce à des images et du parallèle avec le Deep Learning ?

        📸 Application à notre cas :
        **Nous avons montré au modèle des** milliers de mammographies, **afin qu'il apprenne à détecter la présence ou l'absence d'une tumeur**.
        """)

       # 🔍 Comme expliqué dans l'introduction, le Deep Learning est une branche de l'intelligence artificielle.
    # ---------------------- SECTION NOTRE DÉFI ----------------------
    st.subheader("Notre défi ?")

    with st.expander("Analyser les mammographies"):
        st.write("""
        🔍 Notre défi était de pouvoir Développer un modèle capable d'analyser les images de mammographies et détecter la présence d'une tumeur.

        🎯 Pour cela, nous avons réalisé une interface permettant aux utilisateurs de télécharger une image de mammographie, que notre modèle analysera pour fournir une prédiction.
        """)

    # ---------------------- OUTIL DE PRÉDICTION ----------------------
    st.subheader("Outil de prédiction")
    st.write("")

    # ---------------------- PREMIER UPLOAD D'IMAGE AVEC PRÉDICTION ----------------------

    st.subheader("📸 Analyse de Mammographie 1")
    st.write("Téléchargez une image de mammographie et appuyez sur **Prédiction** pour obtenir le résultat.")

    uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        # Redimensionner l’image pour l’affichage
        max_width = 500
        w_percent = max_width / float(image.size[0])
        new_height = int(float(image.size[1]) * w_percent)
        image_resized = image.resize((max_width, new_height), Image.LANCZOS)

        # Afficher l’image redimensionnée
        st.image(image_resized, caption="Image Redimensionnée", use_container_width=False)

        # Bouton de prédiction
        if st.button("Lancer la prédiction"):
            img_bytes = io.BytesIO()
            image.save(img_bytes, format="PNG")
            img_bytes = img_bytes.getvalue()

            files = {"file": ("image.png", img_bytes, "image/png")}

            try:
                response = requests.post(DL_API_URL, files=files)

                if response.status_code == 200:
                    result = response.json()
                    diagnostic = result.get('diagnostic', 'Inconnu')

                    if "Positif" in diagnostic:
                        diagnostic_text = "🔴 Positif (1) : Tumeur détectée"
                        color_code = "#F76C6C"
                    else:
                        diagnostic_text = "🔵 Négatif (0) : Pas de Tumeur détectée"
                        color_code = "#A1C4FD"

                    st.markdown(
                        f'<div style="background-color:{color_code}; padding:15px; border-radius:10px; text-align:center; '
                        f'font-size:16px; color:white; font-weight:bold;">'
                        f'{diagnostic_text}'
                        '</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.error("Erreur lors de la requête à l’API")

            except Exception as e:
                st.error(f"Erreur lors de l’appel API : {e}")

    # ---------------------- DEUXIÈME UPLOAD D'IMAGE AVEC PRÉDICTION ----------------------

    st.subheader("📸 Analyse de Mammographie 2")
    st.write("Téléchargez une image de mammographie et appuyez sur **Prédiction** pour obtenir le résultat.")

    uploaded_file_2 = st.file_uploader("", type=["png", "jpg", "jpeg"], key="uploader_2")

    if uploaded_file_2 is not None:
        image_2 = Image.open(uploaded_file_2)

        # Redimensionner l’image pour l’affichage
        max_width = 500
        w_percent = max_width / float(image_2.size[0])
        new_height = int(float(image_2.size[1]) * w_percent)
        image_resized_2 = image_2.resize((max_width, new_height), Image.LANCZOS)

        # Afficher l’image redimensionnée
        st.image(image_resized_2, caption="Deuxième Image Redimensionnée", use_container_width=False)

        # Bouton de prédiction pour la deuxième image
        if st.button("Lancer la deuxième prédiction"):
            img_bytes_2 = io.BytesIO()
            image_2.save(img_bytes_2, format="PNG")
            img_bytes_2 = img_bytes_2.getvalue()

            files_2 = {"file": ("image2.png", img_bytes_2, "image/png")}

            try:
                response_2 = requests.post(DL_API_URL, files=files_2)

                if response_2.status_code == 200:
                    result_2 = response_2.json()
                    diagnostic_2 = result_2.get('diagnostic', 'Inconnu')

                    if "Positif" in diagnostic_2:
                        diagnostic_text_2 = "🔴 Positif (1) : Tumeur détectée"
                        color_code_2 = "#F76C6C"
                    else:
                        diagnostic_text_2 = "🔵 Négatif (0) : Pas de Tumeur détectée"
                        color_code_2 = "#A1C4FD"

                    st.markdown(
                        f'<div style="background-color:{color_code_2}; padding:15px; border-radius:10px; text-align:center; '
                        f'font-size:16px; color:white; font-weight:bold;">'
                        f'{diagnostic_text_2}'
                        '</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.error("Erreur lors de la requête à l’API")

            except Exception as e:
                st.error(f"Erreur lors de l’appel API : {e}")

# Fonction pour afficher la matrice de confusion dans Streamlit
# 📌 Ajout dans la section "Prédiction Mammographie (DL)"
# 📌 Fonction pour afficher l'image de performance du modèle Deep Learning
if page == "Prédiction Mammographie (DL)":
    st.subheader("📊 Performance de notre Modèle de Deep Learning")

    # 📌 Déterminer dynamiquement le dossier contenant `app.py`
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # 📌 Construire un chemin relatif et généralisable vers l'image de performance
    cm_image_path = os.path.join(BASE_DIR, "app_img", "Performance Deep Learning.png")

    # 📌 Vérifier si le fichier existe avant d'afficher
    if os.path.exists(cm_image_path):
        with st.expander("📊 Afficher l'Évaluation du Modèle"):
            st.image(cm_image_path, caption="Performance du Modèle Deep Learning", use_container_width=True)
    else:
        st.warning(f"⚠️ L'image de performance n'a pas été trouvée : {cm_image_path}")

    st.markdown(
        """
        🔍 **Fiabilité du modèle Deep Learning :**

        ✅ **98.0 % des patientes avec tumeur** sont correctement détectées.

        ✅ **97.9 % des patientes sans tumeur** sont bien identifiées.
        """
    )
