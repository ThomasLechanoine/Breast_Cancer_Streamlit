import streamlit as st
import requests
from PIL import Image
import io
import tensorflow as tf
import joblib
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Machine Learning Imports
from Machine_learning.ml_preprocess import load_data, preprocess_data
from Machine_learning.ml_model import create_model, tune_hyperparameters, evaluate_model

# Add Machine_learning/ to sys.path for module access
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "Machine_learning")))

# ---------------------- CONFIGURATION ----------------------
st.set_page_config(page_title="Application de Détection de Cancer", layout="wide")

# Image à afficher à gauche dans la sidebar
image_path_left = os.path.join("/home", "bren", "code", "ThomasLechanoine", "Breast_Cancer_Project", "app_img", "01.png")
image = Image.open(image_path_left)

# Afficher l'image sur la barre latérale
st.sidebar.image(image_path_left, use_container_width=True)

# Load and display the cover image
# image_path = os.path.join("/home", "bren", "code", "ThomasLechanoine", "Breast_Cancer_Project", "app_img", "01.png")
# st.image(image_path, use_container_width=True)

# ---------------------- CUSTOM PAGE NAVIGATION ----------------------
st.sidebar.title("Navigation")

# Maintain session state for page navigation
if "page" not in st.session_state:
    st.session_state.page = "Graphiques"

if st.sidebar.button("Graphiques"):
    st.session_state.page = "Graphiques"
if st.sidebar.button("Prédiction Mammographie (DL)"):
    st.session_state.page = "Prédiction Mammographie (DL)"
if st.sidebar.button("Prédiction Cancer (ML)"):
    st.session_state.page = "Prédiction Cancer (ML)"

page = st.session_state.page


# ---------------------- GRAPHICS PAGE ----------------------
if page == "Graphiques":
    st.title("Visualisation des Graphiques")
    st.write("Analyse des données avec des visualisations graphiques.")

    # Définition du répertoire contenant les graphiques
    graph_dir = os.path.join("/home", "bren", "code", "ThomasLechanoine", "Breast_Cancer_Project", "app_img")

    # Liste des graphiques avec descriptions
    graph_data = [
        {"file": "image_graph1.png", "title": "Graphique 1", "description": "📊 Ce graphique montre la distribution des caractéristiques du dataset."},
        {"file": "image_graph2.png", "title": "Graphique 2", "description": "🔬 Cette visualisation met en évidence la corrélation entre les différentes variables."},
        {"file": "image_graph3.png", "title": "Graphique 3", "description": "📈 Analyse des performances du modèle avec différentes métriques d’évaluation."}
    ]

    # Affichage des images avec menu déroulant pour description
    for graph in graph_data:
        img_path = os.path.join(graph_dir, graph["file"])

        with st.expander(f"📊 {graph['title']}"):
            st.image(img_path, use_column_width=True)
            st.write(graph["description"])


# ---------------------- LOAD MODELS DL---------------------
@st.cache_resource
def load_dl_model():
    return tf.keras.models.load_model("Deep_learning/models_saved/best_model.h5")
 #//////////

model = load_dl_model()

# ---------------------- LOAD MODELS ML---------------------
# Load the trained model and scaler
@st.cache_resource
def load_model():
    MODEL_PATH = "/home/bren/code/ThomasLechanoine/Breast_Cancer_Project/Machine_learning/models_saved/ml_best_model.pkl"
    SCALER_PATH = "/home/bren/code/ThomasLechanoine/Breast_Cancer_Project/Machine_learning/models_saved/ml_scaler.pkl"
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

model, scaler = load_model()

# ---------------------- LOAD test data---------------------
@st.cache_resource
def load_test_data():
    dataset_path = os.path.join("/home", "bren", "code", "ThomasLechanoine", "Breast_Cancer_Project", "Data", "Machine_learning.csv")
    data = pd.read_csv(dataset_path)
    X = data.drop(columns=["id", "diagnosis"])  # Drop unnecessary columns
    y = data["diagnosis"].map({"B": 0, "M": 1})  # Encode labels (B:0, M:1)

    # Split into train and test (must match how the model was trained)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    return X_test, y_test

# Load test dataset
X_test, y_test = load_test_data()


#//////////////////////Page de prediction DEEP LEARNING/////////////////////////////
if page == "Prédiction Mammographie (DL)":
    # Configuration de la page
    st.title("Prédiction de Cancer via Mammographie")
    st.write("Téléchargez une image de mammographie et appuyez sur **Prédiction** pour obtenir le résultat.")

    # Ajout d'un uploader pour charger une image
    uploaded_file = st.file_uploader("Téléchargez une image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

    API_URL = "http://127.0.0.1:8000/predict"  # Plus tard, il suffira de changer cette URL vers ton API cloud

    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file)

        # Resize image to fit in a smaller area (e.g., max width 500px)
        max_width = 500  # Set maximum width
        w_percent = max_width / float(image.size[0])
        new_height = int(float(image.size[1]) * w_percent)  # Maintain aspect ratio
        image_resized = image.resize((max_width, new_height), Image.LANCZOS)

        # Display resized image
        st.image(image_resized, caption="Image Redimensionnée", use_container_width=False)

        # Bouton de prédiction
        if st.button("Lancer la prédiction"):
            # Convertir l'image en bytes pour l'envoyer à l'API
            img_bytes = io.BytesIO()
            image.save(img_bytes, format="PNG")
            img_bytes = img_bytes.getvalue()

            # Envoi de l'image à l'API
            files = {"file": ("image.png", img_bytes, "image/png")}
            response = requests.post(API_URL, files=files)

            # Vérification de la réponse
            if response.status_code == 200:
                result = response.json()
                st.success(f"Résultat : {result['diagnostic']} ({result['probability']})")
            else:
                st.error("Erreur lors de la requête à l'API.")



# ///////////Page de prédiction via Machine Learning////////////
#----------------------------------------------
st.markdown("""
    <style>
        /* Fond général en bleu pastel */
        .stApp {
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
            border: 3px solid #FF6B6B !important; /* Bordure plus marquée */
            box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.2) !important; /* Ombre douce */
            transition: all 0.3s ease-in-out !important;
        }

        /* Effet hover sur les boutons */
        div.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {
            background-color: #FF6B6B !important; /* Rouge plus foncé */
            border: 3px solid #E63946 !important; /* Bordure plus foncée */
            box-shadow: 4px 4px 10px rgba(0, 0, 0, 0.3) !important; /* Ombre plus marquée */
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

        /* Sections dépliables (Expander) */
        .st-expander {
            background-color: #B6D0E2 !important;
            border: 2px solid #7DA0B6 !important;
            color: #1A1A1A !important;
        }

        /* Sidebar */
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
if page == "Prédiction Cancer (ML)":
    st.title("Prédiction de Cancer via Machine Learning")
    st.write("Veuillez entrer les mesures de la tumeur pour obtenir une prédiction.")

    # Default values for two sample predictions
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

    # Creating forms for two predictions
    st.subheader("Prédiction 1 (Maligne)")
    with st.form(key="prediction_form_1"):
        columns = st.columns(5)
        feature_values_1 = {}  # Dictionnaire pour stocker les valeurs des features

        for i, feature in enumerate(default_values_1.keys()):
            with columns[i % 5]:
                feature_values_1[feature] = st.number_input(
                    feature, min_value=0.0, format="%.4f", value=default_values_1[feature]
                )

        submit_button_1 = st.form_submit_button(label="Lancer la Prédiction 1")

    # Vérifier si submit_button_1 existe avant de l'utiliser
    if submit_button_1:
        input_data_1 = pd.DataFrame([list(feature_values_1.values())], columns=default_values_1.keys())
        input_data_scaled_1 = scaler.transform(input_data_1)
        prediction_1 = model.predict(input_data_scaled_1)[0]

        # Choisir la couleur du résultat selon la prédiction
        if prediction_1 == 1:
            diagnostic_1 = "🔴 Malin (Cancer)"
            color_1 = "#F76C6C"  # Rouge pastel pour cancer
        else:
            diagnostic_1 = "🔵 Bénin (Sans Cancer)"
            color_1 = "#A1C4FD"  # Bleu pastel pour bénin

        # Affichage du résultat avec une carte colorée
        st.markdown(
            f'<div style="background-color:{color_1}; padding:15px; border-radius:10px; text-align:center; font-size:16px; color:white; font-weight:bold;">'
            f'Résultat de la prédiction 1 : {diagnostic_1}'
            '</div>',
            unsafe_allow_html=True
        )


    # ---------------------------------------------------------------------------

    st.subheader("Prédiction 2 (Bénigne)")
    with st.form(key="prediction_form_2"):
        columns = st.columns(5)
        feature_values_2 = {}  # Nouveau dictionnaire pour la deuxième prédiction

        for i, feature in enumerate(default_values_2.keys()):
            with columns[i % 5]:
                feature_values_2[feature] = st.number_input(
                    feature, min_value=0.0, format="%.4f", value=default_values_2[feature]
                )

        submit_button_2 = st.form_submit_button(label="Lancer la Prédiction 2")

    # Vérifier si submit_button_2 est défini avant de l'utiliser
    if submit_button_2:
        input_data_2 = pd.DataFrame([list(feature_values_2.values())], columns=default_values_2.keys())
        input_data_scaled_2 = scaler.transform(input_data_2)
        prediction_2 = model.predict(input_data_scaled_2)[0]

        # Déterminer la couleur et l'affichage du diagnostic
        if prediction_2 == 1:
            diagnostic_2 = "🔴 Malin (Cancer)"
            color_2 = "#F76C6C"  # Rouge pastel pour cancer
        else:
            diagnostic_2 = "🔵 Bénin (Sans Cancer)"
            color_2 = "#A1C4FD"  # Bleu pastel pour bénin

        # Affichage du résultat sous forme de carte colorée
        st.markdown(
            f'<div style="background-color:{color_2}; padding:15px; border-radius:10px; text-align:center; font-size:16px; color:white; font-weight:bold;">'
            f'Résultat de la prédiction 2 : {diagnostic_2}'
            '</div>',
            unsafe_allow_html=True
        )



    #--------------------CONFUSION MATRIX------------------
    if submit_button_1 or submit_button_2:
        # Select appropriate input
        input_data = input_data_1 if submit_button_1 else input_data_2

        # Scale the input data
        input_data_scaled = scaler.transform(input_data)

        # Make the prediction
        prediction = model.predict(input_data_scaled)[0]
        diagnostic = "Malin (Cancer)" if prediction == 1 else "Bénin (Sans Cancer)"


        # Display the result
        st.success(f"Résultat de la prédiction : {diagnostic}")

        # Compute Confusion Matrix
        y_pred = model.predict(scaler.transform(X_test))  # Now X_test is properly loaded
        cm = confusion_matrix(y_test, y_pred)

        # Display the Confusion Matrix as a Heatmap
        fig, ax = plt.subplots(figsize=(4, 3))  # Adjust figure size to be smaller
        sns.heatmap(cm, annot=True, fmt='g', cmap="Purples", ax=ax)
        ax.set_xlabel("True Values")
        ax.set_ylabel("Predictions")
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)
