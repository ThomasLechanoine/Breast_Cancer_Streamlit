import os

##################  VARIABLES ENVIRONNEMENTALES  ##################
GCP_PROJECT = os.environ.get('GCP_PROJECT')
GCP_REGION = os.environ.get('GCP_REGION')
BUCKET_NAME = os.environ.get('BUCKET_NAME')
MODEL_TARGET = os.environ.get('MODEL_TARGET')
BUCKET_MODEL = os.environ.get('BUCKET_MODEL')

####### Déterminer le chemin de base du projet (dossier contenant ce script)####
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

##################  CHEMINS DES DONNÉES  ##################
ML_DATA_PATH = os.path.join(BASE_DIR, "Data", "Machine_learning.csv")
DL_ZIP_PATH = os.path.join(BASE_DIR, "Data", "Data_prepros.zip")
DL_DATA_PATH = os.path.join(BASE_DIR, "Data", "Data_Deep_Learning")

DL_DATASET_PATH = os.path.join(BASE_DIR, "Data/Data_Deep_Learning")

##################  CHEMINS DES MODÈLES  ##################
DL_MODEL_PATH = os.path.join(BASE_DIR, "Deep_learning", "models_saved", "best_model.h5")
ML_MODEL_PATH = os.path.join(BASE_DIR, "Machine_learning", "models_saved", "ml_best_model.pkl")
ML_SCALER_PATH = os.path.join(BASE_DIR, "Machine_learning", "models_saved", "ml_scaler.pkl")

##################  PARAMÈTRES D'ENTRAÎNEMENT  ##################
DL_BATCH_SIZE = 32
DL_IMG_SIZE = (224, 224)
DL_EPOCHS = 5
DL_OPTIMIZER = 'adam'
DL_LOSS_FUNCTION = 'binary_crossentropy'
DL_METRICS = ['accuracy', 'recall']

##################  CONFIGURATION API  ##################
ML_API_URL = "https://breast-cancer-api-768158196387.europe-west1.run.app/predict_ml"  # ✅ Corrigé
DL_API_URL = "https://breast-cancer-api-768158196387.europe-west1.run.app/predict_dl"
