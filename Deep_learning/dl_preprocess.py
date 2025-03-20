import tensorflow as tf
import os
from tensorflow.keras.applications.vgg16 import preprocess_input
from params import *
import numpy as np


def load_custom_dataset(img_size=(224, 224), batch_size=16):
    """
    Charge les images depuis un dossier structuré avec des sous-dossiers par classe.
    """

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(DL_DATASET_PATH, "train"),  # ✅ Utilisation de DL_DATASET_PATH
        image_size=img_size,
        batch_size=batch_size,
        label_mode="int"  # Labels entiers (0 et 1)
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(DL_DATASET_PATH, "valid"),  # ✅ Utilisation de DL_DATASET_PATH
        image_size=img_size,
        batch_size=batch_size,
        label_mode="int"
    )

    def preprocess(image, label):
        image = preprocess_input(tf.image.resize(image, img_size))  # Normalisation
        return image, label

    # Appliquer le prétraitement
    train_ds = train_ds.map(preprocess).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(preprocess).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds

# 📌 Tester si le dataset est bien chargé
train_ds, val_ds = load_custom_dataset()

for image_batch, label_batch in train_ds.take(1):
    print(f"Batch d'images: {image_batch.shape}")  # Devrait être (batch_size, 224, 224, 3)
    print(f"Batch de labels: {label_batch.numpy()}")  # Devrait contenir 0 ou 1


# def extract_features(feature_extractor, dataset):
#     features, labels = [], []
#     for images, labels_batch in dataset:
#         # Vérification : Si images est déjà un ndarray, ne pas appeler .numpy()
#         if isinstance(images, np.ndarray):
#             batch_features = feature_extractor.predict(images, verbose=0)
#         else:
#             batch_features = feature_extractor(images, training=False).numpy()

#         features.append(batch_features)
#         labels.append(labels_batch.numpy())

#     return np.vstack(features), np.concatenate(labels)

def extract_features(feature_extractor, dataset):
    features, labels = [], []
    for images, labels_batch in dataset:
        # Vérifier si images est déjà un numpy array
        if isinstance(images, np.ndarray):
            batch_features = feature_extractor.predict(images, verbose=0)
        else:
            batch_features = feature_extractor(images, training=False).numpy()

        features.append(batch_features)

        # Vérification pour labels_batch
        if hasattr(labels_batch, "numpy"):
            labels.append(labels_batch.numpy())
        else:
            labels.append(labels_batch)

    return np.vstack(features), np.concatenate(labels)
