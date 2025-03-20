import tensorflow_datasets as tfds
import tensorflow as tf
import os

class CustomImageDataset(tfds.core.GeneratorBasedBuilder):
    """Custom Dataset for images stored in folders"""

    VERSION = tfds.core.Version("1.0.0")

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description="Dataset d'images classifiées en 0 et 1",
            features=tfds.features.FeaturesDict({
                "image": tfds.features.Image(),
                "label": tfds.features.ClassLabel(names=["0", "1"]),
            }),
            supervised_keys=("image", "label"),
        )

    def _split_generators(self, dl_manager):
        """Définit les splits train, test et valid"""
        data_dir = os.path.abspath("Data/Data_Deep_Learning")  # Modifie selon ton chemin
        return {
            "train": self._generate_examples(os.path.join(data_dir, "train")),
            "test": self._generate_examples(os.path.join(data_dir, "test")),
            "valid": self._generate_examples(os.path.join(data_dir, "valid")),
        }

    def _generate_examples(self, path):
        """Génère les exemples à partir du dossier"""
        for label in ["0", "1"]:
            class_dir = os.path.join(path, label)
            for img_file in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_file)
                unique_key = os.path.relpath(img_path, start=path)
                yield unique_key, {
                    "image": img_path,
                    "label": int(label),
                }

# Fonction pour charger le dataset TFDS
def load_custom_dataset(data_dir, img_size, batch_size=16):
    """Charge les images depuis le dataset TFDS généré"""
    builder = CustomImageDataset(data_dir=data_dir)
    builder.download_and_prepare()
    datasets = builder.as_dataset(split=["train", "valid"], as_supervised=True)

    def preprocess(image, label):
        image = tf.image.resize(image, img_size) / 255.0  # Normalisation
        return image, label

    train_ds, val_ds = datasets
    train_ds = train_ds.map(preprocess).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(preprocess).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds
