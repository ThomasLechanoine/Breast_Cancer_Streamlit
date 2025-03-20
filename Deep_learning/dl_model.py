import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers, Input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Layer
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import Constant

# Création d'une couche personnalisée pour les poids aléatoires fixes
class RandomFixedDense(Layer):
    def __init__(self, units, **kwargs):
        super(RandomFixedDense, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        stddev = 1.0 / np.sqrt(input_shape[-1])
        random_weights = np.random.normal(0, stddev, (input_shape[-1], self.units))

        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer=Constant(random_weights),
            trainable=False,
            name='kernel'
        )
        self.bias = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=False,
            name='bias'
        )

    def call(self, inputs):
        return tf.nn.relu(tf.matmul(inputs, self.kernel) + self.bias)

# Création de l'extracteur de caractéristiques
def create_feature_extractor(img_size):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(*img_size, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    model = Model(inputs=base_model.input, outputs=x)
    return model

# Définition du modèle edRVFL
def dl_initialize_edRVFL(input_dim, num_classes=1, num_layers=5, hidden_units=50):
    inputs = Input(shape=(input_dim,))
    hidden_layers = []

    for _ in range(num_layers):
        hidden_output = RandomFixedDense(hidden_units)(inputs)
        hidden_layers.append(hidden_output)

    concatenated = tf.keras.layers.Concatenate()(hidden_layers)
    outputs = Dense(num_classes, activation='sigmoid')(concatenated)
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Compilation du modèle
def dl_compile_model(model, optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.Recall()]):
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model

# Entraînement du modèle
def dl_train_model(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
    model_dir = "Deep_learning/models_saved"
    os.makedirs(model_dir, exist_ok=True)

    callbacks = [
        ModelCheckpoint(os.path.join(model_dir, 'best_model.h5'), save_best_only=True, monitor='val_accuracy', mode='max'),
        EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='min')
    ]

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks
    )

    return model, history
