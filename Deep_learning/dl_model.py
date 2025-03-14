import os
import numpy as np
from tensorflow import keras
from keras import Model, Sequential, layers, regularizers, optimizers
from keras.regularizers import l1_l2, l2
# from keras.preprocessing import image_dataset_from_directory
from keras.callbacks import EarlyStopping, ModelCheckpoint

#///////////initialize_model/////////////
def dl_initialize_model():
    model = Sequential([
        layers.Input((224, 224, 3)),
        layers.Conv2D(16, (3, 3), activation="relu", kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
        layers.MaxPool2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation="relu", kernel_regularizer=l2(0.01)),
        layers.MaxPool2D((2, 2)),
        layers.Dropout(0.2),
        layers.Conv2D(64, (3, 3), activation="relu", kernel_regularizer=l2(0.01)),
        layers.MaxPool2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu", kernel_regularizer=l2(0.01)),
        layers.MaxPool2D((2, 2)),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(32, activation="relu", kernel_regularizer=l2(0.01)),
        layers.Dropout(0.2),
        layers.Dense(16, activation="relu"),
        layers.Dense(8, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])

    model.summary()
    print("✅ Model initialized")
    return model

#///////////compile_model/////////////
def dl_compile_model(model, optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'recall']):
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    print("✅ Model compiled")
    return model

#///////////train_model/////////////
def dl_train_model(model, train_dataset, validation_dataset, epochs=30):
    model_dir = "Deep_learning/models_saved"
    os.makedirs(model_dir, exist_ok=True)

    callbacks = [
        ModelCheckpoint(os.path.join(model_dir, 'best_model.h5'),
                        save_best_only=True,
                        monitor='val_recall',
                        mode='max'),
        EarlyStopping(monitor='val_loss', patience=15, verbose=1, mode='min')
    ]

    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=validation_dataset,
        callbacks=callbacks,
        verbose=1
    )
    return model, history
