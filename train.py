"""Plant Disease Detection Using CNN
Created by: Fazin Firoz
"""

import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
import os

IMG_SIZE = (64, 64)
BATCH_SIZE = 4

def load_dataset():
    df = pd.read_csv("dataset.csv")
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    train_gen = datagen.flow_from_dataframe(
        df,
        x_col="image_path",
        y_col="label",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training"
    )

    val_gen = datagen.flow_from_dataframe(
        df,
        x_col="image_path",
        y_col="label",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation"
    )

    return train_gen, val_gen

def build_model(num_classes):
    model = models.Sequential([
        layers.Input(shape=(*IMG_SIZE, 3)),
        layers.Conv2D(16, (3, 3), activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model

def main():
    train_gen, val_gen = load_dataset()
    num_classes = train_gen.num_classes
    model = build_model(num_classes)

    model.summary()

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=3
    )

    model.save("plant_disease_cnn.h5")

if __name__ == "__main__":
    main()
