"""
train_model.py

Treina uma rede neural para classificar gestos de mão a partir dos
landmarks normalizados coletados em dataset.csv (gerado por collect_data.py).

Salva:
  - hand_gesture_model.h5   -> modelo treinado
  - label_encoder.npy       -> mapeamento entre índices e nomes dos gestos
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers

DATA_FILE = "dataset.csv"
MODEL_FILE = "hand_gesture_model.h5"
ENCODER_FILE = "label_encoder.npy"


def load_data():
    df = pd.read_csv(DATA_FILE)
    X = df.drop(columns=["label"]).values.astype("float32")
    y_raw = df["label"].values

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    return X, y, encoder


def build_model(input_dim, n_classes):
    """
    Rede neural densa (MLP) — mais que suficiente para 63 features
    numéricas (coordenadas dos landmarks), com capacidade boa para
    aprender fronteiras de decisão complexas entre gestos parecidos.
    """
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dense(n_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    X, y, encoder = load_data()
    n_classes = len(encoder.classes_)
    print(f"Classes encontradas: {list(encoder.classes_)}")
    print(f"Total de amostras: {len(X)}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_model(input_dim=X.shape[1], n_classes=n_classes)
    model.summary()

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=200,
        batch_size=16,
        callbacks=[early_stop],
        verbose=2,
    )

    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nAcuracia final na validacao: {val_acc*100:.2f}%")

    model.save(MODEL_FILE)
    np.save(ENCODER_FILE, encoder.classes_)
    print(f"Modelo salvo em '{MODEL_FILE}'")
    print(f"Classes salvas em '{ENCODER_FILE}'")


if __name__ == "__main__":
    main()
