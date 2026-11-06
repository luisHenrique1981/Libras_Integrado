#!/usr/bin/env python3
"""Treina um classificador Keras para as features de landmarks da Libras."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

FEATURE_COLUMNS = [
    f"landmark_{landmark}_{axis}"
    for landmark in range(21)
    for axis in ("x", "y", "z")
]


def carregar_dataset(dataset_path: Path):
    """Carrega o CSV, valida o esquema e codifica os rótulos textuais."""
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder

    dataframe = pd.read_csv(dataset_path)
    expected_columns = [*FEATURE_COLUMNS, "label"]
    if list(dataframe.columns) != expected_columns:
        raise ValueError(
            "Esquema inválido. Esperado 63 features na ordem do extrator e label "
            f"como última coluna; recebido {len(dataframe.columns)} colunas."
        )
    if dataframe.empty:
        raise ValueError(f"Dataset vazio: {dataset_path}")
    if dataframe[FEATURE_COLUMNS].isna().any().any():
        raise ValueError("O dataset contém valores ausentes nas features.")

    labels = dataframe["label"].astype(str)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    X = dataframe[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    return X, y, label_encoder


def criar_modelo(input_size: int, class_count: int, dropout: float):
    """Cria uma rede densa decrescente com saída probabilística por classe."""
    import tensorflow as tf

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_size,)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(class_count, activation="softmax"),
        ],
        name="libras_landmarks_classifier",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def salvar_mapeamento(label_encoder, mapping_path: Path) -> None:
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    mapping = {str(index): label for index, label in enumerate(label_encoder.classes_)}
    mapping_path.write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("libras_AI_training/data/processed/dataset_libras.csv"),
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("libras_AI_training/artifacts/models/libras_landmarks.keras"),
    )
    parser.add_argument(
        "--mapping-output",
        type=Path,
        default=Path("libras_AI_training/artifacts/configs/libras_label_mapping.json"),
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 0:
        raise ValueError("epochs e batch-size devem ser positivos; patience não pode ser negativo")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("dropout deve estar entre 0 e 1")

    import numpy as np
    import tensorflow as tf
    from sklearn.model_selection import train_test_split

    X, y, label_encoder = carregar_dataset(args.dataset)
    class_counts = np.bincount(y)
    if len(class_counts) < 2:
        raise ValueError("O dataset precisa conter pelo menos duas classes.")
    if np.min(class_counts) < 2:
        raise ValueError("Cada classe precisa ter pelo menos duas amostras para a divisão estratificada.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=args.random_state,
        stratify=y,
    )
    model = criar_modelo(X.shape[1], len(label_encoder.classes_), args.dropout)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=args.patience,
        restore_best_weights=True,
    )
    model.fit(
        X_train,
        y_train,
        validation_split=0.20,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[early_stopping],
        verbose=1,
        shuffle=True,
    )

    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.model_output)
    salvar_mapeamento(label_encoder, args.mapping_output)
    print(f"Modelo salvo em {args.model_output}")
    print(f"Mapeamento salvo em {args.mapping_output}")
    print(f"Teste: loss={test_loss:.4f}, accuracy={test_accuracy:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())