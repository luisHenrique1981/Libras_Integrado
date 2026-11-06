#!/usr/bin/env python3
"""Prediz o sinal Libras de um vídeo usando landmarks de mãos e um modelo Keras."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


FEATURE_COUNT = 63


def validar_arquivo(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{description} não encontrado: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"{description} vazio: {path}")


def carregar_mapeamento(mapping_path: Path) -> dict[int, str]:
    validar_arquivo(mapping_path, "Mapeamento de classes")
    try:
        raw_mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Mapeamento JSON inválido: {mapping_path}") from error

    if not isinstance(raw_mapping, dict) or not raw_mapping:
        raise ValueError("O mapeamento precisa ser um objeto JSON não vazio.")
    try:
        mapping = {int(index): str(label) for index, label in raw_mapping.items()}
    except (TypeError, ValueError) as error:
        raise ValueError("Os índices do mapeamento devem ser inteiros.") from error

    expected_indices = set(range(len(mapping)))
    if set(mapping) != expected_indices or any(not label.strip() for label in mapping.values()):
        raise ValueError("O mapeamento deve conter índices consecutivos e rótulos não vazios.")
    return mapping


def carregar_modelo(model_path: Path, class_count: int):
    validar_arquivo(model_path, "Modelo Keras")
    try:
        import tensorflow as tf
    except ImportError as error:
        raise RuntimeError(
            "TensorFlow não está instalado na virtualenv. "
            "Instale tensorflow antes da inferência."
        ) from error

    try:
        model = tf.keras.models.load_model(model_path, compile=False)
    except Exception as error:
        raise ValueError(f"Não foi possível carregar o modelo Keras: {model_path}") from error

    input_shape = model.input_shape
    output_shape = model.output_shape
    if len(input_shape) != 2 or input_shape[-1] != FEATURE_COUNT:
        raise ValueError(f"Modelo incompatível: entrada esperada ({FEATURE_COUNT},), recebida {input_shape}.")
    if len(output_shape) != 2 or output_shape[-1] != class_count:
        raise ValueError(
            f"Modelo/mapeamento incompatíveis: modelo retorna {output_shape[-1]} classes, "
            f"mas o mapeamento possui {class_count}."
        )
    return model


def extrair_features(video_path: Path, hands) -> list[list[float]]:
    import cv2

    validar_arquivo(video_path, "Vídeo de entrada")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Não foi possível abrir o vídeo: {video_path}")

    features: list[list[float]] = []
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not result.multi_hand_landmarks:
                continue

            landmarks = result.multi_hand_landmarks[0].landmark
            values = [coordinate for landmark in landmarks for coordinate in (landmark.x, landmark.y, landmark.z)]
            if len(values) == FEATURE_COUNT:
                features.append(values)
    finally:
        capture.release()
    return features


def inferir(video_path: Path, model_path: Path, mapping_path: Path, confidence: float) -> dict:
    import mediapipe as mp
    import numpy as np

    mapping = carregar_mapeamento(mapping_path)
    model = carregar_modelo(model_path, len(mapping))
    hands_module = mp.solutions.hands
    with hands_module.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=confidence,
        min_tracking_confidence=confidence,
    ) as hands:
        features = extrair_features(video_path, hands)

    if not features:
        raise ValueError("Nenhuma mão foi detectada no vídeo; não há amostras para inferência.")

    probabilities = model.predict(np.asarray(features, dtype=np.float32), verbose=0)
    mean_probabilities = np.asarray(probabilities, dtype=np.float64).mean(axis=0)
    predicted_index = int(np.argmax(mean_probabilities))
    return {
        "label": mapping[predicted_index],
        "class_index": predicted_index,
        "confidence": float(mean_probabilities[predicted_index]),
        "valid_frames": len(features),
        "aggregation": "mean_softmax_probabilities",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Caminho local do vídeo a classificar.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("libras_AI_training/artifacts/models/libras_landmarks.keras"),
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=Path("libras_AI_training/artifacts/configs/libras_label_mapping.json"),
    )
    parser.add_argument("--output-json", type=Path, help="Arquivo opcional para salvar o resultado.")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confiança mínima do MediaPipe (0 a 1).")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not 0.0 <= args.confidence <= 1.0:
        raise ValueError("--confidence deve estar entre 0 e 1")

    result = inferir(args.video, args.model, args.mapping, args.confidence)
    serialized = json.dumps(result, ensure_ascii=False, indent=2)
    print(serialized)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(serialized + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())