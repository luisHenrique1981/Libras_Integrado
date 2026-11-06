#!/usr/bin/env python3
"""Extrai landmarks das maos em videos rotulados e grava dataset_libras.csv."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import cv2
import mediapipe as mp


VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}
LANDMARK_COLUMNS = [
    f"landmark_{landmark}_{axis}"
    for landmark in range(21)
    for axis in ("x", "y", "z")
]


def listar_videos(videos_dir: Path) -> list[Path]:
    """Retorna videos locais em ordem estável, sem seguir links externos."""
    return sorted(
        path
        for path in videos_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def obter_rotulo(video_path: Path, videos_dir: Path) -> str:
    """Usa a pasta como rótulo; aceita também arquivos planos letra_rotulo.ext."""
    relative_parent = video_path.parent.relative_to(videos_dir)
    if relative_parent != Path("."):
        return relative_parent.parts[0]

    match = re.match(r"^[^_]+_(.+)$", video_path.stem)
    if match:
        return match.group(1)
    raise ValueError(
        f"Não foi possível inferir o rótulo de '{video_path}'. "
        "Organize o vídeo em videos_dir/rotulo/arquivo.mp4 ou use letra_rotulo.mp4."
    )


def extrair_video(
    video_path: Path,
    label: str,
    hands: object,
    writer: csv.writer,
) -> tuple[int, int]:
    """Extrai uma amostra por frame detectado e retorna (frames, amostras)."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        print(f"Aviso: não foi possível abrir {video_path}")
        return 0, 0

    frames = 0
    samples = 0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            frames += 1
            result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not result.multi_hand_landmarks:
                continue

            landmarks = result.multi_hand_landmarks[0].landmark
            values = [coordinate for landmark in landmarks for coordinate in (landmark.x, landmark.y, landmark.z)]
            if len(values) != len(LANDMARK_COLUMNS):
                continue

            writer.writerow([*values, label])
            samples += 1
    finally:
        capture.release()

    return frames, samples


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=Path("teachablemachine/videos_libras"),
        help="Diretório raiz com videos/rotulo/arquivo.mp4.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("libras_AI_training/data/processed/dataset_libras.csv"),
        help="CSV de saída com 63 features e a coluna label.",
    )
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    parser.add_argument("--max-num-hands", type=int, choices=(1, 2), default=1)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not 0.0 <= args.min_detection_confidence <= 1.0:
        raise ValueError("--min-detection-confidence deve estar entre 0 e 1")
    if not 0.0 <= args.min_tracking_confidence <= 1.0:
        raise ValueError("--min-tracking-confidence deve estar entre 0 e 1")
    if not args.videos_dir.is_dir():
        raise FileNotFoundError(f"Diretório de vídeos não encontrado: {args.videos_dir}")

    videos = listar_videos(args.videos_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    hands_module = mp.solutions.hands
    total_frames = 0
    total_samples = 0

    with hands_module.Hands(
        static_image_mode=False,
        max_num_hands=args.max_num_hands,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    ) as hands, args.output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow([*LANDMARK_COLUMNS, "label"])
        for video_path in videos:
            try:
                label = obter_rotulo(video_path, args.videos_dir)
                frames, samples = extrair_video(video_path, label, hands, writer)
            except (OSError, ValueError) as error:
                print(f"Aviso: vídeo ignorado ({video_path}): {error}")
                continue
            total_frames += frames
            total_samples += samples
            print(f"{video_path}: {samples} amostras válidas de {frames} frames")

    print(f"Dataset salvo em {args.output}: {total_samples} amostras, {len(LANDMARK_COLUMNS) + 1} colunas")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())