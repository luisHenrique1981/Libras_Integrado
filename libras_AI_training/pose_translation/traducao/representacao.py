"""Projeção aproximada DWPose/MediaPipe para 62 marcos 2D.

A correspondência é aproximada, especialmente no rosto. A estrutura é idêntica
no treinamento e na inferência, mas a mudança de domínio exige validação.
"""
import numpy as np

# Ordem COCO-18 do DWPose: nariz, ombros, cotovelos, punhos, quadris, olhos
BODY_DW = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 14, 15]
# MP: nariz, centro ombros, ombro direito, cotovelo dir, punho dir,
# ombro esquerdo, cotovelo esq, punho esq, quadril dir, quadril esq, olhos
BODY_MP = [0, None, 12, 14, 16, 11, 13, 15, 24, 23, 5, 2]
# Pontos faciais próximos aos cantos dos olhos, da boca e centros dos lábios
FACE_DW = [36, 39, 42, 45, 48, 54, 62, 66]
FACE_MP = [33, 133, 362, 263, 61, 291, 13, 14]
N_PONTOS = 12 + 21 + 21 + 8
DIM = N_PONTOS * 3 + N_PONTOS * 2


def _normalizar(p):
    """Normaliza coordenadas usando ombros. Preserva confiança e movimento."""
    p = np.asarray(p, dtype=np.float32).copy()
    ok = p[:, 2] > 0.1
    ombros = p[[2, 5]]
    if (ombros[:, 2] > 0.1).all():
        centro = (ombros[0, :2] + ombros[1, :2]) / 2
        escala = np.linalg.norm(ombros[0, :2] - ombros[1, :2])
    else:
        corpo = p[:12]
        validos = corpo[corpo[:, 2] > 0.1, :2]
        if len(validos) == 0:
            centro, escala = np.array([0.5, 0.5]), 0.3
        else:
            centro = validos.mean(axis=0)
            escala = np.linalg.norm(validos.max(axis=0) - validos.min(axis=0)) * .4
    escala = max(float(escala), 0.05)
    p[:, :2] = np.clip((p[:, :2] - centro) / escala, -5, 5)
    p[~ok, :2] = 0.0
    p[~ok, 2] = 0.0
    return p


def _features(p):
    """Aceita T x 62 x 3; adiciona deslocamento relativo entre frames."""
    p = np.asarray(p, dtype=np.float32)
    if p.ndim != 3 or p.shape[1:] != (N_PONTOS, 3):
        raise ValueError(f'Esperado (T,{N_PONTOS},3), recebido {p.shape}')
    base = np.stack([_normalizar(frame) for frame in p])
    delta = np.diff(base[:, :, :2], axis=0, prepend=base[:1, :, :2])
    delta = np.clip(delta, -2, 2)
    return np.concatenate([base.reshape(len(p), -1), delta.reshape(len(p), -1)], axis=1).astype('float32')


def dwpose_para_features(sequencia):
    """Entrada SignVerse (T,128,3), x/y/confiança já normalizados."""
    s = np.asarray(sequencia, dtype=np.float32)
    if s.ndim != 3 or s.shape[1:] != (128, 3):
        raise ValueError(f'Esperado (T,128,3), recebido {s.shape}')
    # Ordenação arquivo: corpo[0:18], mão esquerda[18:39], direita[39:60], face[60:128]
    pontos = np.concatenate([
        s[:, BODY_DW], s[:, 18:39], s[:, 39:60], s[:, np.array(FACE_DW) + 60]
    ], axis=1)
    return _features(pontos)


def _get(lms, idx, fallback=None):
    if lms is None:
        return np.zeros(3, dtype=np.float32)
    if idx is None:
        a = _get(lms, 11)
        b = _get(lms, 12)
        return (a + b) * .5
    v = lms.landmark[idx]
    conf = float(getattr(v, 'visibility', 1.0))
    return np.array([v.x, v.y, conf], dtype=np.float32)


def mediapipe_frame(resultado):
    """Um frame bruto (62,3), API clássica mp.solutions.holistic."""
    corpo = np.stack([_get(resultado.pose_landmarks, k) for k in BODY_MP])
    esq = np.stack([_get(resultado.left_hand_landmarks, k) for k in range(21)])
    dire = np.stack([_get(resultado.right_hand_landmarks, k) for k in range(21)])
    rosto = np.stack([_get(resultado.face_landmarks, k) for k in FACE_MP])
    return np.concatenate([corpo, esq, dire, rosto], axis=0)


def mediapipe_sequencia(frames):
    return _features(np.asarray(frames, dtype=np.float32))
