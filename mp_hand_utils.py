"""
mp_hand_utils.py

Funcoes compartilhadas para deteccao de landmarks de mao usando a API
NOVA do MediaPipe (Tasks API / HandLandmarker), em vez da API antiga
'mediapipe.solutions' - que esta quebrada em varias instalacoes recentes
do mediapipe (erro: AttributeError: module 'mediapipe' has no attribute
'solutions'). A Tasks API e a que o Google mantem ativamente.

Na primeira execucao, baixa automaticamente o modelo 'hand_landmarker.task'
(alguns MB) direto dos servidores do Google, caso ainda nao exista na
pasta do projeto.
"""

import os
import urllib.request
import numpy as np
import cv2
import mediapipe as mp

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

# Conexoes entre os 21 pontos da mao (esqueleto), para desenho manual -
# a API nova nao traz mais 'drawing_utils' pronto como a antiga.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # polegar
    (0, 5), (5, 6), (6, 7), (7, 8),          # indicador
    (5, 9), (9, 10), (10, 11), (11, 12),     # medio
    (9, 13), (13, 14), (14, 15), (15, 16),   # anelar
    (13, 17), (17, 18), (18, 19), (19, 20),  # mindinho
    (0, 17),                                  # base da palma
]


def ensure_model():
    """Garante que o arquivo do modelo existe, baixando se necessario."""
    if not os.path.exists(MODEL_PATH):
        print("Baixando modelo hand_landmarker.task (uma vez so, ~10MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Modelo baixado com sucesso.")
    return MODEL_PATH


def create_detector(max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.6):
    """
    Cria um HandLandmarker no modo VIDEO (processamento sincrono, frame a
    frame, ideal para loop de webcam com cv2).
    """
    model_path = ensure_model()

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=max_num_hands,
        min_hand_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    return HandLandmarker.create_from_options(options)


def frame_to_mp_image(rgb_frame):
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)


def normalize_landmarks(landmarks):
    """
    Recebe uma lista de landmarks (com atributos x, y, z) e retorna um
    vetor numpy normalizado: centralizado no pulso e escalado pela maior
    distancia entre o pulso e os demais pontos.
    """
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    wrist = pts[0].copy()
    pts -= wrist

    max_dist = np.max(np.linalg.norm(pts, axis=1))
    if max_dist > 0:
        pts /= max_dist

    return pts.flatten()


def draw_landmarks(frame, landmarks):
    """Desenha os pontos e as conexoes da mao manualmente no frame (BGR)."""
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, pts[start], pts[end], (0, 255, 0), 2)
    for p in pts:
        cv2.circle(frame, p, 4, (0, 0, 255), -1)
