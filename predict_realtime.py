"""
predict_realtime.py

Usa o modelo treinado (hand_gesture_model.h5) para reconhecer sinais de
LIBRAS em tempo real pela webcam, usando a API nova do MediaPipe
(HandLandmarker / Tasks API).
"""

import cv2
import time
import numpy as np
from tensorflow import keras

from mp_hand_utils import create_detector, frame_to_mp_image, normalize_landmarks, draw_landmarks

MODEL_FILE = "hand_gesture_model.h5"
ENCODER_FILE = "label_encoder.npy"
CONFIDENCE_THRESHOLD = 0.75  # abaixo disso, mostra "incerto"


def main():
    model = keras.models.load_model(MODEL_FILE)
    classes = np.load(ENCODER_FILE, allow_pickle=True)
    detector = create_detector(max_num_hands=1)

    cap = cv2.VideoCapture(0)
    start_time = time.time()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = frame_to_mp_image(rgb)
        timestamp_ms = int((time.time() - start_time) * 1000)
        result = detector.detect_for_video(mp_image, timestamp_ms)

        label_text = "Nenhuma mao detectada"

        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]
            draw_landmarks(frame, landmarks)

            vec = normalize_landmarks(landmarks)
            pred = model.predict(vec.reshape(1, -1), verbose=0)[0]
            idx = int(np.argmax(pred))
            confidence = float(pred[idx])

            if confidence >= CONFIDENCE_THRESHOLD:
                label_text = f"{classes[idx]} ({confidence*100:.1f}%)"
            else:
                label_text = f"Incerto ({confidence*100:.1f}%)"

        cv2.putText(frame, label_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Reconhecimento de sinais - LIBRAS", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
