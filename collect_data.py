"""
collect_data.py

Coleta dados de sinais de LIBRAS usando a webcam + MediaPipe HandLandmarker
(API nova / Tasks API), com aprendizado incremental por palavra.

FLUXO:
  1. Ao iniciar (ou quando quiser trocar de sinal), digite no terminal
     a PALAVRA do sinal que vai gravar (ex: "oi").
  2. Na janela da webcam, pressione 'r' para LIGAR a gravacao continua
     (grava um frame a cada instante enquanto a mao estiver visivel).
     Pressione 'r' de novo para PAUSAR a gravacao daquele sinal.
     Voce pode ligar/pausar quantas vezes quiser - cada amostra ja e
     salva no disco na hora, entao pode parar a qualquer momento sem
     perder nada.
  3. Pressione 'n' a qualquer momento para trocar de palavra (digita a
     nova palavra no terminal). Se digitar uma palavra que ja existe
     no dataset, as novas amostras se SOMAM as que ja existem para
     aquele mesmo sinal (nao cria um rotulo duplicado).
  4. Pressione 'q' para sair.

Cada linha salva no CSV = 1 frame = 63 valores (21 pontos x,y,z) + label.
"""

import cv2
import csv
import os
import time

from mp_hand_utils import create_detector, frame_to_mp_image, normalize_landmarks, draw_landmarks

OUTPUT_FILE = "dataset.csv"
RECORD_INTERVAL = 0.1  # segundos entre amostras gravadas (evita frames quase identicos)


def ensure_csv_header(path, n_features=63):
    if not os.path.exists(path):
        header = [f"f{i}" for i in range(n_features)] + ["label"]
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)


def count_existing_samples(path, label):
    """Conta quantas amostras ja existem no CSV para essa palavra/label."""
    if not os.path.exists(path):
        return 0
    count = 0
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["label"] == label:
                count += 1
    return count


def ask_label():
    label = input("\nDigite a palavra/sinal que vai gravar agora: ").strip()
    while not label:
        label = input("Palavra vazia, digite novamente: ").strip()
    return label


def main():
    ensure_csv_header(OUTPUT_FILE)
    detector = create_detector(max_num_hands=1)

    current_label = ask_label()
    existing = count_existing_samples(OUTPUT_FILE, current_label)
    session_count = 0
    print(f"Sinal atual: '{current_label}' (ja tem {existing} amostras salvas)")
    print("Pressione 'r' para ligar/pausar a gravacao, 'n' para trocar de palavra, 'q' para sair.\n")

    cap = cv2.VideoCapture(0)
    recording = False
    last_saved = 0.0
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

        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]
            draw_landmarks(frame, landmarks)

            if recording:
                now = time.time()
                if now - last_saved >= RECORD_INTERVAL:
                    vec = normalize_landmarks(landmarks)
                    with open(OUTPUT_FILE, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(list(vec) + [current_label])
                    session_count += 1
                    existing += 1
                    last_saved = now

        # Overlay de status
        status = "GRAVANDO" if recording else "pausado"
        color = (0, 0, 255) if recording else (0, 255, 0)
        cv2.putText(frame, f"Sinal: {current_label} [{status}]", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Amostras desse sinal: {existing} (sessao: {session_count})",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, "r: liga/pausa | n: trocar palavra | q: sair",
                    (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (200, 200, 200), 1)

        cv2.imshow("Coleta de sinais de LIBRAS", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('r'):
            recording = not recording
            print(f"Gravacao {'LIGADA' if recording else 'PAUSADA'} para '{current_label}'")

        elif key == ord('n'):
            recording = False
            current_label = ask_label()
            existing = count_existing_samples(OUTPUT_FILE, current_label)
            session_count = 0
            print(f"Sinal atual: '{current_label}' (ja tem {existing} amostras salvas)")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nColeta finalizada. Dados salvos em '{OUTPUT_FILE}'.")


if __name__ == "__main__":
    main()
