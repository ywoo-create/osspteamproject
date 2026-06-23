import numpy as np
import sounddevice as sd
import librosa
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
import threading
import queue
import time
import sys

# ─── 설정 ───────────────────────────────────────────────────────────────────
WEIGHTS_PATH = r"C:\Users\parks\OSSP\audio\my_audio_weights.weights.h5"
SAMPLE_RATE  = 22050
DURATION     = 3.0
IMG_SIZE     = 64
CLASS_LABELS = ['0_indoor_alarms', '1_outdoor_warnings', '2_emergency_alarms']
ICONS        = ['🔵', '🟡', '🔴']

# ─── 모델 구조 재건 + 가중치 로드 ────────────────────────────────────────────
print("모델 구조 생성 ")
model = models.Sequential([
    layers.Input(shape=(64, 64, 1)),
    layers.Rescaling(1.0/80.0, offset=1.0),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(3, activation='softmax')
])
model.load_weights(WEIGHTS_PATH)
print(f" 가중치 로드 완료 입력 shape: {model.input_shape}\n")

# ─── 상태 관리 ───────────────────────────────────────────────────────────────
current_status = "대기 중"
buffer_size    = 0
CHUNK_SAMPLES  = int(SAMPLE_RATE * DURATION)

# ─── 특징 추출 ───────────────────────────────────────────────────────────────
def extract_features(audio: np.ndarray):
    try:
        mel     = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE)
        mel_db  = librosa.power_to_db(mel, ref=np.max)
        resized = cv2.resize(mel_db, (IMG_SIZE, IMG_SIZE))
        tensor  = resized[np.newaxis, :, :, np.newaxis]
        return tensor.astype(np.float32)
    except Exception as e:
        return None

def classify(audio: np.ndarray):
    features = extract_features(audio)
    if features is None:
        return "ERROR", 0.0, np.zeros(len(CLASS_LABELS))
    preds = model.predict(features, verbose=0)[0]
    idx   = int(np.argmax(preds))
    return CLASS_LABELS[idx], float(preds[idx]) * 100, preds

# ─── 상태 표시줄 ─────────────────────────────────────────────────────────────
def print_status_bar():
    filled = min(int((buffer_size / CHUNK_SAMPLES) * 20), 20)
    bar    = "▓" * filled + "░" * (20 - filled)
    pct    = int((buffer_size / CHUNK_SAMPLES) * 100)
    sys.stdout.write(f"\r⏺ 상태: {current_status:15s} │ 버퍼: [{bar}] {pct:3d}%  (Ctrl+C 로 종료)  ")
    sys.stdout.flush()

# ─── 오디오 콜백 ─────────────────────────────────────────────────────────────
audio_queue = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    audio_queue.put(indata[:, 0].copy())

# ─── 상태 표시 스레드 ────────────────────────────────────────────────────────
status_running = True

def status_loop():
    while status_running:
        print_status_bar()
        time.sleep(0.1)

# ─── 분류 루프 ───────────────────────────────────────────────────────────────
def classification_loop():
    global buffer_size, current_status
    buffer = np.zeros(0, dtype=np.float32)

    print("\n" + "─" * 72)
    print(f"  {'시각':^10} {'예측 클래스':^28} {'신뢰도':^8} {'확률 분포'}")
    print("─" * 72)

    while True:
        try:
            chunk        = audio_queue.get(timeout=1.0)
            buffer       = np.concatenate([buffer, chunk])
            buffer_size  = len(buffer)
            current_status = "🎙 입력 수집 중"

            if len(buffer) >= CHUNK_SAMPLES:
                current_status = "⚙ 분류 처리 중"
                segment = buffer[:CHUNK_SAMPLES]
                buffer  = buffer[CHUNK_SAMPLES:]
                buffer_size = len(buffer)

                label, conf, preds = classify(segment)
                idx  = CLASS_LABELS.index(label)
                icon = ICONS[idx]
                bar  = "█" * int(conf / 5) + "░" * (20 - int(conf / 5))
                prob_str = "  ".join(
                    f"{CLASS_LABELS[i].split('_',1)[1][:8]}:{preds[i]*100:4.1f}%"
                    for i in range(len(CLASS_LABELS))
                )

                sys.stdout.write("\n")
                print(f"  [{time.strftime('%H:%M:%S')}] {icon} {label:28s} {conf:5.1f}%  {bar}")
                print(f"             └─ {prob_str}")
                print("─" * 72)

                current_status = "🎙 입력 수집 중"

        except queue.Empty:
            current_status = "💤 대기 중"
            buffer_size    = 0
        except KeyboardInterrupt:
            break

# ─── 메인 ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n 실시간 오디오 분류기 시작")
    print("   종료하려면 → Ctrl + C\n")

    st = threading.Thread(target=status_loop, daemon=True)
    st.start()

    ct = threading.Thread(target=classification_loop, daemon=True)
    ct.start()

    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1,
        dtype="float32", blocksize=int(SAMPLE_RATE * 0.1),
        callback=audio_callback
    ):
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            status_running = False
            print("\n\n 종료되었습니다.")
