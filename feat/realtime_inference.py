# feat/alert-system/realtime_inference.py
# ywoo 브랜치용 실시간 환경음 추론 및 알림 출력 코드

import argparse
import os
import tempfile

import numpy as np
import librosa
import sounddevice as sd
from PIL import Image


# =========================
# 기본 설정
# =========================

SAMPLE_RATE = 22050
DURATION = 3
N_MELS = 64
MAX_LEN = 64

CLASS_NAMES = [
    "0_indoor_alarms",
    "1_outdoor_warnings",
    "2_emergency_alarms"
]

ALERT_INFO = {
    "0_indoor_alarms": {
        "title": "위험도 1: 실내 생활 알림음",
        "message": "실내 생활 알림음이 감지되었습니다. 전화벨, 초인종, 타이머 등을 확인하세요.",
        "color": "YELLOW"
    },
    "1_outdoor_warnings": {
        "title": "위험도 2: 실외 주의 소리",
        "message": "실외 주의 소리가 감지되었습니다. 자동차 경적이나 사이렌 등 주변 상황을 확인하세요.",
        "color": "ORANGE"
    },
    "2_emergency_alarms": {
        "title": "위험도 3: 응급/위험 경보음",
        "message": "응급 또는 위험 경보음이 감지되었습니다. 즉시 주변 상황을 확인하세요.",
        "color": "RED"
    }
}


# =========================
# 1. 마이크 입력
# =========================

def record_audio():
    """
    로컬 PC 마이크로 3초 동안 소리를 녹음한다.
    """

    print(f"\n{DURATION}초 동안 주변 소리를 녹음합니다...")

    audio = sd.rec(
        int(SAMPLE_RATE * DURATION),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )

    sd.wait()

    return audio.flatten()


# =========================
# 2. 오디오 → Mel-Spectrogram
# =========================

def audio_to_mel(audio):
    """
    녹음된 오디오 데이터를 Mel-Spectrogram으로 변환한다.
    CNN 입력 shape: (64, 64, 1)
    """

    target_length = SAMPLE_RATE * DURATION

    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)))
    else:
        audio = audio[:target_length]

    audio = librosa.util.normalize(audio)

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_mels=N_MELS
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)

    if mel_db.shape[1] < MAX_LEN:
        mel_db = np.pad(
            mel_db,
            ((0, 0), (0, MAX_LEN - mel_db.shape[1]))
        )
    else:
        mel_db = mel_db[:, :MAX_LEN]

    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)

    return mel_db[..., np.newaxis]


def save_mel_as_image(mel_data):
    """
    YOLOv8-cls 모델(.pt) 사용 시 Mel-Spectrogram을 이미지 파일로 저장한다.
    """

    mel_img = mel_data.squeeze()
    mel_img = (mel_img * 255).astype(np.uint8)

    image = Image.fromarray(mel_img).convert("RGB")

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    image.save(temp_file.name)

    return temp_file.name


# =========================
# 3. 모델 로드
# =========================

def load_sound_model(model_path):
    """
    학습된 모델 파일을 불러온다.
    지원 형식:
    - .h5
    - .keras
    - .pt
    """

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")

    if model_path.endswith(".pt"):
        from ultralytics import YOLO
        model = YOLO(model_path)
        model_type = "yolo"

    elif model_path.endswith(".h5") or model_path.endswith(".keras"):
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path)
        model_type = "keras"

    else:
        raise ValueError("지원하지 않는 모델 형식입니다. .pt, .h5, .keras 중 하나를 사용하세요.")

    return model, model_type


# =========================
# 4. 모델 예측
# =========================

def normalize_class_name(class_name):
    """
    모델이 반환한 클래스명을 우리 프로젝트 기준 클래스명으로 정리한다.
    """

    class_name = str(class_name)

    if class_name.startswith("0"):
        return "0_indoor_alarms"

    if class_name.startswith("1"):
        return "1_outdoor_warnings"

    if class_name.startswith("2"):
        return "2_emergency_alarms"

    return class_name


def predict_sound(model, model_type, mel_data):
    """
    Mel-Spectrogram 데이터를 학습된 모델에 넣어
    위험도 클래스를 예측한다.
    """

    if model_type == "keras":
        input_data = np.expand_dims(mel_data, axis=0)

        prediction = model.predict(input_data, verbose=0)[0]

        class_idx = int(np.argmax(prediction))
        confidence = float(prediction[class_idx])
        class_name = CLASS_NAMES[class_idx]

        return class_name, confidence

    if model_type == "yolo":
        mel_image_path = save_mel_as_image(mel_data)

        results = model.predict(
            source=mel_image_path,
            imgsz=64,
            verbose=False
        )

        probs = results[0].probs
        class_idx = int(probs.top1)

        confidence = float(probs.top1conf.cpu().item())

        raw_class_name = model.names[class_idx]
        class_name = normalize_class_name(raw_class_name)

        os.remove(mel_image_path)

        return class_name, confidence


# =========================
# 5. 알림 생성
# =========================

def create_alert_result(class_name, confidence):
    """
    모델 예측 결과를 위험도 알림 정보로 변환한다.
    """

    if confidence < 0.70:
        return {
            "class_name": class_name,
            "title": "판단 보류",
            "confidence": confidence,
            "color": "GRAY",
            "message": "소리가 명확하지 않습니다. 다시 확인해 주세요."
        }

    info = ALERT_INFO.get(class_name)

    if info is None:
        return {
            "class_name": class_name,
            "title": "알 수 없는 소리",
            "confidence": confidence,
            "color": "GRAY",
            "message": "등록되지 않은 소리가 감지되었습니다."
        }

    return {
        "class_name": class_name,
        "title": info["title"],
        "confidence": confidence,
        "color": info["color"],
        "message": info["message"]
    }


def print_alert(alert_result):
    """
    알림 결과를 콘솔 화면에 출력한다.
    """

    print("\n" + "=" * 60)
    print("[생활 소리 감지 결과]")
    print(f"분류: {alert_result['title']}")
    print(f"클래스: {alert_result['class_name']}")
    print(f"신뢰도: {alert_result['confidence'] * 100:.2f}%")
    print(f"화면 색상: {alert_result['color']}")
    print(f"알림 메시지: {alert_result['message']}")
    print("=" * 60)


# =========================
# 6. 실시간 실행
# =========================

def run_realtime(model_path):
    """
    마이크 입력을 반복적으로 받아 소리를 예측한다.
    """

    model, model_type = load_sound_model(model_path)

    print("실시간 환경음 인식 기반 배리어프리 알림 시스템")
    print(f"사용 모델: {model_path}")
    print(f"모델 형식: {model_type}")
    print("종료하려면 q를 입력하세요.")

    while True:
        user_input = input("\n소리 감지를 시작하려면 Enter를 누르세요: ")

        if user_input.lower() == "q":
            print("프로그램을 종료합니다.")
            break

        audio = record_audio()

        mel_data = audio_to_mel(audio)

        class_name, confidence = predict_sound(
            model,
            model_type,
            mel_data
        )

        alert_result = create_alert_result(
            class_name,
            confidence
        )

        print_alert(alert_result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        help="학습된 모델 파일 경로. 예: model/best.pt 또는 model/sound_alert_model.keras"
    )

    args = parser.parse_args()

    run_realtime(args.model)
