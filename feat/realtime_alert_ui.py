# feat/realtime_alert_ui.py
# 실시간 마이크 입력 기반 생활 소리 감지 + 알림 화면 출력 프로그램

import os
import threading
import time
import tkinter as tk
from tkinter import messagebox

import numpy as np
import librosa
import sounddevice as sd
import tensorflow as tf


# =========================
# 1. 기본 설정
# =========================

SAMPLE_RATE = 22050
DURATION = 3
N_MELS = 64
IMG_SIZE = 64
CONFIDENCE_THRESHOLD = 0.70

# 학습된 모델 파일 경로
# 모델 파일 이름이 다르면 여기만 바꾸면 됨
MODEL_PATH = "sound_alert_model.h5"

CLASS_NAMES = [
    "0_indoor_alarms",
    "1_outdoor_warnings",
    "2_emergency_alarms"
]

ALERT_INFO = {
    "0_indoor_alarms": {
        "risk": 1,
        "title": "위험도 1: 실내 생활 알림음",
        "message": "실내 생활 알림음이 감지되었습니다.\n전화벨, 초인종, 타이머 등을 확인하세요.",
        "color": "#FFD966"
    },
    "1_outdoor_warnings": {
        "risk": 2,
        "title": "위험도 2: 실외 주의 소리",
        "message": "실외 주의 소리가 감지되었습니다.\n자동차 경적이나 사이렌 등 주변 상황을 확인하세요.",
        "color": "#F6B26B"
    },
    "2_emergency_alarms": {
        "risk": 3,
        "title": "위험도 3: 응급/위험 경보음",
        "message": "응급 또는 위험 경보음이 감지되었습니다.\n즉시 주변 상황을 확인하세요.",
        "color": "#E06666"
    }
}


# =========================
# 2. 오디오 전처리 함수
# =========================

def audio_to_mel_spectrogram(audio):
    """
    마이크로 입력받은 오디오 데이터를
    CNN 입력용 Mel-Spectrogram 형태로 변환한다.

    최종 shape:
    (64, 64, 1)
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

    # 64 x 64 크기로 맞추기
    if mel_db.shape[1] < IMG_SIZE:
        mel_db = np.pad(
            mel_db,
            ((0, 0), (0, IMG_SIZE - mel_db.shape[1]))
        )
    else:
        mel_db = mel_db[:, :IMG_SIZE]

    # 만약 행 크기가 64가 아니면 보정
    if mel_db.shape[0] != IMG_SIZE:
        mel_db = mel_db[:IMG_SIZE, :]

    # 0~1 정규화
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)

    # CNN 입력 채널 추가
    mel_db = mel_db[..., np.newaxis]

    return mel_db


# =========================
# 3. 예측 함수
# =========================

def predict_sound(model, audio):
    """
    녹음된 소리를 Mel-Spectrogram으로 변환한 뒤
    CNN 모델로 3개 클래스 유사도/신뢰도를 계산한다.
    """

    mel_data = audio_to_mel_spectrogram(audio)

    input_data = np.expand_dims(mel_data, axis=0)

    prediction = model.predict(input_data, verbose=0)[0]

    class_index = int(np.argmax(prediction))
    confidence = float(prediction[class_index])
    class_name = CLASS_NAMES[class_index]

    return class_name, confidence, prediction


# =========================
# 4. GUI 클래스
# =========================

class SoundAlertApp:
    def __init__(self, root):
        self.root = root
        self.root.title("실시간 환경음 인식 기반 배리어프리 알림 시스템")
        self.root.geometry("520x520")
        self.root.resizable(False, False)

        self.model = None

        self.title_label = tk.Label(
            root,
            text="생활 소리 감지 시스템",
            font=("맑은 고딕", 22, "bold")
        )
        self.title_label.pack(pady=25)

        self.status_label = tk.Label(
            root,
            text="시작 버튼을 누르면 주변 소리를 듣습니다.",
            font=("맑은 고딕", 14),
            wraplength=440,
            justify="center"
        )
        self.status_label.pack(pady=20)

        self.countdown_label = tk.Label(
            root,
            text="",
            font=("맑은 고딕", 36, "bold")
        )
        self.countdown_label.pack(pady=10)

        self.result_label = tk.Label(
            root,
            text="",
            font=("맑은 고딕", 15),
            wraplength=440,
            justify="center"
        )
        self.result_label.pack(pady=20)

        self.similarity_label = tk.Label(
            root,
            text="",
            font=("맑은 고딕", 11),
            wraplength=460,
            justify="left"
        )
        self.similarity_label.pack(pady=10)

        self.start_button = tk.Button(
            root,
            text="소리 감지 시작",
            font=("맑은 고딕", 14, "bold"),
            width=18,
            height=2,
            command=self.start_listening
        )
        self.start_button.pack(pady=20)

        self.load_model()

    def load_model(self):
        """
        학습된 CNN 모델을 불러온다.
        """

        if not os.path.exists(MODEL_PATH):
            messagebox.showerror(
                "모델 파일 오류",
                f"모델 파일을 찾을 수 없습니다.\n\n현재 설정된 경로: {MODEL_PATH}\n\n"
                "학습된 모델 파일을 이 코드와 같은 위치 또는 프로젝트 폴더에 두고 MODEL_PATH를 수정하세요."
            )
            return

        try:
            self.model = tf.keras.models.load_model(MODEL_PATH)
            self.status_label.config(text="모델 로드 완료. 소리 감지를 시작할 수 있습니다.")
        except Exception as e:
            messagebox.showerror("모델 로드 실패", str(e))

    def start_listening(self):
        """
        버튼을 누르면 별도 스레드에서 녹음과 예측을 실행한다.
        GUI가 멈추지 않도록 threading 사용.
        """

        if self.model is None:
            messagebox.showerror("오류", "모델이 로드되지 않았습니다.")
            return

        self.start_button.config(state="disabled")
        self.root.configure(bg="white")
        self.status_label.config(
            text="소리를 듣고 있습니다...",
            bg="white"
        )
        self.result_label.config(text="", bg="white")
        self.similarity_label.config(text="", bg="white")
        self.countdown_label.config(text=str(DURATION), bg="white")

        thread = threading.Thread(target=self.listen_and_predict)
        thread.daemon = True
        thread.start()

    def listen_and_predict(self):
        """
        실제 마이크 녹음 → 예측 → 결과 화면 갱신을 수행한다.
        """

        try:
            # 카운트다운 화면 표시
            for remain in range(DURATION, 0, -1):
                self.root.after(0, self.update_countdown, remain)
                time.sleep(1)

            self.root.after(0, self.update_countdown, "분석 중...")

            # 3초 동안 마이크 입력
            audio = sd.rec(
                int(SAMPLE_RATE * DURATION),
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32"
            )
            sd.wait()

            audio = audio.flatten()

            # 모델 예측
            class_name, confidence, prediction = predict_sound(
                self.model,
                audio
            )

            # 결과 화면 출력
            self.root.after(
                0,
                self.show_result,
                class_name,
                confidence,
                prediction
            )

        except Exception as e:
            self.root.after(0, self.show_error, str(e))

    def update_countdown(self, text):
        self.countdown_label.config(text=str(text))

    def show_result(self, class_name, confidence, prediction):
        """
        예측 결과를 화면에 표시한다.
        """

        self.countdown_label.config(text="")

        if confidence < CONFIDENCE_THRESHOLD:
            bg_color = "#B7B7B7"
            title = "판단 보류"
            message = "소리가 명확하지 않습니다.\n다시 감지해 주세요."
        else:
            info = ALERT_INFO[class_name]
            bg_color = info["color"]
            title = info["title"]
            message = info["message"]

        self.root.configure(bg=bg_color)

        for widget in [
            self.title_label,
            self.status_label,
            self.countdown_label,
            self.result_label,
            self.similarity_label
        ]:
            widget.config(bg=bg_color)

        self.status_label.config(text="소리 감지 완료")

        self.result_label.config(
            text=(
                f"{title}\n\n"
                f"{message}\n\n"
                f"최종 유사도/신뢰도: {confidence * 100:.2f}%"
            )
        )

        similarity_text = "[클래스별 유사도]\n"
        for idx, prob in enumerate(prediction):
            similarity_text += f"- {CLASS_NAMES[idx]}: {prob * 100:.2f}%\n"

        self.similarity_label.config(text=similarity_text)

        self.start_button.config(state="normal")

    def show_error(self, error_message):
        self.countdown_label.config(text="")
        self.status_label.config(text="오류가 발생했습니다.")
        self.result_label.config(text=error_message)
        self.start_button.config(state="normal")


# =========================
# 5. 프로그램 실행
# =========================

if __name__ == "__main__":
    root = tk.Tk()
    app = SoundAlertApp(root)
    root.mainloop()
