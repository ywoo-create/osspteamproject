from __future__ import annotations

import queue
import threading
import traceback
from collections import deque
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
import librosa
import noisereduce as nr
import numpy as np
import sounddevice as sd
import tensorflow as tf
from tensorflow.keras import layers, models


# ============================================================
# 기본 설정
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = BASE_DIR / "my_audio_weights.weights.h5"

SAMPLE_RATE = 22_050
WINDOW_SECONDS = 3.0
HOP_SECONDS = 1.5
BLOCK_SECONDS = 0.1

BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_SECONDS)
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SECONDS)
HOP_SAMPLES = int(SAMPLE_RATE * HOP_SECONDS)

IMG_SIZE = 64
MIN_RMS = 0.003
CONFIDENCE_THRESHOLD = 0.60
SMOOTHING_COUNT = 3

CLASS_LABELS = [
    "0_indoor_alarms",
    "1_outdoor_warnings",
    "2_emergency_alarms",
]

CLASS_NAMES_KO = [
    "실내 알림음",
    "실외 경고음",
    "긴급 경보음",
]


# ============================================================
# CNN(6_18).ipynb와 동일한 모델 구조
# ============================================================
def create_model() -> tf.keras.Model:
    return models.Sequential([
        layers.Input(shape=(64, 64, 1)),
        layers.Rescaling(1.0 / 80.0, offset=1.0),

        layers.Conv2D(
            32,
            (3, 3),
            activation="relu",
            padding="same",
        ),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(
            64,
            (3, 3),
            activation="relu",
            padding="same",
        ),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(
            128,
            (3, 3),
            activation="relu",
            padding="same",
        ),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(3, activation="softmax"),
    ])


# ============================================================
# CNN(6_18).ipynb와 동일한 전처리
# 노이즈 제거 → 무음 제거 → 피크 정규화
# ============================================================
def reduce_noise(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    try:
        return nr.reduce_noise(
            y=y,
            sr=sr,
            stationary=False,
        )
    except Exception:
        return y


def trim_silence(
    y: np.ndarray,
    top_db: int = 30,
) -> np.ndarray:
    trimmed, _ = librosa.effects.trim(
        y,
        top_db=top_db,
    )

    if len(trimmed) == 0:
        return y

    return trimmed


def peak_normalize(
    y: np.ndarray,
    target_peak: float = 0.95,
) -> np.ndarray:
    max_value = float(np.max(np.abs(y))) if len(y) else 0.0

    if max_value > 0:
        y = y / max_value * target_peak

    return y.astype(np.float32)


def preprocess_audio(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    y = np.nan_to_num(y)

    y = reduce_noise(y, sr)
    y = trim_silence(y, top_db=30)
    y = peak_normalize(y, target_peak=0.95)

    return y


def extract_features(audio: np.ndarray) -> np.ndarray:
    """
    학습 코드와 같은 방식으로 Mel Spectrogram을 생성하고
    64×64 크기의 CNN 입력으로 변환합니다.
    """
    audio = preprocess_audio(audio, SAMPLE_RATE)

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max,
    )

    resized = cv2.resize(
        mel_db,
        (IMG_SIZE, IMG_SIZE),
        interpolation=cv2.INTER_AREA,
    )

    tensor = resized[np.newaxis, :, :, np.newaxis]
    return tensor.astype(np.float32)


def classify(
    model: tf.keras.Model,
    audio: np.ndarray,
) -> np.ndarray:
    features = extract_features(audio)
    return model.predict(features, verbose=0)[0]


# ============================================================
# 자동 실행 GUI
# ============================================================
class ImprovedAlarmWindow:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("실시간 소리 알림 모듈")
        self.root.geometry("780x620")
        self.root.minsize(720, 570)

        self.model: tf.keras.Model | None = None
        self.audio_stream: sd.InputStream | None = None
        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue()

        self.running_event = threading.Event()
        self.closed = False

        # 최근 예측 확률을 평균 내어 순간적인 오분류를 줄입니다.
        self.prediction_history: deque[np.ndarray] = deque(
            maxlen=SMOOTHING_COUNT
        )

        self.status_var = tk.StringVar(
            value="CNN 모델을 불러오는 중입니다."
        )
        self.result_var = tk.StringVar(value="-")
        self.confidence_var = tk.StringVar(value="-")
        self.buffer_var = tk.StringVar(
            value="오디오 버퍼 수집률: 0%"
        )
        self.rms_var = tk.StringVar(value="-")
        self.detail_var = tk.StringVar(
            value="모델 준비 후 자동으로 마이크 입력을 시작합니다."
        )

        self.probability_vars = [
            tk.DoubleVar(value=0.0)
            for _ in CLASS_LABELS
        ]
        self.probability_text_vars = [
            tk.StringVar(value="0.0%")
            for _ in CLASS_LABELS
        ]

        self.build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        threading.Thread(
            target=self.load_model_worker,
            daemon=True,
        ).start()

    def build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=24)
        outer.pack(fill="both", expand=True)

        ttk.Label(
            outer,
            text="실시간 소리 알림 모듈",
            font=("맑은 고딕", 22, "bold"),
        ).pack(pady=(0, 5))

        ttk.Label(
            outer,
            text=(
                "창이 열리면 자동으로 마이크를 듣습니다.\n"
                "3초 구간을 1.5초 간격으로 겹쳐 분석합니다."
            ),
            justify="center",
            font=("맑은 고딕", 11),
        ).pack(pady=(0, 18))

        status_frame = ttk.LabelFrame(
            outer,
            text="현재 상태",
            padding=14,
        )
        status_frame.pack(fill="x")

        ttk.Label(
            status_frame,
            textvariable=self.status_var,
            font=("맑은 고딕", 13, "bold"),
            anchor="center",
        ).pack(fill="x", pady=(0, 8))

        self.buffer_progress = ttk.Progressbar(
            status_frame,
            maximum=100,
            mode="determinate",
        )
        self.buffer_progress.pack(fill="x")

        ttk.Label(
            status_frame,
            textvariable=self.buffer_var,
            anchor="center",
        ).pack(fill="x", pady=(5, 0))

        result_frame = ttk.LabelFrame(
            outer,
            text="분류 결과",
            padding=16,
        )
        result_frame.pack(fill="x", pady=14)

        grid = ttk.Frame(result_frame)
        grid.pack(fill="x")

        items = [
            ("예측 소리:", self.result_var, 15),
            ("신뢰도:", self.confidence_var, 13),
            ("입력 음량(RMS):", self.rms_var, 11),
        ]

        for row, (label, variable, size) in enumerate(items):
            ttk.Label(
                grid,
                text=label,
                width=17,
                font=("맑은 고딕", 11),
            ).grid(
                row=row,
                column=0,
                sticky="w",
                pady=5,
            )

            ttk.Label(
                grid,
                textvariable=variable,
                font=("맑은 고딕", size, "bold"),
            ).grid(
                row=row,
                column=1,
                sticky="w",
                pady=5,
            )

        ttk.Label(
            result_frame,
            textvariable=self.detail_var,
            font=("맑은 고딕", 10),
            anchor="center",
        ).pack(fill="x", pady=(10, 0))

        probability_frame = ttk.LabelFrame(
            outer,
            text="최근 예측을 반영한 클래스별 확률",
            padding=16,
        )
        probability_frame.pack(fill="both", expand=True)

        for row, class_name in enumerate(CLASS_NAMES_KO):
            ttk.Label(
                probability_frame,
                text=class_name,
                width=15,
            ).grid(
                row=row,
                column=0,
                sticky="w",
                padx=(0, 10),
                pady=10,
            )

            ttk.Progressbar(
                probability_frame,
                maximum=100,
                variable=self.probability_vars[row],
            ).grid(
                row=row,
                column=1,
                sticky="ew",
                pady=10,
            )

            ttk.Label(
                probability_frame,
                textvariable=self.probability_text_vars[row],
                width=9,
                anchor="e",
            ).grid(
                row=row,
                column=2,
                sticky="e",
                padx=(10, 0),
                pady=10,
            )

        probability_frame.columnconfigure(1, weight=1)

        ttk.Label(
            outer,
            text=(
                "학습에 없던 일반 소리 또는 작은 소리는 "
                "'분류 보류'로 표시합니다."
            ),
            anchor="center",
            font=("맑은 고딕", 9),
        ).pack(fill="x", pady=(12, 0))

    def safe_ui(self, callback) -> None:
        if not self.closed:
            self.root.after(0, callback)

    def load_model_worker(self) -> None:
        try:
            if not WEIGHTS_PATH.exists():
                raise FileNotFoundError(
                    "가중치 파일을 찾을 수 없습니다.\n\n"
                    f"필요한 위치:\n{WEIGHTS_PATH}"
                )

            model = create_model()
            model.load_weights(str(WEIGHTS_PATH))
            self.model = model

            self.safe_ui(self.finish_model_loading)

        except Exception:
            detail = traceback.format_exc()
            self.safe_ui(
                lambda: self.show_model_error(detail)
            )

    def finish_model_loading(self) -> None:
        self.status_var.set(
            "모델 준비 완료 — 마이크 입력을 시작합니다."
        )
        self.root.after(300, self.start_listening)

    def show_model_error(self, detail: str) -> None:
        self.status_var.set("모델을 불러오지 못했습니다.")
        messagebox.showerror("모델 로딩 오류", detail)

    def start_listening(self) -> None:
        if self.model is None or self.running_event.is_set():
            return

        try:
            self.running_event.set()

            self.audio_stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=BLOCK_SIZE,
                callback=self.audio_callback,
            )
            self.audio_stream.start()

            threading.Thread(
                target=self.classification_loop,
                daemon=True,
            ).start()

            self.status_var.set(
                "마이크 소리를 실시간으로 듣고 있습니다."
            )
            self.detail_var.set(
                "첫 3초 분량이 모이면 분석을 시작합니다."
            )

        except Exception as error:
            self.running_event.clear()
            self.close_audio_stream()

            messagebox.showerror(
                "마이크 시작 오류",
                (
                    "마이크를 시작하지 못했습니다.\n\n"
                    f"{error}\n\n"
                    "Windows 마이크 권한과 기본 입력 장치를 "
                    "확인해 주세요."
                ),
            )

    def audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info,
        status,
    ) -> None:
        if status:
            print(status)

        if self.running_event.is_set():
            self.audio_queue.put(indata[:, 0].copy())

    def classification_loop(self) -> None:
        buffer = np.zeros(0, dtype=np.float32)

        while self.running_event.is_set():
            try:
                chunk = self.audio_queue.get(timeout=0.5)
                buffer = np.concatenate([buffer, chunk])

                percentage = min(
                    int(len(buffer) / WINDOW_SAMPLES * 100),
                    100,
                )
                self.safe_ui(
                    lambda value=percentage:
                    self.update_buffer_progress(value)
                )

                if len(buffer) < WINDOW_SAMPLES:
                    continue

                # 가장 최근 3초를 분석합니다.
                segment = buffer[:WINDOW_SAMPLES]

                # 1.5초만 제거하여 다음 분석 구간과 절반이 겹치게 합니다.
                buffer = buffer[HOP_SAMPLES:]

                rms = float(
                    np.sqrt(np.mean(np.square(segment)))
                )

                self.safe_ui(
                    lambda value=rms:
                    self.rms_var.set(f"{value:.6f}")
                )

                if rms < MIN_RMS:
                    self.prediction_history.clear()
                    self.safe_ui(self.show_quiet_result)
                    continue

                self.safe_ui(
                    lambda:
                    self.status_var.set("CNN 분류 처리 중입니다.")
                )

                raw_predictions = classify(
                    self.model,
                    segment,
                )

                self.prediction_history.append(
                    raw_predictions.copy()
                )

                smoothed_predictions = np.mean(
                    np.stack(self.prediction_history),
                    axis=0,
                )

                self.safe_ui(
                    lambda preds=smoothed_predictions.copy(),
                    raw=raw_predictions.copy():
                    self.show_prediction(preds, raw)
                )

            except queue.Empty:
                continue
            except Exception:
                detail = traceback.format_exc()
                self.running_event.clear()
                self.safe_ui(
                    lambda:
                    self.show_classification_error(detail)
                )
                break

    def update_buffer_progress(self, value: int) -> None:
        self.buffer_progress["value"] = value
        self.buffer_var.set(
            f"오디오 버퍼 수집률: {value}%"
        )

    def show_quiet_result(self) -> None:
        self.status_var.set(
            "입력 소리가 너무 작아 분류를 보류합니다."
        )
        self.result_var.set("소리 없음")
        self.confidence_var.set("-")
        self.detail_var.set(
            "충분한 크기의 소리가 입력되면 자동으로 다시 분석합니다."
        )

        for index in range(len(CLASS_LABELS)):
            self.probability_vars[index].set(0.0)
            self.probability_text_vars[index].set("0.0%")

    def show_prediction(
        self,
        predictions: np.ndarray,
        raw_predictions: np.ndarray,
    ) -> None:
        index = int(np.argmax(predictions))
        confidence = float(predictions[index])

        for class_index, probability in enumerate(predictions):
            percentage = float(probability) * 100.0
            self.probability_vars[class_index].set(percentage)
            self.probability_text_vars[class_index].set(
                f"{percentage:.1f}%"
            )

        self.confidence_var.set(
            f"{confidence * 100:.1f}%"
        )

        if confidence < CONFIDENCE_THRESHOLD:
            self.result_var.set("분류 보류")
            self.detail_var.set(
                "세 클래스 중 확실한 결과가 없어 알림을 보류합니다."
            )
            self.status_var.set(
                "신뢰도가 낮습니다 — 다음 소리를 분석 중입니다."
            )
            return

        self.result_var.set(CLASS_NAMES_KO[index])
        self.detail_var.set(
            f"{CLASS_LABELS[index]} 클래스로 분류되었습니다."
        )
        self.status_var.set(
            "분류 완료 — 다음 겹침 구간을 수집 중입니다."
        )

    def show_classification_error(self, detail: str) -> None:
        self.close_audio_stream()
        self.status_var.set(
            "소리 분류 중 오류가 발생했습니다."
        )
        messagebox.showerror("분류 오류", detail)

    def close_audio_stream(self) -> None:
        if self.audio_stream is None:
            return

        try:
            self.audio_stream.stop()
        except Exception:
            pass

        try:
            self.audio_stream.close()
        except Exception:
            pass

        self.audio_stream = None

    def on_close(self) -> None:
        self.closed = True
        self.running_event.clear()
        self.close_audio_stream()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    ImprovedAlarmWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
