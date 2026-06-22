from __future__ import annotations

import queue
import threading
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
import librosa
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
DURATION = 3.0
BLOCK_SECONDS = 0.1
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_SECONDS)
CHUNK_SAMPLES = int(SAMPLE_RATE * DURATION)
IMG_SIZE = 64

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


def create_model() -> tf.keras.Model:
    """getaudio.py와 동일한 CNN 구조를 생성합니다."""
    model = models.Sequential([
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

    return model


def extract_features(audio: np.ndarray) -> np.ndarray:
    """3초 음성을 64×64 Mel Spectrogram으로 변환합니다."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    audio = np.nan_to_num(audio)

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
) -> tuple[int, float, np.ndarray]:
    features = extract_features(audio)
    predictions = model.predict(features, verbose=0)[0]

    predicted_index = int(np.argmax(predictions))
    confidence = float(predictions[predicted_index]) * 100.0

    return predicted_index, confidence, predictions


class AlarmModuleWindow:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("실시간 소리 알림 모듈")
        self.root.geometry("760x560")
        self.root.minsize(700, 520)

        self.model: tf.keras.Model | None = None
        self.audio_stream: sd.InputStream | None = None
        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue()

        self.running_event = threading.Event()
        self.classification_thread: threading.Thread | None = None
        self.closed = False

        self.status_var = tk.StringVar(
            value="CNN 모델을 불러오는 중입니다."
        )
        self.result_var = tk.StringVar(value="-")
        self.confidence_var = tk.StringVar(value="-")
        self.buffer_var = tk.StringVar(value="오디오 버퍼 수집률: 0%")
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
                "창이 열리면 마이크 입력을 자동으로 시작하고\n"
                "3초마다 CNN 모델로 소리 유형을 분류합니다."
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

        result_grid = ttk.Frame(result_frame)
        result_grid.pack(fill="x")

        ttk.Label(
            result_grid,
            text="예측 소리:",
            width=16,
            font=("맑은 고딕", 11),
        ).grid(row=0, column=0, sticky="w", pady=5)

        ttk.Label(
            result_grid,
            textvariable=self.result_var,
            font=("맑은 고딕", 15, "bold"),
        ).grid(row=0, column=1, sticky="w", pady=5)

        ttk.Label(
            result_grid,
            text="신뢰도:",
            width=16,
            font=("맑은 고딕", 11),
        ).grid(row=1, column=0, sticky="w", pady=5)

        ttk.Label(
            result_grid,
            textvariable=self.confidence_var,
            font=("맑은 고딕", 13, "bold"),
        ).grid(row=1, column=1, sticky="w", pady=5)

        ttk.Label(
            result_frame,
            textvariable=self.detail_var,
            font=("맑은 고딕", 10),
            anchor="center",
        ).pack(fill="x", pady=(9, 0))

        probability_frame = ttk.LabelFrame(
            outer,
            text="클래스별 예측 확률",
            padding=16,
        )
        probability_frame.pack(
            fill="both",
            expand=True,
        )

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
                pady=8,
            )

            progress = ttk.Progressbar(
                probability_frame,
                maximum=100,
                variable=self.probability_vars[row],
            )
            progress.grid(
                row=row,
                column=1,
                sticky="ew",
                pady=8,
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
                pady=8,
            )

        probability_frame.columnconfigure(1, weight=1)

        ttk.Label(
            outer,
            text="창을 닫으면 마이크 입력과 분류가 종료됩니다.",
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
            "모델 준비 완료 — 마이크 입력을 자동으로 시작합니다."
        )
        self.root.after(300, self.start_listening)

    def show_model_error(self, detail: str) -> None:
        self.status_var.set("모델을 불러오지 못했습니다.")

        messagebox.showerror(
            "모델 로딩 오류",
            detail,
        )

    def start_listening(self) -> None:
        if self.model is None or self.running_event.is_set():
            return

        try:
            self.clear_audio_queue()
            self.buffer_progress["value"] = 0
            self.buffer_var.set("오디오 버퍼 수집률: 0%")

            self.running_event.set()

            self.audio_stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=BLOCK_SIZE,
                callback=self.audio_callback,
            )
            self.audio_stream.start()

            self.classification_thread = threading.Thread(
                target=self.classification_loop,
                daemon=True,
            )
            self.classification_thread.start()

            self.status_var.set("마이크 소리를 실시간으로 듣고 있습니다.")
            self.detail_var.set(
                "3초 분량이 모이면 자동으로 분류합니다."
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

    def clear_audio_queue(self) -> None:
        while True:
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

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
                    int(len(buffer) / CHUNK_SAMPLES * 100),
                    100,
                )

                self.safe_ui(
                    lambda value=percentage:
                    self.update_buffer_progress(value)
                )

                if len(buffer) >= CHUNK_SAMPLES:
                    segment = buffer[:CHUNK_SAMPLES]
                    buffer = buffer[CHUNK_SAMPLES:]

                    self.safe_ui(
                        lambda:
                        self.status_var.set("CNN 분류 처리 중입니다.")
                    )

                    predicted_index, confidence, predictions = classify(
                        self.model,
                        segment,
                    )

                    self.safe_ui(
                        lambda index=predicted_index,
                        conf=confidence,
                        preds=predictions.copy():
                        self.show_prediction(index, conf, preds)
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
        self.buffer_var.set(f"오디오 버퍼 수집률: {value}%")

    def show_prediction(
        self,
        predicted_index: int,
        confidence: float,
        predictions: np.ndarray,
    ) -> None:
        korean_name = CLASS_NAMES_KO[predicted_index]
        original_label = CLASS_LABELS[predicted_index]

        self.result_var.set(korean_name)
        self.confidence_var.set(f"{confidence:.1f}%")
        self.detail_var.set(
            f"{original_label} 클래스로 분류되었습니다."
        )

        for index, probability in enumerate(predictions):
            percentage = float(probability) * 100.0
            self.probability_vars[index].set(percentage)
            self.probability_text_vars[index].set(
                f"{percentage:.1f}%"
            )

        if self.running_event.is_set():
            self.status_var.set(
                "분류 완료 — 다음 3초 소리를 수집 중입니다."
            )

    def show_classification_error(self, detail: str) -> None:
        self.close_audio_stream()
        self.status_var.set("소리 분류 중 오류가 발생했습니다.")

        messagebox.showerror(
            "분류 오류",
            detail,
        )

    def on_close(self) -> None:
        self.closed = True
        self.running_event.clear()
        self.close_audio_stream()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    AlarmModuleWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
