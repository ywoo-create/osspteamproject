# feat/alert_module.py
# 실시간 환경음 인식 기반 배리어프리 알림 시스템 - 알림 모듈
# CNN 모델의 예측 결과를 위험도별 시각 알림 정보로 변환한다.

import csv
from datetime import datetime
from pathlib import Path

import numpy as np


# =========================
# 1. 기본 설정
# =========================

CONFIDENCE_THRESHOLD = 0.70

BASE_DIR = Path(__file__).resolve().parent
HISTORY_PATH = BASE_DIR / "alert_history.csv"


CLASS_NAMES = [
    "0_indoor_alarms",
    "1_outdoor_warnings",
    "2_emergency_alarms"
]


ALERT_INFO = {
    "0_indoor_alarms": {
        "risk": 1,
        "title": "위험도 1: 실내 생활 알림음",
        "message": "실내 생활 알림음이 감지되었습니다. 전화벨, 초인종, 타이머 등을 확인하세요.",
        "color_name": "YELLOW",
        "bg_color": "#FFD966",
        "action": "화면 노란색 표시 및 생활 알림 아이콘 출력"
    },
    "1_outdoor_warnings": {
        "risk": 2,
        "title": "위험도 2: 실외 주의 소리",
        "message": "실외 주의 소리가 감지되었습니다. 자동차 경적이나 사이렌 등 주변 상황을 확인하세요.",
        "color_name": "ORANGE",
        "bg_color": "#F6B26B",
        "action": "화면 주황색 표시 및 주의 알림 출력"
    },
    "2_emergency_alarms": {
        "risk": 3,
        "title": "위험도 3: 응급/위험 경보음",
        "message": "응급 또는 위험 경보음이 감지되었습니다. 즉시 주변 상황을 확인하세요.",
        "color_name": "RED",
        "bg_color": "#E06666",
        "action": "화면 빨간색 강조 및 긴급 알림 출력"
    }
}


# =========================
# 2. 클래스 이름 정리
# =========================

def normalize_class_name(class_value):
    """
    모델이 반환한 클래스 값을 프로젝트 기준 클래스명으로 변환한다.

    입력 예시:
    0, "0", "0_indoor_alarms"

    출력 예시:
    "0_indoor_alarms"
    """

    class_value = str(class_value)

    if class_value.startswith("0"):
        return "0_indoor_alarms"

    if class_value.startswith("1"):
        return "1_outdoor_warnings"

    if class_value.startswith("2"):
        return "2_emergency_alarms"

    return class_value


# =========================
# 3. 예측 결과 해석
# =========================

def analyze_prediction(prediction):
    """
    CNN 모델의 softmax 예측 결과에서
    가장 높은 확률의 클래스 번호와 신뢰도를 구한다.

    prediction 예시:
    [[0.1, 0.8, 0.1]]
    또는
    [0.1, 0.8, 0.1]
    """

    prediction = np.array(prediction)

    # prediction이 [[...]] 형태이면 첫 번째 결과만 사용
    if prediction.ndim == 2:
        prediction = prediction[0]

    class_idx = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    return class_idx, confidence


# =========================
# 4. 알림 결과 생성
# =========================

def create_alert_result(class_value, confidence):
    """
    클래스 번호 또는 클래스명과 신뢰도를 받아
    사용자에게 보여줄 알림 정보를 생성한다.
    """

    class_name = normalize_class_name(class_value)
    confidence = float(confidence)

    # confidence가 86처럼 퍼센트로 들어오면 0.86으로 변환
    if confidence > 1:
        confidence = confidence / 100

    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 신뢰도가 기준값보다 낮으면 판단 보류
    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "created_at": created_at,
            "class_name": class_name,
            "risk": "판단 보류",
            "title": "판단 보류",
            "confidence": confidence,
            "color_name": "GRAY",
            "bg_color": "#B7B7B7",
            "message": "소리가 명확하지 않습니다. 다시 확인해 주세요.",
            "action": "추가 입력 또는 재감지 필요"
        }

    info = ALERT_INFO.get(class_name)

    # 등록되지 않은 클래스일 경우
    if info is None:
        return {
            "created_at": created_at,
            "class_name": class_name,
            "risk": "알 수 없음",
            "title": "알 수 없는 소리",
            "confidence": confidence,
            "color_name": "GRAY",
            "bg_color": "#B7B7B7",
            "message": "등록되지 않은 소리가 감지되었습니다.",
            "action": "분류 기준 확인 필요"
        }

    return {
        "created_at": created_at,
        "class_name": class_name,
        "risk": info["risk"],
        "title": info["title"],
        "confidence": confidence,
        "color_name": info["color_name"],
        "bg_color": info["bg_color"],
        "message": info["message"],
        "action": info["action"]
    }


# =========================
# 5. 콘솔 알림 출력
# =========================

def print_alert(alert_result):
    """
    알림 결과를 콘솔에 출력한다.
    """

    print("\n" + "=" * 60)
    print("[생활 소리 감지 결과]")
    print(f"시간: {alert_result['created_at']}")
    print(f"분류: {alert_result['title']}")
    print(f"클래스: {alert_result['class_name']}")
    print(f"신뢰도: {alert_result['confidence'] * 100:.2f}%")
    print(f"화면 색상: {alert_result['color_name']}")
    print(f"알림 메시지: {alert_result['message']}")
    print(f"조치: {alert_result['action']}")
    print("=" * 60)


# =========================
# 6. 팝업 알림 출력
# =========================

def show_alert_window(alert_result, auto_close_seconds=4):
    """
    Tkinter를 이용해 시각적 팝업 알림창을 출력한다.
    로컬 PC 환경에서 실행 가능하다.
    """

    try:
        import tkinter as tk

        root = tk.Tk()
        root.title("생활 소리 감지 알림")
        root.geometry("480x330")
        root.configure(bg=alert_result["bg_color"])
        root.attributes("-topmost", True)

        title_label = tk.Label(
            root,
            text=alert_result["title"],
            font=("맑은 고딕", 20, "bold"),
            bg=alert_result["bg_color"],
            fg="black",
            wraplength=420
        )
        title_label.pack(pady=25)

        message_label = tk.Label(
            root,
            text=alert_result["message"],
            font=("맑은 고딕", 14),
            bg=alert_result["bg_color"],
            fg="black",
            wraplength=420,
            justify="center"
        )
        message_label.pack(pady=10)

        confidence_label = tk.Label(
            root,
            text=f"신뢰도: {alert_result['confidence'] * 100:.2f}%",
            font=("맑은 고딕", 12),
            bg=alert_result["bg_color"],
            fg="black"
        )
        confidence_label.pack(pady=5)

        action_label = tk.Label(
            root,
            text=alert_result["action"],
            font=("맑은 고딕", 11),
            bg=alert_result["bg_color"],
            fg="black",
            wraplength=420
        )
        action_label.pack(pady=5)

        close_button = tk.Button(
            root,
            text="확인",
            font=("맑은 고딕", 12),
            command=root.destroy
        )
        close_button.pack(pady=15)

        root.after(auto_close_seconds * 1000, root.destroy)
        root.mainloop()

    except Exception as e:
        print("[팝업 알림 오류]", e)


# =========================
# 7. 경고음 출력
# =========================

def play_alert_sound(alert_result):
    """
    위험도에 따라 간단한 경고음을 출력한다.
    Windows 환경에서는 winsound 사용.
    """

    try:
        import winsound

        risk = alert_result["risk"]

        if risk == 3:
            winsound.Beep(1200, 700)
        elif risk == 2:
            winsound.Beep(900, 500)
        elif risk == 1:
            winsound.Beep(700, 300)

    except Exception:
        pass


# =========================
# 8. 알림 기록 저장
# =========================

def save_alert_history(alert_result, history_path=HISTORY_PATH):
    """
    감지 결과를 CSV 파일로 저장한다.
    """

    history_path = Path(history_path)
    history_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = history_path.exists()

    with open(history_path, "a", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "created_at",
                "class_name",
                "risk",
                "title",
                "confidence",
                "color",
                "message",
                "action"
            ])

        writer.writerow([
            alert_result["created_at"],
            alert_result["class_name"],
            alert_result["risk"],
            alert_result["title"],
            f"{alert_result['confidence'] * 100:.2f}",
            alert_result["color_name"],
            alert_result["message"],
            alert_result["action"]
        ])


# =========================
# 9. 모델 예측 + 알림 통합 함수
# =========================

def predict_and_alert(model, preprocessed_data, use_popup=False, use_sound=True, save_history=True):
    """
    CNN 모델과 Mel-Spectrogram 데이터를 입력받아
    예측, 위험도 판단, 알림 출력까지 한 번에 수행한다.

    Parameters
    ----------
    model:
        학습된 CNN 모델

    preprocessed_data:
        Mel-Spectrogram 데이터
        shape 예시:
        (64, 64, 1) 또는 (1, 64, 64, 1)

    use_popup:
        True이면 팝업 알림창 출력

    use_sound:
        True이면 위험도별 경고음 출력

    save_history:
        True이면 alert_history.csv에 기록 저장
    """

    input_data = np.array(preprocessed_data)

    # 단일 이미지인 경우 batch 차원 추가
    if input_data.ndim == 3:
        input_data = np.expand_dims(input_data, axis=0)

    prediction = model.predict(input_data, verbose=0)

    class_idx, confidence = analyze_prediction(prediction)

    alert_result = create_alert_result(class_idx, confidence)

    print_alert(alert_result)

    if use_sound:
        play_alert_sound(alert_result)

    if save_history:
        save_alert_history(alert_result)

    if use_popup:
        show_alert_window(alert_result)

    return alert_result


# =========================
# 10. 모델 없이 알림 모듈만 테스트
# =========================

if __name__ == "__main__":
    print("알림 모듈 단독 테스트를 시작합니다.")

    test_cases = [
        (0, 0.92),
        (1, 0.88),
        (2, 0.95),
        (1, 0.42)
    ]

    for class_idx, confidence in test_cases:
        result = create_alert_result(class_idx, confidence)
        print_alert(result)
        save_alert_history(result)
