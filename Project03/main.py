import json
import os
import threading
import time

import cv2
import numpy as np

import conversation
import settings
import vision


def check_files():
    # 모델을 새로 받지 않고 기존 파일과 실행 권한만 확인한다.
    paths = [settings.YOLO_MODEL, settings.HAND_MODEL, settings.GEMMA_MODEL,
             settings.WHISPER_MODEL, settings.WHISPER_PROGRAM,
             settings.PIPER_MODEL, settings.PIPER_MODEL + ".json"]
    for path in paths:
        if not os.path.isfile(path):
            raise FileNotFoundError("파일 경로를 확인하세요: " + path)
    if not os.access(settings.WHISPER_PROGRAM, os.X_OK):
        raise PermissionError("Whisper 실행 파일의 실행 권한을 확인하세요.")


def show_camera(shared):
    # 이 함수는 메인에서만 호출한다. 화면에 쓸 정보만 Lock 안에서 가져온다.
    if shared["stop"].is_set() or settings.HEADLESS:
        return
    with shared["lock"]:
        frame = shared["frame"]
        frame_time = shared["frame_time"]
        label = "none"
        if shared["target"] is not None:
            label = shared["target"]["class"]
        status = shared["status"]
        error = shared["error"]
    if frame is not None:
        display = frame.copy()
        if time.monotonic() - frame_time > 2.0:
            cv2.putText(display, "STALE CAMERA", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        display = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(display, status + " | target: " + label, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    if error:
        cv2.putText(display, "ERROR: see console", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    window = "Gunggeumi - caregiver"
    cv2.imshow(window, display)
    key = cv2.waitKey(1) & 0xFF
    if key in [27, ord("q")] or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
        shared["stop"].set()


def save_record(records):
    # 복잡한 이벤트 기록 대신 실제 대화만 종료할 때 JSON으로 저장한다.
    path = settings.ROOT + "/sessions/" + time.strftime("%Y%m%d_%H%M%S") + ".json"
    with open(path + ".tmp", "w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False, indent=2)
    os.replace(path + ".tmp", path)
    print("대화 기록 저장: " + path)


def main():
    # 실행 폴더를 준비하고 두 작업이 공유할 최소 정보만 만든다.
    os.makedirs(settings.ROOT + "/runtime", exist_ok=True)
    os.makedirs(settings.ROOT + "/sessions", exist_ok=True)
    check_files()
    shared = {
        "lock": threading.Lock(), "stop": threading.Event(),
        "target": None, "visible": False, "selected": None,
        "candidate": None, "candidate_since": 0.0, "candidate_seen": 0.0,
        "frame": None, "frame_time": 0.0, "camera_ready": False,
        "status": "LOADING", "error": "",
    }
    records = []
    camera_thread = threading.Thread(target=vision.run_camera, args=(shared,))
    camera_thread.start()
    try:
        # CUDA나 영상 모델에 문제가 있으면 Gemma까지 불러오기 전에 알린다.
        while not shared["stop"].is_set():
            show_camera(shared)
            with shared["lock"]:
                ready = shared["camera_ready"]
            if ready:
                break
            shared["stop"].wait(0.03)
        if not shared["stop"].is_set():
            conversation.run_conversation(shared, records, show_camera)
    except KeyboardInterrupt:
        pass
    except Exception as error:
        print("궁그미 실행 오류: " + str(error))
    finally:
        # 영상 작업이 카메라와 MediaPipe를 닫을 때까지 기다린다.
        shared["stop"].set()
        camera_thread.join()
        if not settings.HEADLESS:
            cv2.destroyAllWindows()
        try:
            save_record(records)
        except OSError as error:
            print("대화 기록 저장 실패: " + str(error))


if __name__ == "__main__":
    main()
