import time

import cv2
import mediapipe as mp
import torch
from mediapipe.tasks import python
from mediapipe.tasks.python import vision as mp_vision
from ultralytics import YOLO

import settings


def load_models():
    # 영상 모델은 카메라 스레드 시작 때 한 번만 불러온다.
    if settings.YOLO_DEVICE != "cpu" and not torch.cuda.is_available():
        raise RuntimeError('YOLO CUDA를 사용할 수 없습니다. CUDA 환경을 확인하거나 '
                           'settings.py의 YOLO_DEVICE를 "cpu"로 바꾸세요.')
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    model = YOLO(settings.YOLO_MODEL)
    if list(model.names.values()) != list(settings.COCO):
        raise ValueError("YOLO 모델은 COCO 80종 모델이어야 합니다.")
    options = mp_vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=settings.HAND_MODEL),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    hand_detector = mp_vision.HandLandmarker.create_from_options(options)
    return model, hand_detector


def detect_objects(model, frame):
    # 전체 COCO 물체를 검출한다. 표시 기준과 학습 후보 기준은 기존 값이다.
    with torch.no_grad():
        results = model.predict(source=frame, conf=0.35, iou=0.45, classes=None,
                                device=settings.YOLO_DEVICE, imgsz=416, verbose=False)
    result = results[0]
    objects = []
    for box in result.boxes:
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        bbox = box.xyxy[0].cpu().tolist()
        objects.append({"class": result.names[class_id],
                        "confidence": round(confidence, 3), "bbox": bbox})
    return objects, result.plot()


def detect_hands(hand_detector, frame, display, time_ms):
    # 손 좌표를 픽셀로 바꾸고 기존처럼 관절 위치를 화면에 표시한다.
    height, width = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = hand_detector.detect_for_video(image, time_ms)
    hands = []
    for landmarks in result.hand_landmarks:
        hand = []
        for point in landmarks:
            x, y = point.x * width, point.y * height
            hand.append((x, y))
            cv2.circle(display, (int(x), int(y)), 3, (0, 80, 255), -1)
        hands.append(hand)
    return hands


def select_target(objects, hands):
    # 사람·식탁을 제외하고 신뢰도 0.50 이상인 물체만 학습 후보로 삼는다.
    candidates = []
    for obj in objects:
        if obj["class"] not in ["person", "dining table"] and obj["confidence"] >= 0.50:
            candidates.append(obj)
    if len(candidates) == 1:
        return candidates[0]

    selected = None
    closest = 32 * 32 + 1
    for obj in candidates:
        x1, y1, x2, y2 = obj["bbox"]
        distance = float("inf")
        for hand in hands:
            # 엄지, 검지, 중지, 약지, 새끼손가락 끝을 사용한다.
            for index in [4, 8, 12, 16, 20]:
                x, y = hand[index]
                dx = max(x1 - x, x - x2, 0)
                dy = max(y1 - y, y - y2, 0)
                distance = min(distance, dx * dx + dy * dy)
        if distance > 32 * 32:
            continue
        if distance < closest:
            selected, closest = obj, distance
        elif distance == closest:
            selected = None
    return selected


def boxes_overlap(first, second):
    # 후보 위치가 멀리 뛰면 2초 확인을 다시 시작한다. 실물 식별은 아니다.
    return (max(first[0], second[0]) < min(first[2], second[2])
            and max(first[1], second[1]) < min(first[3], second[3]))


def update_target(shared, objects, selected, frame_time):
    # 이 함수는 Lock 안에서 호출한다. 가려져도 저장된 대상은 없애지 않는다.
    target = shared["target"]
    shared["visible"] = False
    if target is not None:
        closest = float("inf")
        old = target["bbox"]
        for obj in objects:
            if obj["class"] == target["class"]:
                box = obj["bbox"]
                distance = ((box[0] + box[2] - old[0] - old[2]) ** 2
                            + (box[1] + box[3] - old[1] - old[3]) ** 2)
                if distance < closest:
                    closest, visible_object = distance, obj
        if closest != float("inf"):
            # 같은 종류는 이전 위치와 가장 가까운 검출만 갱신한다.
            target.update(visible_object)
            shared["visible"] = True

    shared["selected"] = selected
    if selected is None or (target is not None and selected["class"] == target["class"]):
        shared["candidate"] = None
        return
    candidate = shared["candidate"]
    if (candidate is None or candidate["class"] != selected["class"]
            or not boxes_overlap(candidate["bbox"], selected["bbox"])
            or frame_time - shared["candidate_seen"] > 0.7):
        shared["candidate_since"] = frame_time
    shared["candidate"] = selected
    shared["candidate_seen"] = frame_time
    if frame_time - shared["candidate_since"] >= 2.0:
        # 새 대상일 때만 새 사전을 만든다. 이 차이로 이전 응답을 구분한다.
        shared["target"] = dict(selected)
        shared["visible"] = True
        shared["candidate"] = None
        shared["status"] = "OBJECT_SELECTED"
    # 같은 종류의 여러 실물을 자동으로 구별하지 않는다.
    # 같은 종류에서 새 대화를 원하면 '새 물체야' 명령을 사용한다.


def run_camera(shared):
    # 영상 처리만 반복한다. 화면 출력과 음성 대화는 메인에서 처리한다.
    camera = hand_detector = model = None
    try:
        model, hand_detector = load_models()
        camera = cv2.VideoCapture(settings.CAMERA_PIPELINE, cv2.CAP_GSTREAMER)
        with shared["lock"]:
            shared["camera_ready"] = True
        last_ms = 0
        while not shared["stop"].is_set():
            start = time.monotonic()
            ok, frame = camera.read()
            if not ok:
                with shared["lock"]:
                    shared["visible"] = False
                    shared["candidate"] = shared["selected"] = None
                    shared["error"] = "카메라 입력 실패. 다시 연결합니다."
                print("카메라 입력 실패. 다시 연결합니다.")
                camera.release()
                if shared["stop"].wait(1):
                    break
                camera = cv2.VideoCapture(settings.CAMERA_PIPELINE, cv2.CAP_GSTREAMER)
                continue
            objects, display = detect_objects(model, frame)
            last_ms = max(last_ms + 1, int(start * 1000))
            hands = detect_hands(hand_detector, frame, display, last_ms)
            selected = select_target(objects, hands)
            with shared["lock"]:
                shared["frame"], shared["frame_time"] = display, start
                if time.monotonic() - start <= 2.0:
                    update_target(shared, objects, selected, start)
                else:
                    # 처리가 오래 걸린 영상으로 새 대상을 선택하지 않는다.
                    shared["visible"] = False
                    shared["candidate"] = shared["selected"] = None
            shared["stop"].wait(max(0, 1 / 8 - (time.monotonic() - start)))
    except Exception as error:
        with shared["lock"]:
            shared["error"] = "영상 처리 오류: " + str(error)
        print("영상 처리 오류: " + str(error))
        shared["stop"].set()
    finally:
        if camera is not None:
            camera.release()
        if hand_detector is not None:
            hand_detector.close()
        # YOLO에 별도의 close API는 없으므로 모델 참조를 해제한다.
        del model
