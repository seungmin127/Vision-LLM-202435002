import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from datetime import datetime
from pathlib import Path
from time import monotonic

project_folder = Path(__file__).resolve().parent
model_path = project_folder / "hand_landmarker.task"
save_folder = project_folder / "memos"
save_hold_seconds = 1.0
pen_color = (110, 45, 15)  # 메모 선/펜촉 커서: 진한 남색(BGR, RGB로는 #0F2D6E)

pipeline = (
    "v4l2src device=/dev/video0 ! "
    "image/jpeg,width=1280,height=720,framerate=30/1 ! "
    "jpegdec ! videoconvert ! video/x-raw,format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 "
    "max-size-bytes=0 max-size-time=0 ! "
    "appsink drop=true max-buffers=1 sync=false"
)

def create_kalman():
    "노트북 L의 상태 [x, y, vx, vy]와 측정 [x, y]를 사용."
    kalman = cv2.KalmanFilter(4, 2)
    kalman.transitionMatrix = np.array(
        [[1, 0, 1, 0],
         [0, 1, 0, 1],
         [0, 0, 1, 0],
         [0, 0, 0, 1]], dtype=np.float32
    )
    kalman.measurementMatrix = np.array(
        [[1, 0, 0, 0],
         [0, 1, 0, 0]], dtype=np.float32
    )
    kalman.processNoiseCov = np.array(
        [[0.01, 0, 0, 0],
         [0, 0.01, 0, 0],
         [0, 0, 0.05, 0],
         [0, 0, 0, 0.05]], dtype=np.float32
    )
    kalman.measurementNoiseCov = np.eye(2, dtype=np.float32)
    kalman.errorCovPost = np.eye(4, dtype=np.float32)

    return kalman


def preprocess_pen(image):
    "밝기 보정 → 필터링 → HSV 마스크 → 형태학 연산 순서"
    # LAB에서 밝기(L)만 보정.
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(image_lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l)
    enhanced_lab = cv2.merge((enhanced_l, a, b))

    # 색상 검출을 위해 LAB을 다시 BGR로 변환.
    enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    # Gaussian 필터로 작은 잡음을 부드럽게.
    blurred_image = cv2.GaussianBlur(enhanced_image, (5, 5), 0)

    # 필터 적용 이미지를 HSV 색공간으로 변환.
    hsv_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2HSV)

    # H: 색상, S: 색의 진하기(채도), V: 밝기
    # 진한 남색 펜의 HSV 시작 범위. 카메라와 조명에 맞게 미세 조정.
    lower_navy = np.array([105, 75, 15], dtype=np.uint8)
    upper_navy = np.array([130, 255, 200], dtype=np.uint8)

    mask = cv2.inRange(hsv_image, lower_navy, upper_navy)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    final_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel)

    return final_mask


def detect_pen(image, final_mask):
    "윤곽과 중심점을 구한 뒤, 펜촉 후보를 검출"
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = None
    center = None
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            center = (cX, cY)

    tip_candidates = find_pen_tip(image, largest_contour, center)
    return largest_contour, center, tip_candidates


def find_pen_tip(image, contour, center):
    "펜 양 끝의 후보 A/B는 실제로 사용할 끝은 손에 따라 선택."
    tip_candidates = {}
    if center is None:
        return tip_candidates

    cX, cY = center
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, 50, 150)

    # 몸통 윤곽에 직선을 맞춥니다. vx, vy는 연필의 길이 방향.
    vx, vy, x0, y0 = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01).flatten()

    # 각 윤곽점이 중심에서 길이 방향으로 얼마나 떨어져 있는지 계산.
    points = contour.reshape(-1, 2)
    body_position = (points[:, 0] - cX) * vx + (points[:, 1] - cY) * vy
    end_a = body_position.min()
    end_b = body_position.max()
    body_length = end_b - end_a

    # 모든 픽셀에 대해 길이 방향 거리와 옆 방향 거리를 계산.
    yy, xx = np.indices(gray_image.shape)
    along = (xx - cX) * vx + (yy - cY) * vy
    across = -(xx - cX) * vy + (yy - cY) * vx

    # 양 끝 바깥으로 몸통 길이의 35%, 중심선에서 옆으로 3픽셀까지 탐색.
    search_length = body_length * 0.35
    for label in ("A", "B"):
        if label == "A":
            end_region = (along < end_a) & (along >= end_a - search_length)
        else:
            end_region = (along > end_b) & (along <= end_b + search_length)

        search_region = end_region & (np.abs(across) <= 3)
        edge_y, edge_x = np.where(search_region & (edges > 0))

        if len(edge_x) > 0:
            # 몸통에서 바깥으로 가장 멀리 있는 경계점을 끝을 후보.
            positions = along[edge_y, edge_x]
            if label == "A":
                index = np.argmin(positions)
            else:
                index = np.argmax(positions)

            candidate = (int(edge_x[index]), int(edge_y[index]))
            tip_candidates[label] = candidate

    return tip_candidates


def select_pen_tip(tip_candidates, center, hand_side):
    "화면 기준 오른손은 왼쪽 끝, 왼손은 오른쪽 끝을 사용."
    if center is None:
        return None

    if hand_side == "Right":
        left_points = [point for point in tip_candidates.values() if point[0] < center[0]]
        if left_points:
            return min(left_points)
    elif hand_side == "Left":
        right_points = [point for point in tip_candidates.values() if point[0] > center[0]]
        if right_points:
            return max(right_points)

    return None


def calculate_angle(p1, p2, p3, width, height):
    # 화면의 가로·세로 비율을 반영. 추정 깊이 z는 사용하지 않음.
    vector1 = np.array([(p1.x - p2.x) * width, (p1.y - p2.y) * height])
    vector2 = np.array([(p3.x - p2.x) * width, (p3.y - p2.y) * height])
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)

    if magnitude1 == 0 or magnitude2 == 0:
        return None  # 각도를 구할 수 없으면 펼침/굽힘도 판정하지 않음.

    cosine = np.dot(vector1, vector2) / (magnitude1 * magnitude2)
    cosine = np.clip(cosine, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def detect_hand(hand_detector, image):
    "반전하지 않은 원본에서 손, 손가락 상태, 좌우 손 구분을 반환."
    height, width = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hand_detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    hand_side = None
    if len(result.hand_landmarks) == 1 and len(result.handedness) == 1:
        if result.handedness[0]:
            hand_side = result.handedness[0][0].category_name
    finger_tips = (4, 8, 12, 16, 20)
    # 관절 세 점의 각도로 손가락 펼침을 판정.
    finger_angle_points = ((1, 2, 3), (5, 6, 7), (9, 10, 11), (13, 14, 15), (17, 18, 19))
    angle_threshold = 160  # 엄지 판정
    finger_angle_threshold = 145  # 나머지 손가락은 각도와 뻗은 거리를 함께 확인.
    extension_ratio = 1.1  # 손끝이 중간 관절보다 손목에서 10% 이상 멀리.
    hand_finger_states = []

    for hand in result.hand_landmarks:
        finger_states = []
        for finger_index, (p1_idx, p2_idx, p3_idx) in enumerate(finger_angle_points):
            angle = calculate_angle(
                hand[p1_idx], hand[p2_idx], hand[p3_idx], width, height
            )

            is_open = None
            if angle is not None:
                if finger_index == 0:
                    is_open = bool(angle >= angle_threshold)
                else:
                    # 살짝 굽혀 들어 올린 손가락도 구분하기 위해 거리를 함께 확인.
                    finger_tip_landmark = hand[finger_tips[finger_index]]
                    wrist_point = np.array([hand[0].x * width, hand[0].y * height])
                    joint_point = np.array([hand[p2_idx].x * width, hand[p2_idx].y * height])
                    finger_tip_point = np.array([finger_tip_landmark.x * width, finger_tip_landmark.y * height])
                    joint_distance = np.linalg.norm(joint_point - wrist_point)
                    tip_distance = np.linalg.norm(finger_tip_point - wrist_point)
                    if joint_distance > 0:
                        is_open = bool(angle >= finger_angle_threshold and tip_distance > joint_distance * extension_ratio)

            finger_states.append(is_open)

        # 순서는 엄지, 검지, 중지, 약지, 소지. True=펼침, False=굽힘, None=판정 불가.
        hand_finger_states.append(finger_states)

    return result.hand_landmarks, hand_finger_states, hand_side


def choose_mode(hand_finger_states, tip):
    "손 한 개의 자세와 펜촉 유무를 받아 동작 이름만 결정."
    hand_count = len(hand_finger_states)
    # 먼저 펜 유무와 관계없이 손모양 자체의 모드를 결정.
    gesture_mode = "PAUSE"
    if hand_count == 1:
        states = hand_finger_states[0]

        # 펜을 내려놓고 손을 펼칠 수 있으므로 저장 요청은 펜 검출과 무관.
        if states == [True, True, True, True, True]:
            gesture_mode = "SAVE_EXIT"
        else:
            # 엄지는 펜을 잡아도 펴진 것으로 보일 수 있어 기본/지우기 판정에서 제외.
            # states[1:]은 검지, 중지, 약지, 소지 네 개입니다.
            if states[1:] == [False, False, False, True]:
                gesture_mode = "ERASE"
            elif states[1:] == [False, False, False, False]:
                gesture_mode = "DRAW"

    # 펜촉을 놓치면 그리기/지우기를 중단.
    mode = gesture_mode
    if tip is None and mode in ("DRAW", "ERASE"):
        mode = "PAUSE"

    return mode


def make_camera_view(image, contour, center, tip_candidates, hand_landmarks, tip, hand_side):
    "카메라 화면에 윤곽, 끝 후보와 손 관절을 표시."
    view = image.copy()
    height, width = image.shape[:2]
    if contour is not None:
        cv2.drawContours(view, [contour], -1, (200, 110, 45), 1)  # 윤곽선: 밝은 파랑
    if center is not None:
        cv2.circle(view, center, 3, (0, 255, 255), -1)
    for label, point in tip_candidates.items():
        cv2.circle(view, point, 3, (0, 0, 255), 1)
        cv2.putText(view, label, (point[0] - 5, point[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    for hand in hand_landmarks:
        points = [(int(p.x * width), int(p.y * height)) for p in hand]
        for connection in vision.HandLandmarksConnections.HAND_CONNECTIONS:
            cv2.line(view, points[connection.start], points[connection.end], (200, 110, 45), 1)
    if tip is not None:
        cv2.circle(view, tip, 3, pen_color, -1)
    label = "Hand: unknown"
    if hand_side == "Right":
        label = "Right hand -> left tip"
    elif hand_side == "Left":
        label = "Left hand -> right tip"
    cv2.putText(view, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 110, 45), 1)
    return view


def make_memo_view(camera_view, canvas, tip, mode, held_seconds, save_failed, eraser_radius):
    "왼쪽은 카메라, 오른쪽은 메모입니다. 커서는 저장할 메모에 섞이지 않음."
    canvas_view = canvas.copy()
    if tip is not None:
        if mode == "DRAW":
            cv2.circle(canvas_view, tip, 4, pen_color, -1)
        elif mode == "ERASE":
            cv2.circle(canvas_view, tip, eraser_radius, (0, 0, 255), 1)
        else:
            cv2.circle(canvas_view, tip, 4, (150, 150, 150), 1)

    view = np.hstack((camera_view, canvas_view))
    text_x = canvas.shape[1] + 10
    cv2.putText(view, "Memo", (text_x, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(view, f"Mode: {mode}", (text_x, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    message = "Q / ESC: quit"
    if mode == "SAVE_EXIT":
        message = f"Hold open: {held_seconds:.1f} / {save_hold_seconds:.1f} s"
    if save_failed:
        message = "Save failed: lower hand and retry"
    cv2.putText(view, message, (text_x, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    return view


def save_memo(canvas, save_folder):
    "저장 폴더를 만들고, 안내 글씨와 커서가 없는 메모를 저장."
    save_path = save_folder / datetime.now().strftime("memo_%Y%m%d_%H%M%S_%f.png")
    try:
        save_folder.mkdir(parents=True, exist_ok=True)
        save_ok = cv2.imwrite(str(save_path), canvas)
    except (OSError, cv2.error):
        save_ok = False
    return save_ok, save_path


def main():
    # 1. 카메라, 모델과 프레임 사이에 유지할 상태를 준비.
    cap = None
    hand_detector = None
    canvas = None
    prev_point = None
    prev_mode = "PAUSE"
    pen_thickness = 4
    eraser_radius = 12
    save_start_time = None
    held_seconds = 0.0
    save_attempted = False
    save_failed = False
    kalman_ready = False
    prev_hand_side = None

    try:
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            print("카메라를 열지 못했습니다. /dev/video0 연결과 MJPEG 1280x720 30fps 지원을 확인하세요.")
            print("GStreamer 지원 OpenCV 환경으로 실행하세요.")
            raise SystemExit(1)
        kalman = create_kalman()
        base_option = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.HandLandmarkerOptions(base_options=base_option, num_hands=2)
        hand_detector = vision.HandLandmarker.create_from_options(options)
        cv2.namedWindow("Memo Board", cv2.WINDOW_NORMAL)

        while True:
            ret, image = cap.read()
            if not ret:
                print("카메라 프레임을 읽지 못했습니다. 연결 상태를 확인하세요.")
                break

            # 크기를 줄이되 가로세로 비율은 유지.
            scale = 400 / image.shape[1]
            image = cv2.resize(image, None, fx=scale, fy=scale)
            height, width = image.shape[:2]

            # 2. 펜과 손을 검출하고, 손에 따라 사용할 펜촉 방향을 선택.
            final_mask = preprocess_pen(image)
            contour, center, tip_candidates = detect_pen(image, final_mask)
            hand_landmarks, hand_finger_states, hand_side = detect_hand(hand_detector, image)
            raw_tip = select_pen_tip(tip_candidates, center, hand_side)

            # 좌우 손이 바뀌면 이전 손의 좌표와 선을 이어 쓰지 않음.
            if hand_side != prev_hand_side:
                kalman_ready = False
                prev_point = None
                prev_mode = "PAUSE"
            prev_hand_side = hand_side

            # 관측한 좌표(raw_tip)와 필터로 보정한 좌표(tip)를 구분.
            if raw_tip is not None:
                measurement = np.array([[raw_tip[0]], [raw_tip[1]]], dtype=np.float32)

                if not kalman_ready:
                    # 처음 또는 재검출 때는 현재 위치에서 시작.
                    initial_state = np.array([[raw_tip[0]], [raw_tip[1]], [0], [0]], dtype=np.float32)
                    kalman.statePre = initial_state.copy()
                    kalman.statePost = initial_state.copy()
                    kalman.errorCovPost = np.eye(4, dtype=np.float32)
                    corrected_state = initial_state
                    kalman_ready = True
                else:
                    predicted_state = kalman.predict()
                    corrected_state = kalman.correct(measurement)

                # 화면 밖 좌표가 나오지 않도록 범위를 제한.
                tip = (int(np.clip(corrected_state[0, 0], 0, image.shape[1] - 1)),
                       int(np.clip(corrected_state[1, 0], 0, image.shape[0] - 1)))
            else:
                # 관측이 없으면 예상 위치만으로 그리지 않음.
                tip = None
                kalman_ready = False

            if canvas is None:
                canvas = np.full((height, width, 3), 255, dtype=np.uint8)

            # 3. 한 손의 자세와 펜촉 유무로 현재 동작을 결정.
            mode = choose_mode(hand_finger_states, tip)

            # 손을 펼친 상태를 실제 시계로 1초 이상 유지해야 저장.
            if mode == "SAVE_EXIT":
                now = monotonic()
                if save_start_time is None:
                    save_start_time = now
                held_seconds = now - save_start_time
            else:
                save_start_time = None
                held_seconds = 0.0
                save_attempted = False
                save_failed = False
            save_ready = mode == "SAVE_EXIT" and held_seconds >= save_hold_seconds

            # 모드에 따라 원래 메모 공간(canvas)에 선을 그리거나 지움.
            if mode == "DRAW" and tip is not None:
                if prev_point is not None and prev_mode == "DRAW":
                    cv2.line(canvas, prev_point, tip, pen_color, pen_thickness, cv2.LINE_AA)
                else:
                    # 처음 검출했거나 중간에 끊겼다면 이전 점과 연결하지 않음.
                    cv2.circle(canvas, tip, pen_thickness // 2, pen_color, -1)
                prev_point = tip

            elif mode == "ERASE" and tip is not None:
                if prev_point is not None and prev_mode == "ERASE":
                    # 빠르게 이동해도 중간이 남지 않도록 흰색 선으로 연결.
                    cv2.line(canvas, prev_point, tip, (255, 255, 255), eraser_radius * 2)
                cv2.circle(canvas, tip, eraser_radius, (255, 255, 255), -1)
                prev_point = tip

            else:
                # 손/펜 미검출, 다른 자세, 저장 요청에서는 선을 끊음.
                prev_point = None

            prev_mode = mode

            # 4. 카메라와 메모를 한 창에 표시.
            camera_view = make_camera_view(
                image, contour, center, tip_candidates, hand_landmarks, tip, hand_side
            )
            memo_view = make_memo_view(
                camera_view, canvas, tip, mode, held_seconds, save_failed, eraser_radius
            )
            cv2.imshow("Memo Board", memo_view)

            # 5. 메모 저장에 성공하면 종료. 실패하면 자세를 바꾼 뒤 재시도.
            if save_ready and not save_attempted:
                save_attempted = True
                save_ok, save_path = save_memo(canvas, save_folder)
                if save_ok:
                    print("메모 저장 완료:", save_path)
                    break
                save_failed = True
                print("저장하지 못했습니다. 손을 내렸다가 다시 펼치면 재시도합니다.")

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            if cv2.getWindowProperty("Memo Board", cv2.WND_PROP_VISIBLE) < 1:
                break

    finally:
        if hand_detector is not None:
            hand_detector.close()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
