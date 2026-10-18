# TODO: 배운 기술(이미지 전처리, 특징 추출)을 활용하여 실시간으로 객체를 검출
import cv2
import numpy as np

# 초록 LAB lower/upper range
green_lower = np.array([30, 60, 90], dtype=np.uint8)
green_upper = np.array([230, 115, 180], dtype=np.uint8)

pipeline = (
    "v4l2src device=/dev/video0 ! "
    "image/jpeg,width=1280,height=720,framerate=30/1 ! "
    "jpegdec ! videoconvert ! video/x-raw,format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 "
    "max-size-bytes=0 max-size-time=0 ! "
    "appsink drop=true max-buffers=1 sync=false"
)

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

while True:
    ret, frame = cap.read()

    if not ret:
        break


# TODO: 각 프레임 이미지를 전처리
    ball_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

# TODO: Color Segmentation으로 Binary Mask 생성
    ball_mask = cv2.inRange(ball_hsv, green_lower, green_upper)

# TODO: Morphological Operation을 적용하여 객체 영역 추출
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    ball_mask_opened = cv2.morphologyEx(ball_mask, cv2.MORPH_OPEN, kernel)
    ball_mask_final = cv2.morphologyEx(ball_mask_opened, cv2.MORPH_OPEN, kernel)

# TODO: 객체의 중심과 반지름을 구해 프레임 위에 overlay
    contours, _ = cv2.findContours(
    ball_mask_final,          # binary mask
    cv2.RETR_EXTERNAL,       # 구멍 뚫린 객체의 바깥쪽 테두리만 검출
    cv2.CHAIN_APPROX_SIMPLE  # 직선 경로 위에 있는 픽셀 좌표 전체 대신 꼭짓점 좌표만 저장
    )          

    if len(contours) > 0:
    # 가장 큰 contour
        largest_contour = max(contours, key=cv2.contourArea)

    # 최소 외접원 중심 및 반지름
    (x, y), radius = cv2.minEnclosingCircle(largest_contour)
    center_circle = (int(x), int(y))
    radius = int(radius)

    cv2.circle(frame, center_circle, radius, (255,0,0), 2)
    cv2.circle(frame, center_circle, 5, (255,0,0),-1)

    cv2.imshow("VideoCapture with GStreamer", ball_mask_final)
    cv2.imshow("VideoCapture with GStreamer", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()