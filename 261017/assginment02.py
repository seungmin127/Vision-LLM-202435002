import cv2
import numpy as np
import matplotlib.pyplot as plt

lower_basketball_lab = np.array([70, 130, 70])
upper_basketball_lab = np.array([180, 230, 185])

basketball = cv2.imread('../src/images/basketball_crop.jpg')

# TODO: LAB 색공간으로 변환
basketball_lab = cv2.cvtColor(basketball, cv2.COLOR_BGR2LAB)

# TODO: 각 채널 분리
l, a, b = cv2.split(basketball_lab)

# TODO: CLAHE 생성 (clipLimit은 3.0, tileGridSize는 8x8로 설정)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

# TODO: L 채널에 CLAHE 적용 후 강화된 이미지 생성
l_clahe_enhanced = clahe.apply(l)
basketball_enhanced_lab = cv2.merge((l_clahe_enhanced, a, b))

# TODO: 강화된 이미지로 binary mask 생성
mask_enhanced = cv2.inRange(basketball_enhanced_lab, lower_basketball_lab, upper_basketball_lab)

# TODO: 강화된 이미지를 RGB로 변환 후 마스크를 통해 이미지 내 관심 영역만 추출
basketball_enhanced_rgb = cv2.cvtColor(basketball_enhanced_lab, cv2.COLOR_LAB2RGB)

# TODO: Morphological operation을 적용하여 최종 binary mask 생성
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

ball_mask_opened2 = cv2.morphologyEx(mask_enhanced, cv2.MORPH_OPEN, kernel, iterations=2)

ball_mask_closed1 = cv2.morphologyEx(ball_mask_opened2, cv2.MORPH_CLOSE, kernel)
ball_mask_closed2 = cv2.morphologyEx(ball_mask_opened2, cv2.MORPH_CLOSE, kernel, iterations=2)
ball_mask_closed3 = cv2.morphologyEx(ball_mask_opened2, cv2.MORPH_CLOSE, kernel, iterations=3)

plt.figure(figsize=(16, 8))
plt.subplot(2,3,2), plt.imshow(ball_mask_opened2, cmap="gray"), plt.title("Opening x2")
plt.subplot(2,3,4), plt.imshow(ball_mask_closed1, cmap="gray"), plt.title("Opening x2 & Closing x1")
plt.subplot(2,3,5), plt.imshow(ball_mask_closed2, cmap="gray"), plt.title("Opening x2 & Closing x2")
plt.subplot(2,3,6), plt.imshow(ball_mask_closed3, cmap="gray"), plt.title("Opening x2 & Closing x3")

for ax in plt.gcf().axes:
    ax.axis("off")

plt.show()

# 최종 선택: Opening 2회 + Closing 1회
final_ball_mask = ball_mask_closed1

# TODO: Binary mask로 외곽선 검출
contours, _ = cv2.findContours(
    final_ball_mask,         # binary mask
    cv2.RETR_EXTERNAL,       # 구멍 뚫린 객체의 바깥쪽 테두리만 검출
    cv2.CHAIN_APPROX_SIMPLE  # 직선 경로 위에 있는 픽셀 좌표 전체 대신 꼭짓점 좌표만 저장
)

# TODO: 최종 농구공 영역 및 외곽선 선택
final_img_enhanced = basketball_enhanced_rgb.copy()

if len(contours) > 0:
    largest_contour = max(contours, key=cv2.contourArea)
    
    final_mask = np.zeros_like(final_ball_mask)
    
    cv2.drawContours(
        final_mask, 
        [largest_contour],
        -1,
        255, 
        thickness=cv2.FILLED
    )

    mask_color = np.zeros_like(final_img_enhanced)
    mask_color[:, :, 0] = final_mask  # RGB 이미지의 빨간 채널
    alpha = 0.15
    final_img_enhanced = cv2.addWeighted(final_img_enhanced, 1.0, mask_color, alpha, 0)  

    cv2.drawContours(
        final_img_enhanced, [largest_contour], -1, (255, 0, 0), 2)

plt.figure(figsize=(16, 5))

plt.subplot(1, 3, 1), plt.imshow(final_ball_mask, cmap="gray"), plt.title("Morphological Mask")
plt.subplot(1, 3, 2), plt.imshow(final_mask, cmap="gray"), plt.title("Selected Ball Mask")
plt.subplot(1, 3, 3), plt.imshow(final_img_enhanced), plt.title("Ball Region and Contour")

for ax in plt.gcf().axes:
    ax.axis("off")

plt.show()