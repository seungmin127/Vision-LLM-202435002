
# Non-Maximum Suppression (NMS)
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import skimage.data
from random import randint, gauss

def calculate_iou(bbox_a, bbox_b):
    x1 = max(bbox_a[0], bbox_b[0])
    y1 = max(bbox_a[1], bbox_b[1])
    x2 = min(bbox_a[2], bbox_b[2])
    y2 = min(bbox_a[3], bbox_b[3])

    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

    area_a = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1])
    area_b = (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1])

    union_area = area_a + area_b - intersection_area

    # TODO: ZeroDivisionError 주의
    if union_area == 0:
        return 0.0

    return intersection_area / union_area

def draw_bbox(img, coord1, coord2, color, thickness=3, txt=""):
    _, txt_h = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    cv2.rectangle(img, coord1, coord2, color, thickness)
    cv2.putText(img, txt, (coord1[0], coord1[1]-txt_h+15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


def random_bbox(img, min_w=100, min_h=400, max_w=600, max_h=600):
    img_h, img_w = img.shape[:2]

    w = randint(min_w, min(max_w, img_w))
    h = randint(min_h, min(max_h, img_h))

    x1 = randint(0, img_w - w)
    y1 = randint(0, img_h - h)

    return x1, y1, x1 + w, y1 + h

def random_conf():
    return min(1.0, max(0.0, gauss(0.6, 0.1)))

hubble = skimage.data.hubble_deep_field()

N = 20

boxes = []
scores = []

for i in range(N):
    boxes.append(random_bbox(hubble))
    scores.append(random_conf())

for i, box in enumerate(boxes):
    draw_bbox(
        hubble,
        (box[0], box[1]),
        (box[2], box[3]),
        (randint(0,255), randint(0,255), randint(0,255)),
        txt=f"{scores[i]:.2f}"
    )

plt.figure(figsize=(10,8))
plt.imshow(hubble)
plt.title("1. Random BBox (20 Boxes)")
plt.axis("off")
plt.show()

# TODO: Confidence Thresholding 함수 정의
# TODO: Confidence Score가 특정 임계값보다 낮으면 해당 BBox 제거

def prune_low_conf(boxes, scores, conf_threshold=0.5):

    pruned_boxes = []
    pruned_scores = []

    for box, score in zip(boxes, scores):
        if score >= conf_threshold:
            pruned_boxes.append(box)
            pruned_scores.append(score)

    return pruned_boxes, pruned_scores

# TODO: NMS 함수 정의
# TODO: np.sort() 함수를 사용하면 1차원 배열을 정렬
# TODO: np.argsort() 함수를 사용하면 1차원 배열의 정렬된 인덱스를 반환

def apply_nms(boxes, scores, iou_threshold=0.5):

    nms_boxes = []
    nms_scores = []

    sorted_indices = np.argsort(scores)[::-1]

    while len(sorted_indices) > 0:
        current_idx = sorted_indices[0]

        nms_boxes.append(boxes[current_idx])
        nms_scores.append(scores[current_idx])

        remaining_idices = []

        for idx in sorted_indices[1:]:
            iou = calculate_iou(boxes[current_idx], boxes[idx])

            if iou <= iou_threshold:
                remaining_idices.append(idx)

        sorted_indices = np.array(remaining_idices, dtype=int)

    return nms_boxes, nms_scores

print(len(boxes), len(scores))

new_boxes, new_scores = prune_low_conf(boxes, scores)
new_boxes, new_scores = apply_nms(new_boxes, new_scores, iou_threshold=0.2)

print(len(new_boxes), len(new_scores))

hubble = skimage.data.hubble_deep_field()

for i, box in enumerate(new_boxes):
    draw_bbox(
        hubble,
        (box[0], box[1]),
        (box[2], box[3]),
        (randint(0,255), randint(0,255), randint(0,255)),
        txt=f"{new_scores[i]:.2f}"
    )

hubble_custom_nms = hubble.copy()

plt.figure(figsize=(10,8))
plt.imshow(hubble_custom_nms)
plt.title("2. Custom NMS Result")
plt.axis("off")
plt.show()

def apply_nms_cv2(boxes, scores, conf_threshold=0.5, iou_threshold=0.5):

    if len(boxes) == 0:
        return [], []

    # cv2.dnn.NMSBoxes()는 (x, y, width, height) 형식을 사용
    boxes_xywh = []

    for box in boxes:
        x1, y1, x2, y2 = box

        width = x2 - x1
        height = y2 - y1

        boxes_xywh.append([int(x1), int(y1), int(width), int(height)])

    # OpenCV NMS 적용하여 BBox 제거 후 남은 BBox의 인덱스
    keep_indices = cv2.dnn.NMSBoxes(
        bboxes=boxes_xywh,
        scores=scores,
        score_threshold=conf_threshold,
        nms_threshold=iou_threshold,
    )

    # 1차원 리스트로 변환
    keep_indices = np.array(keep_indices).reshape(-1).tolist()

    # 원래 xyxy 형식의 박스에서 선택
    nms_boxes = [boxes[index] for index in keep_indices]
    nms_scores = [scores[index] for index in keep_indices]

    return nms_boxes, nms_scores

hubble = skimage.data.hubble_deep_field()

for i, box in enumerate(boxes):
    draw_bbox(
        hubble,
        (box[0], box[1]),
        (box[2], box[3]),
        (randint(0,255), randint(0,255), randint(0,255)),
        txt=f"{scores[i]:.2f}"
    )

plt.figure(figsize=(10,8))
plt.imshow(hubble)
plt.title("3. Before OpenCV NMS")
plt.axis("off")
plt.show()

print(len(boxes), len(scores))

cv2_new_boxes, cv2_new_scores = apply_nms_cv2(
    boxes, scores, conf_threshold=0.5, iou_threshold=0.2
)

print(len(cv2_new_boxes), len(cv2_new_scores))

hubble_cv2_nms = skimage.data.hubble_deep_field()

for i, box in enumerate(cv2_new_boxes):
    draw_bbox(
        hubble_cv2_nms,
        (box[0], box[1]),
        (box[2], box[3]),
        (randint(0,255), randint(0,255), randint(0,255)),
        txt=f"{cv2_new_scores[i]:.2f}"
    )

plt.figure(figsize=(10,8))
plt.imshow(hubble_cv2_nms)
plt.title("4. After OpenCV NMS")
plt.axis("off")
plt.show()

plt.figure(figsize=(16,10))

plt.subplot(1,2,1), plt.imshow(hubble_custom_nms), plt.title("Custom NMS Result")
plt.subplot(1,2,2), plt.imshow(hubble_cv2_nms), plt.title("CV2 NMS Result")

plt.suptitle("5. Custom NMS vs OpenCV NMS", fontsize=16)

for ax in plt.gcf().axes:
    ax.axis("off")

plt.show()
