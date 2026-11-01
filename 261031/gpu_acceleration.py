# GPU 가속 (잔상효과, Convolution2D)
import time
import cv2
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt

IMAGE_PATH = "../src/images/seagull.jpg"

TARGET_WIDTH = 1280
TARGET_HEIGHT = 720

TRAIL_LENGTH = 32

img_gray = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)

if img_gray is None:
    raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {IMAGE_PATH}")

img_gray = cv2.resize(
    img_gray,
    (TARGET_WIDTH, TARGET_HEIGHT),
    interpolation=cv2.INTER_AREA
)

input_np = img_gray.astype(np.float32) / 255.0

plt.imshow(input_np, cmap="gray")
plt.axis("off");

# TODO: NumPy를 사용하여 잔상 효과 함수 정의

def motion_trail_np(image, weights):
    """출력 픽셀을 하나씩 계산하는 CPU 버전."""
    image_height, image_width = image.shape
    output = np.zeros_like(image, dtype=np.float32)

    for output_y in range(image_height):
        for output_x in range(image_width):
            for shift in range(len(weights)):
                # 왼쪽 입력 픽셀을 현재 출력 위치에 누적
                input_x = output_x - shift

                if input_x >= 0:
                    output[output_y, output_x] += (
                        image[output_y, input_x] * weights[shift]
                    )

    return output

# TODO: CuPy를 사용하여 잔상 효과 함수 정의
def motion_trail_cp(image, weights):
    """각 이동 거리에서 이미지 영역 전체를 GPU로 계산하는 버전."""
    image_height, image_width = image.shape
    output = cp.zeros_like(image, dtype=cp.float32)

    for shift in range(min(len(weights), image_width)):
        input_region = image[:, :image_width - shift]
        output_region = output[:, shift:]

        output_region += input_region * weights[shift]

    return output

# 이동 거리가 멀어질수록 가중치를 작게 설정
weights_np = np.linspace(
    1.0,
    0.05,
    TRAIL_LENGTH,
    dtype=np.float32
)

# 전체 가중치의 합을 1로 정규화
weights_np /= np.sum(weights_np)

cpu_start = time.perf_counter()

output_np = motion_trail_np(
    image=input_np,
    weights=weights_np
)

cpu_end = time.perf_counter()

cpu_time = cpu_end - cpu_start

# TODO: NumPy 배열을 CuPy 배열로 변환

upload_start = time.perf_counter()

input_cp = cp.asarray(input_np)
weights_cp = cp.asarray(weights_np)

cp.cuda.Stream.null.synchronize()

upload_end = time.perf_counter()

upload_time = upload_end - upload_start

warmup_image = cp.zeros((64, 64), dtype=cp.float32)
warmup_weights = cp.asarray(weights_np[:4])

_ = motion_trail_cp(
    image=warmup_image,
    weights=warmup_weights
)

cp.cuda.Stream.null.synchronize()
cp.cuda.Stream.null.synchronize()

gpu_start = time.perf_counter()

output_cp = motion_trail_cp(
    image=input_cp,
    weights=weights_cp
)

cp.cuda.Stream.null.synchronize()

gpu_end = time.perf_counter()

gpu_time = gpu_end - gpu_start

# TODO: output_cp를 CPU 메모리로 복사

download_start = time.perf_counter()

output_cp_np = cp.asnumpy(output_cp)

download_end = time.perf_counter()

download_time = download_end - download_start

compute_speedup = cpu_time / gpu_time

gpu_total_time = upload_time + gpu_time + download_time
total_speedup = cpu_time / gpu_total_time

print("========== NumPy CPU ==========")
print(f"Processing time: {cpu_time:.4f} seconds")
print()

print("========== CuPy GPU ==========")
print(f"CPU to GPU: {upload_time:.4f} seconds")
print(f"GPU processing: {gpu_time:.4f} seconds")
print(f"GPU to CPU: {download_time:.4f} seconds")
print(f"Total GPU time: {gpu_total_time:.4f} seconds")
print()

print("========== Comparison ==========")
print(f"Compute-only speedup: {compute_speedup:.2f}x")
print(f"Including transfers: {total_speedup:.2f}x")
plt.figure(figsize=(18,12))

plt.subplot(1,3,1), plt.imshow(input_np, cmap="gray"), plt.title("Original Image")
plt.subplot(1,3,2), plt.imshow(output_np, cmap="gray"), plt.title("Motion Trail (CPU)")
plt.subplot(1,3,3), plt.imshow(output_cp_np, cmap="gray"), plt.title("Motion Trail (GPU)")

for ax in plt.gcf().axes:
    ax.axis("off")

plt.figure(figsize=(12,8))
plt.imshow(output_cp_np, cmap="gray")
plt.axis("off");

# Convolution 연산
plt.imshow(input_np, cmap="gray")
plt.axis("off");

# TODO: CuPy를 사용하여 2차원 convolution 함수 정의

def convolution2d_cp(image, kernel, bias=0.0, stride=1, padding=0):
    """
    커널 위치는 Python에서 반복하고,
    각 위치의 출력 배열 전체 연산은 GPU에서 처리한다.
    """
    kernel_height, kernel_width = kernel.shape

    padded_image = cp.pad(
        image,
        ((padding, padding), (padding, padding)),
        mode="constant",
        constant_values=0
    )

    output_height = (
        padded_image.shape[0] - kernel_height
    ) // stride + 1

    output_width = (
        padded_image.shape[1] - kernel_width
    ) // stride + 1

    output = cp.zeros(
        (output_height, output_width),
        dtype=cp.float32
    )

    for kernel_y in range(kernel_height):
        for kernel_x in range(kernel_width):
            # 모든 출력 위치에서 현재 커널 원소에 대응하는 픽셀 선택
            input_region = padded_image[
                kernel_y:kernel_y + output_height * stride:stride,
                kernel_x:kernel_x + output_width * stride:stride
            ]

            output += input_region * kernel[kernel_y, kernel_x]

    output += bias
    return output

# TODO: 기존에 정의한 convolution2d 함수를 사용하여 convolution을 CPU로 처리
# TODO: 변수 재사용 가능
# 세로 경계 검출 커널
dx_edge_filter = np.array(
    [
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ],
    dtype=np.float32
)

# 가로 경계 검출 커널
dy_edge_filter = np.array(
    [
        [-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1]
    ],
    dtype=np.float32
)

dx_bias = -0.2
dy_bias = -0.2

def relu(x):
    return np.maximum(0, x)

def convolution2d(image, kernel, bias=0.0, stride=1, padding=0):
    """
    2차원 이미지에 하나의 CNN 필터를 적용.

    Parameters
    ----------
    image : np.ndarray
        입력 이미지, shape = (height, width)
    kernel : np.ndarray
        CNN 필터, shape = (kernel_height, kernel_width)
    bias : float
        필터 출력에 더할 편향
    stride : int
        필터가 이동하는 간격
    padding : int
        입력 이미지 가장자리에 추가할 픽셀의 개수

    Returns
    -------
    output : np.ndarray
        편향까지 적용된 convolution 결과
    """

    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    # 이미지 가장자리에 zero padding 적용
    padded_image = np.pad(
        image,
        ((padding, padding), (padding, padding)),
        mode="constant",
        constant_values=0
    )

    # (H, W: 입력 이미지 높이 및 너비, P: Padding 픽셀 값, S: Stride 픽셀 값)
    # Convolution 출력 이미지 높이 = floor((H + 2P - K) / S) + 1
    output_height = (padded_image.shape[0] - kernel_height) // stride + 1

    # Convolution 출력 이미지 너비 = floor((W + 2P - K) / S) + 1
    output_width = (padded_image.shape[1] - kernel_width) // stride + 1

    # 계산한 크기로 출력 이미지 준비
    output = np.zeros((output_height, output_width), dtype=np.float32)

    # 커널을 움직이며 필터 적용
    for output_y in range(output_height):
        for output_x in range(output_width):
            # 커널을 적용할 이미지 영역의 좌측상단
            start_y = output_y * stride
            start_x = output_x * stride

            # 커널을 적용할 이미지 영역을 crop
            image_region = padded_image[
                start_y:start_y + kernel_height,
                start_x:start_x + kernel_width
            ]

            # 필터를 적용하여 가중합 계산
            weighted_sum = np.sum(image_region * kernel)

            # 편향 적용
            output[output_y, output_x] = weighted_sum + bias

    return output

cpu_start = time.perf_counter()

output_dx_np = convolution2d(
    image=input_np,
    kernel=dx_edge_filter,
    bias=dx_bias,
    stride=1,
    padding=1
)

output_dy_np = convolution2d(
    image=input_np,
    kernel=dy_edge_filter,
    bias=dy_bias,
    stride=1,
    padding=1
)

cpu_time = time.perf_counter() - cpu_start

relu_dx_np = relu(output_dx_np)
relu_dy_np = relu(output_dy_np)

# TODO: convolution을 GPU로 처리 (변환-워밍업-처리 파이프라인 구현)
upload_start = time.perf_counter()

input_cp = cp.asarray(input_np)
dx_filter_cp = cp.asarray(dx_edge_filter)
dy_filter_cp = cp.asarray(dy_edge_filter)

cp.cuda.Stream.null.synchronize()
upload_time = time.perf_counter() - upload_start


# 워밍업
warmup_image = cp.zeros((64, 64), dtype=cp.float32)

_ = convolution2d_cp(
    warmup_image, dx_filter_cp,
    bias=dx_bias, stride=1, padding=1
)

_ = convolution2d_cp(
    warmup_image, dy_filter_cp,
    bias=dy_bias, stride=1, padding=1
)

cp.cuda.Stream.null.synchronize()

# TODO: 변수 재사용 가능

gpu_start = time.perf_counter()

output_dx_cp = convolution2d_cp(
    image=input_cp,
    kernel=dx_filter_cp,
    bias=dx_bias,
    stride=1,
    padding=1
)

output_dy_cp = convolution2d_cp(
    image=input_cp,
    kernel=dy_filter_cp,
    bias=dy_bias,
    stride=1,
    padding=1
)

cp.cuda.Stream.null.synchronize()
gpu_time = time.perf_counter() - gpu_start

# GPU → CPU 복사
download_start = time.perf_counter()

output_dx_cp_np = cp.asnumpy(output_dx_cp)
output_dy_cp_np = cp.asnumpy(output_dy_cp)

download_time = time.perf_counter() - download_start


# 기존 relu()는 np.maximum을 사용하므로 CPU 배열에 적용
relu_dx_cp_np = relu(output_dx_cp_np)
relu_dy_cp_np = relu(output_dy_cp_np)

# TODO: CPU/GPU 결과 비교 및 출력
gpu_total_time = upload_time + gpu_time + download_time

print(f"CPU convolution: {cpu_time:.4f} seconds")
print(f"GPU convolution: {gpu_time:.4f} seconds")
print(f"CPU to GPU: {upload_time:.4f} seconds")
print(f"GPU to CPU: {download_time:.4f} seconds")
print(f"Total GPU time: {gpu_total_time:.4f} seconds")
print()

print(f"Compute-only speedup: {cpu_time / gpu_time:.2f}x")
print(f"Including transfers: {cpu_time / gpu_total_time:.2f}x")
print()

# 누적 순서에 따른 작은 실수 오차를 허용하여 비교
print(
    "dx 결과 일치:",
    np.allclose(output_dx_np, output_dx_cp_np, rtol=1e-5, atol=1e-6)
)
print(
    "dy 결과 일치:",
    np.allclose(output_dy_np, output_dy_cp_np, rtol=1e-5, atol=1e-6)
)

plt.figure(figsize=(18,8))

plt.subplot(2,3,1), plt.imshow(input_np, cmap="gray"), plt.title("Input Image")
plt.subplot(2,3,2), plt.imshow(relu_dx_np, cmap="gray"), plt.title("Vertical Convolution (CPU)")
plt.subplot(2,3,3), plt.imshow(relu_dx_cp_np, cmap="gray"), plt.title("Vertical Convolution (GPU)")

plt.subplot(2,3,4), plt.imshow(input_np, cmap="gray"), plt.title("Input Image")
plt.subplot(2,3,5), plt.imshow(relu_dy_np, cmap="gray"), plt.title("Horizontal Convolution (CPU)")
plt.subplot(2,3,6), plt.imshow(relu_dy_cp_np, cmap="gray"), plt.title("Horizontal Convolution (GPU)")

for ax in plt.gcf().axes:
    ax.axis("off")

plt.show()