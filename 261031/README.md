# 261031 과제-GPU 가속 및 딥러닝 학습, IoU·NMS 구현

## 파일 구조
실습 순서인 GPU 가속 → MNIST → CIFAR-10 → IoU → NMS 순으로 구성.

- **gpu_acceleration.py**: NumPy와 CuPy로 잔상 효과 및 2차원 Convolution을 처리하고 CPU/GPU 처리 시간을 비교하는 코드.
- **mnist_gpu.py**: `main_gpu()` 함수에서 MNIST 데이터셋을 CUDA GPU로 학습하고 평가하는 코드.
- **cifar_10_gpu.py**: CIFAR-10 데이터셋으로 CNN을 학습하고 모델 저장 및 재로딩 후 예측을 확인하는 코드.
- **iou.py**: 빨간 사과와 초록 사과의 바운딩 박스를 구하고 IoU를 계산하는 코드.
- **nms.py**: 임의의 바운딩 박스와 신뢰도 점수에 직접 구현한 NMS 및 OpenCV NMS를 적용하여 비교하는 코드.

```text
Vision-LLM-202435002/
├─ 261031/
│  ├─ gpu_acceleration.py
│  ├─ mnist_gpu.py
│  ├─ cifar_10_gpu.py
│  ├─ iou.py
│  ├─ nms.py
│  ├─ CIFAR10_CNN.pth 
│  └─ README.md
└─ src/
   ├─ images/
   │  ├─ seagull.jpg
   │  ├─ two_apples1.png
   │  ├─ two_apples2.png
   │  └─ two_apples3.png
   ├─ datasets/               
   └─ models/
      └─ CIFAR10/               
         └─ CIFAR10_CNN.pth   
```

## 실행 방식

아래 명령은 위 파일 구조를 기준으로 저장소 최상위 폴더인 `Vision-LLM-202435002`에서 각각 실행.

### 1. gpu_acceleration.py — GPU 가속

필요한 라이브러리를 설치.

```bash
python -m pip install opencv-python numpy matplotlib
```

NVIDIA CUDA GPU와 호환 드라이버가 필요하며, CUDA 버전에 맞는 CuPy를 추가로 설치.

```bash
python -m pip install cupy-cuda12x
```

GPU 사용 가능 여부를 확인.

```bash
python -c "import cupy as cp; print('CuPy GPU count:', cp.cuda.runtime.getDeviceCount())"
```

저장소 최상위 폴더에서 아래 명령을 실행.

```bash
python 261031/gpu_acceleration.py
```

이미지 경로가 `src/images/seagull.jpg`로 지정 -> 저장소 최상위 폴더에서 실행.

### 2. mnist_gpu.py — MNIST GPU 학습

필요한 라이브러리인 NumPy, PyTorch, torchvision을 설치.

```bash
python -m pip install numpy
```

```bash
python -c "import torch; print('PyTorch CUDA:', torch.cuda.is_available())"
```

저장소 최상위 폴더에서 아래 명령을 실행.

```bash
python 261031/mnist_gpu.py
```

데이터셋 경로가 `src/datasets`로 지정 -> 저장소 최상위 폴더에서 실행.

### 3. cifar_10_gpu.py — CIFAR-10 GPU 학습 및 모델 저장

필요한 라이브러리인 NumPy, Matplotlib, PyTorch, torchvision, visualtorch를 설치.

```bash
python -m pip install numpy matplotlib visualtorch
```

```bash
python -c "import torch; print('PyTorch CUDA:', torch.cuda.is_available())"
```

저장소 최상위 폴더에서 모델 저장 폴더를 먼저 생성한 뒤 실행. 소스에 저장 폴더를 자동 생성하는 코드가 없으므로 학습 전에 준비.

```bash
python -c "from pathlib import Path; Path('src/models/CIFAR10').mkdir(parents=True, exist_ok=True)"
python 261031/cifar_10_gpu.py
```

데이터셋 경로가 `src/datasets`, 모델 저장 및 재로딩 경로가 `src/models/CIFAR10/CIFAR10_CNN.pth`로 지정 -> 저장소 최상위 폴더에서 실행. 데이터가 없으면 CIFAR-10을 다운로드하므로 인터넷 연결이 필요.

### 4. iou.py — IoU 계산

필요한 라이브러리를 설치. `scikit-image`는 현재 소스에서 `skimage.data`를 import하므로 함께 설치.

```bash
python -m pip install opencv-python numpy matplotlib scikit-image
```

저장소 최상위 폴더에서 아래 명령을 실행.

```bash
python 261031/iou.py
```

이미지 경로가 다음과 같이 지정 -> 세 이미지를 준비하고 저장소 최상위 폴더에서 실행.

- `src/images/two_apples1.png`
- `src/images/two_apples2.png`
- `src/images/two_apples3.png`

### 5. nms.py — NMS 구현 및 비교

필요한 라이브러리를 설치.

```bash
python -m pip install opencv-python numpy matplotlib scikit-image
```

저장소 최상위 폴더에서 아래 명령을 실행.

```bash
python 261031/nms.py
```

## 기타 참고 사항

### 작업 중 경험한 문제점
- 상대 경로: 소스가 있는 `261031` 폴더에서 실행하면 `src`를 그 폴더 아래에서 찾게 됨. 위 실행 명령처럼 저장소 최상위 폴더에서 실행해야 입력·데이터·모델 경로가 일치해야 함.

### AI 활용 및 본인 기여도
- AI 활용: 막히는 부분에 대한 선행 코드 확인 및 일부 작성, 오류 분석 및 README 초안 작성에 활용했습니다.
- 본인 기여: 해당 파트 실습 중 배운 코드를 활용하여 직접 코드를 작성 및 실행하고 결과를 확인했습니다. 또한 제출 코드와 폴더 구조를 정리했습니다.
