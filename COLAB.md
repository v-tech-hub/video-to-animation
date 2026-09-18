# Google Colab GPU quick start

1. Open https://colab.research.google.com/
2. Runtime > Change runtime type > select a GPU.
3. Create cells below and run in order.

## Cell 1 — GPU
```python
!nvidia-smi
```

## Cell 2 — clone and install
```python
!git clone https://github.com/v-tech-hub/video-to-animation.git
%cd video-to-animation
!pip uninstall -y onnxruntime onnxruntime-gpu >/dev/null 2>&1
!pip install -q "numpy<2.4" "opencv-python-headless>=4.10" "onnxruntime-gpu>=1.20"
```

## Cell 3 — verify CUDA
```python
import onnxruntime as ort
print(ort.get_available_providers())
assert "CUDAExecutionProvider" in ort.get_available_providers()
```

## Cell 4 — upload source
```python
from google.colab import files
uploaded = files.upload()
INPUT = next(iter(uploaded))
print(INPUT)
```

## Cell 5 — baseline render
```python
!python football_cartoon.py "$INPUT" -o goal-anime.mp4 --temporal 0 --inference-width 960
```

## Cell 6 — temporal render
```python
!python football_cartoon.py "$INPUT" -o goal-anime-temporal.mp4 --temporal 0.18 --inference-width 960
```

## Cell 7 — download
```python
from google.colab import files
files.download("goal-anime-temporal.mp4")
```

The runner automatically prefers CUDA and prints frame count, percent, FPS and ETA.
