# Colab T4 baseline

This branch tracks upstream AnimeGANv3 and adds a minimal inference benchmark for Google Colab.

## 1. Install

```bash
!apt-get -qq update && apt-get -qq install -y ffmpeg
!pip install -q -r requirements-colab.txt
```

Restart the Colab runtime after installation if ONNX Runtime was already imported.

## 2. Verify T4 / CUDA provider

```bash
!nvidia-smi
!python - <<'PY'
import onnxruntime as ort
print("ORT device:", ort.get_device())
print("Providers:", ort.get_available_providers())
assert "CUDAExecutionProvider" in ort.get_available_providers(), "CUDA provider unavailable"
PY
```

## 3. Run the upstream AnimeGANv3 video baseline

Download/copy an AnimeGANv3 ONNX model into the runtime, then run:

```bash
!mkdir -p output
!python tools/video2anime.py \
  -i input.mp4 \
  -o output \
  -m /content/AnimeGANv3_Hayao_36.onnx \
  -d gpu
```

Upstream currently processes one frame per ONNX Runtime call and limits the longest input edge to 1280 pixels. This is intentionally left unchanged for the first benchmark.

## 4. Measure wall-clock throughput

```bash
!ffprobe -v error -select_streams v:0 \
  -show_entries stream=avg_frame_rate,nb_frames,width,height \
  -of default=noprint_wrappers=1 input.mp4

!time python tools/video2anime.py \
  -i input.mp4 \
  -o output \
  -m /content/AnimeGANv3_Hayao_36.onnx \
  -d gpu
```

Start with a 10-30 second clip. Record source resolution/FPS, frame count, total wall time, GPU utilization and VRAM. Once the baseline is known, optimize batching, FP16/TensorRT, decode/encode, and resolution independently.
