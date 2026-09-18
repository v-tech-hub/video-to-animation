#!/usr/bin/env python3
"""Football video -> temporally stabilized AnimeGANv2 (ONNX).

No generative video model: all motion/timing/camera come from source frames.
"""
from __future__ import annotations
import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def preprocess(frame: np.ndarray, max_width: int = 960) -> np.ndarray:
    h, w = frame.shape[:2]
    if max_width and w > max_width:
        scale = max_width / w
        w, h = int(w * scale), int(h * scale)
    rw = 256 if w < 256 else w - w % 32
    rh = 256 if h < 256 else h - h % 32
    x = cv2.resize(frame, (rw, rh))
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
    return x[None]


def stylize(session: ort.InferenceSession, frame: np.ndarray, max_width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    inp = session.get_inputs()[0].name
    y = session.run(None, {inp: preprocess(frame, max_width)})[0]
    y = np.squeeze(y)
    y = np.clip((y + 1.0) * 127.5, 0, 255).astype(np.uint8)
    y = cv2.resize(y, (w, h))
    return cv2.cvtColor(y, cv2.COLOR_RGB2BGR)


def temporal_blend(prev_src, src, prev_out, current, strength: float):
    if prev_src is None or strength <= 0:
        return current
    # Dense Farneback baseline. RAFT can replace this without changing renderer API.
    a = cv2.cvtColor(prev_src, cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    scale = min(1.0, 720.0 / max(src.shape[:2]))
    if scale < 1:
        a_small = cv2.resize(a, None, fx=scale, fy=scale)
        b_small = cv2.resize(b, None, fx=scale, fy=scale)
    else:
        a_small, b_small = a, b
    flow = cv2.calcOpticalFlowFarneback(a_small, b_small, None, .5, 3, 21, 3, 5, 1.2, 0)
    if scale < 1:
        flow = cv2.resize(flow, (src.shape[1], src.shape[0])) / scale
    hh, ww = a.shape
    gx, gy = np.meshgrid(np.arange(ww), np.arange(hh))
    # backward approximation; conservative blend avoids trails around ball/players
    mapx = (gx - flow[..., 0]).astype(np.float32)
    mapy = (gy - flow[..., 1]).astype(np.float32)
    warped = cv2.remap(prev_out, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    diff = cv2.absdiff(src, prev_src)
    motion = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    reliable = (motion < 35).astype(np.float32)[..., None]
    alpha = strength * reliable
    return np.clip(current * (1-alpha) + warped * alpha, 0, 255).astype(np.uint8)


def mux_audio(video_no_audio: Path, source: Path, output: Path):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        shutil.move(str(video_no_audio), str(output))
        print("WARNING: ffmpeg not found; output has no source audio")
        return
    subprocess.run([
        ffmpeg, "-y", "-i", str(video_no_audio), "-i", str(source),
        "-map", "0:v:0", "-map", "1:a?", "-c:v", "libx264", "-crf", "18",
        "-preset", "medium", "-c:a", "aac", "-shortest", str(output)
    ], check=True)


def run(args):
    import time
    model = Path(args.model)
    if not model.exists():
        raise SystemExit(f"Model not found: {model}")

    print("[1/8] Opening video...", flush=True)
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open: {args.input}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w, h = int(cap.get(3)), int(cap.get(4))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    print(f"[2/8] Video: {w}x{h} | {fps:.3f} FPS | {total or '?'} frames", flush=True)

    available = ort.get_available_providers()
    providers = [p for p in ("CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider") if p in available]
    print(f"[3/8] Available ONNX providers: {available}", flush=True)
    print(f"[4/8] Creating ONNX session with: {providers}", flush=True)
    t_session = time.perf_counter()
    session = ort.InferenceSession(str(model), providers=providers)
    print(f"[5/8] Session ready in {time.perf_counter()-t_session:.2f}s | active: {session.get_providers()}", flush=True)

    print("[6/8] Reading first frame...", flush=True)
    ok, first_frame = cap.read()
    if not ok:
        raise SystemExit("Could not read first video frame")
    prep = preprocess(first_frame, args.inference_width)
    print(f"[7/8] First frame ready | source={w}x{h} | inference={prep.shape[2]}x{prep.shape[1]}", flush=True)
    print("[8/8] First GPU inference...", flush=True)
    t_first = time.perf_counter()
    first_cur = stylize(session, first_frame, args.inference_width)
    print(f"[8/8] First inference done in {time.perf_counter()-t_first:.2f}s", flush=True)
    with tempfile.TemporaryDirectory() as td:
        raw = Path(td) / "stylized.mp4"
        writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        prev_src = prev_out = None
        count = 0
        started = time.perf_counter()
        frame = first_frame
        cur = first_cur
        while True:
            out = temporal_blend(prev_src, frame, prev_out, cur, args.temporal)
            writer.write(out)
            prev_src, prev_out = frame, out
            count += 1
            if count == 1 or count % 10 == 0 or (total and count == total):
                elapsed = max(time.perf_counter() - started, 1e-6)
                rate = count / elapsed
                pct = (100.0 * count / total) if total else 0.0
                eta = ((total - count) / rate) if total and rate else 0.0
                print(f"frame {count}/{total or '?'} | {pct:.1f}% | {rate:.2f} FPS | ETA {eta:.0f}s", flush=True)
            ok, frame = cap.read()
            if not ok:
                break
            cur = stylize(session, frame, args.inference_width)
        cap.release(); writer.release()
        mux_audio(raw, Path(args.input), Path(args.output))
    print(f"done: {args.output} ({count} frames)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("-o", "--output", default="football_anime.mp4")
    p.add_argument("--model", default="pb_and_onnx_model/Shinkai_53.onnx")
    p.add_argument("--temporal", type=float, default=.18, help="0 disables stabilization; recommended .10-.25")
    p.add_argument("--inference-width", type=int, default=960, help="AnimeGAN inference width; output remains source resolution")
    run(p.parse_args())


if __name__ == "__main__":
    main()
