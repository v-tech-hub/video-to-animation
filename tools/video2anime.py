import argparse
import os
import queue
import subprocess
import sys
import threading
import time

import cv2
import numpy as np
import onnxruntime as ort
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def parse_args():
    parser = argparse.ArgumentParser(description="AnimeGANv3 video renderer")
    parser.add_argument("-i", "--input_video_path", required=True, help="input video")
    parser.add_argument("-m", "--model_path", default="models/AnimeGANv3_Hayao_36.onnx")
    parser.add_argument("-o", "--output", default="video/output/", help="output directory")
    parser.add_argument("-t", "--IfConcat", default="None", choices=["None", "Horizontal", "Vertical"])
    parser.add_argument("-d", "--device", default="gpu", choices=["cpu", "gpu", "trt"])
    parser.add_argument("--max-edge", type=int, default=960, help="maximum inference image edge")
    parser.add_argument("--style", default="strong", choices=["base", "strong"], help="post-process style")
    parser.add_argument("--output-max-edge", type=int, default=0, help="limit output edge; 0 keeps source size")
    parser.add_argument("--preset", default="ultrafast", help="ffmpeg libx264 preset")
    parser.add_argument("--crf", type=int, default=21, help="ffmpeg H.264 CRF")
    parser.add_argument("--max-frames", type=int, default=0, help="render only N frames; 0 renders all")
    return parser.parse_args()


def check_folder(path):
    os.makedirs(path, exist_ok=True)
    return path


def fit_size(width, height, limit, multiple=2):
    if not limit or max(width, height) <= limit:
        return width - width % multiple, height - height % multiple
    scale = limit / max(width, height)
    w = max(multiple, int(round(width * scale)) // multiple * multiple)
    h = max(multiple, int(round(height * scale)) // multiple * multiple)
    return w, h


class Videocap:
    def __init__(self, video, model_name, limit=960):
        self.model_name = model_name
        vid = cv2.VideoCapture(video)
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = vid.get(cv2.CAP_PROP_FPS)
        self.ori_width, self.ori_height = width, height
        if not vid.isOpened() or width <= 0 or height <= 0 or self.total <= 0 or self.fps <= 0:
            raise RuntimeError(f"Could not open video or read metadata: {video}")

        width, height = fit_size(width, height, limit, 16 if "tiny" in model_name else 8)
        self.width, self.height = width, height
        print(f"[video] source={self.ori_width}x{self.ori_height} fps={self.fps:.3f} frames={self.total}", flush=True)
        print(f"[video] inference={self.width}x{self.height} (max edge limit={limit})", flush=True)

        self.count = 0
        self.cap = vid
        self.done = False
        self.error = None
        self.q = queue.Queue(maxsize=8)
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                self.q.put(self.process_frame(frame, self.width, self.height))
                self.count += 1
        except Exception as exc:
            self.error = exc
        finally:
            self.done = True
            self.cap.release()

    def read(self):
        while True:
            if self.error:
                raise RuntimeError(f"Video reader failed: {self.error}")
            try:
                return self.q.get(timeout=1.0)
            except queue.Empty:
                if self.done:
                    return None

    @staticmethod
    def process_frame(img, width, height):
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = img / 127.5 - 1.0
        return np.expand_dims(img, axis=0)


def strong_cartoon_rgb(img):
    """Reduce photographic micro-texture and push flatter animation-like colors."""
    # Edge-preserving smoothing is intentionally cheap: one bilateral pass at inference size.
    img = cv2.bilateralFilter(img, 5, 24, 24)
    # Quantize luminance/chroma slightly; this creates flatter painted regions without hard posterization.
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = ((l.astype(np.uint16) // 12) * 12 + 6).clip(0, 255).astype(np.uint8)
    a = ((a.astype(np.uint16) // 10) * 10 + 5).clip(0, 255).astype(np.uint8)
    b = ((b.astype(np.uint16) // 10) * 10 + 5).clip(0, 255).astype(np.uint8)
    img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)
    # Mild saturation/contrast lift after flattening.
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv[..., 1] = np.clip(hsv[..., 1].astype(np.float32) * 1.12, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return img


class Cartoonizer:
    def __init__(self, arg):
        self.args = arg
        available = ort.get_available_providers()
        print(f"[ort] version={ort.__version__} device={ort.get_device()}", flush=True)
        print(f"[ort] available providers={available}", flush=True)
        if arg.device == "gpu":
            if "CUDAExecutionProvider" not in available:
                raise RuntimeError("GPU requested but CUDAExecutionProvider is unavailable; refusing CPU fallback.")
            requested = ["CUDAExecutionProvider"]
        elif arg.device == "trt":
            if "TensorrtExecutionProvider" not in available:
                raise RuntimeError("TensorRT requested but TensorrtExecutionProvider is unavailable.")
            requested = ["TensorrtExecutionProvider", "CUDAExecutionProvider"]
        else:
            requested = ["CPUExecutionProvider"]

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        print(f"[ort] requested providers={requested}", flush=True)
        t0 = time.perf_counter()
        self.session = ort.InferenceSession(arg.model_path, sess_options=options, providers=requested)
        print(f"[ort] active providers={self.session.get_providers()}", flush=True)
        print(f"[ort] session init={time.perf_counter() - t0:.2f}s", flush=True)
        if arg.device == "gpu" and self.session.get_providers()[0] != "CUDAExecutionProvider":
            raise RuntimeError(f"CUDA session was not activated: {self.session.get_providers()}")
        self.input_name = self.session.get_inputs()[0].name
        self.name = os.path.basename(arg.model_path).rsplit(".", 1)[0]

    def postprocess(self, output, output_size):
        img = (output.squeeze() + 1.0) * 127.5
        img = img.clip(0, 255).astype(np.uint8)
        if self.args.style == "strong":
            img = strong_cartoon_rgb(img)
        if (img.shape[1], img.shape[0]) != output_size:
            img = cv2.resize(img, output_size, interpolation=cv2.INTER_LINEAR)
        return img

    def __call__(self):
        vid = Videocap(self.args.input_video_path, self.name, limit=self.args.max_edge)
        base_size = fit_size(vid.ori_width, vid.ori_height, self.args.output_max_edge, 2)
        if self.args.IfConcat == "Horizontal":
            output_size = (base_size[0] * 2, base_size[1])
        elif self.args.IfConcat == "Vertical":
            output_size = (base_size[0], base_size[1] * 2)
        else:
            output_size = base_size

        suffix = f"{self.name}_{self.args.style}_{self.args.max_edge}"
        output_video = os.path.join(self.args.output, os.path.basename(self.args.input_video_path).rsplit(".", 1)[0] + f"_{suffix}.mp4")
        output_with_audio = output_video.rsplit(".", 1)[0] + "_sounds.mp4"

        ffmpeg_cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s:v", f"{output_size[0]}x{output_size[1]}",
            "-r", f"{vid.fps:.8f}", "-i", "-",
            "-an", "-c:v", "libx264", "-preset", self.args.preset, "-crf", str(self.args.crf),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", output_video,
        ]
        print(f"[style] mode={self.args.style}", flush=True)
        print(f"[output] size={output_size[0]}x{output_size[1]} preset={self.args.preset} crf={self.args.crf}", flush=True)
        print(f"[output] direct ffmpeg H.264 pipe={output_video}", flush=True)
        encoder = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

        target_frames = min(vid.total, self.args.max_frames) if self.args.max_frames > 0 else vid.total
        render_start = time.perf_counter()
        preprocess_wait_s = inference_s = postprocess_s = write_s = 0.0
        rendered = 0
        pbar = tqdm(total=target_frames, mininterval=1.0, file=sys.stdout)
        pbar.set_description(f"Running: {os.path.basename(output_video)}")

        try:
            while rendered < target_frames:
                t = time.perf_counter()
                frame = vid.read()
                preprocess_wait_s += time.perf_counter() - t
                if frame is None:
                    break

                t = time.perf_counter()
                fake = self.session.run(None, {self.input_name: frame})[0]
                inference_s += time.perf_counter() - t

                t = time.perf_counter()
                fake = self.postprocess(fake, base_size)
                if self.args.IfConcat != "None":
                    original = ((frame.squeeze() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
                    original = cv2.resize(original, base_size, interpolation=cv2.INTER_LINEAR)
                    fake = np.hstack((original, fake)) if self.args.IfConcat == "Horizontal" else np.vstack((original, fake))
                bgr = np.ascontiguousarray(fake[:, :, ::-1])
                postprocess_s += time.perf_counter() - t

                t = time.perf_counter()
                encoder.stdin.write(bgr.tobytes())
                write_s += time.perf_counter() - t
                rendered += 1
                pbar.update(1)
        except BrokenPipeError:
            err = encoder.stderr.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"ffmpeg encoder pipe failed: {err.strip()}")
        finally:
            pbar.close()
            if encoder.stdin:
                encoder.stdin.close()

        encoder_stderr = encoder.stderr.read().decode("utf-8", errors="replace")
        encoder_rc = encoder.wait()
        if encoder_rc != 0:
            raise RuntimeError(f"ffmpeg H.264 encode failed: {encoder_stderr.strip()}")
        if rendered == 0:
            raise RuntimeError("No frames were rendered.")

        elapsed = time.perf_counter() - render_start
        print(f"[render] frames={rendered} elapsed={elapsed:.2f}s throughput={rendered / elapsed:.2f} FPS", flush=True)
        measured = preprocess_wait_s + inference_s + postprocess_s + write_s
        for label, value in [
            ("preprocess/wait", preprocess_wait_s),
            ("onnx inference", inference_s),
            ("postprocess/style", postprocess_s),
            ("ffmpeg pipe/write", write_s),
        ]:
            print(f"[profile] {label:17s} {value:8.2f}s  {value / elapsed * 100:5.1f}%  {value / rendered * 1000:7.2f} ms/frame", flush=True)
        print(f"[profile] {'other/overlap':17s} {max(0.0, elapsed-measured):8.2f}s", flush=True)

        # Preview renders deliberately stay silent; avoid needless audio work.
        if self.args.max_frames > 0:
            return output_video

        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=index",
             "-of", "csv=p=0", self.args.input_video_path],
            capture_output=True, text=True,
        )
        if probe.returncode != 0 or not probe.stdout.strip():
            print("[audio] no readable source audio; keeping silent video.", flush=True)
            return output_video

        print("[audio] muxing source audio...", flush=True)
        command = [
            "ffmpeg", "-loglevel", "error", "-i", output_video, "-i", self.args.input_video_path, "-y",
            "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-shortest", output_with_audio,
        ]
        subprocess.check_call(command)
        print(f"[audio] final video={output_with_audio}", flush=True)
        return output_with_audio


if __name__ == "__main__":
    arg = parse_args()
    check_folder(arg.output)
    print(f"output video: {Cartoonizer(arg)()}")
