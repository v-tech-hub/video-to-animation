import argparse,subprocess
import os
import sys
import time
import cv2
from PIL import Image
from tqdm import tqdm
import numpy as np
import queue
import threading
import onnxruntime as ort

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def parse_args():
    desc = "Tensorflow implementation of AnimeGANv3"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-i','--input_video_path', type=str, default='/home/ada/'+ 'v3-3.mp4', help='video file or number for webcam')
    parser.add_argument('-m','--model_path', type=str, default='models/AnimeGANv3_Hayao_36.onnx',  help='file path to save the modles')
    parser.add_argument('-o','--output', type=str, default='video/output/' ,help='output path')
    parser.add_argument('-t', '--IfConcat', type=str, default="None", choices=["None", "Horizontal", "Vertical"], help='Whether to splice the original video with the converted video')
    parser.add_argument('-d','--device', type=str, default='gpu', choices=["cpu","gpu","trt"] ,help='running device')
    parser.add_argument('--max-edge', type=int, default=1280, help='maximum inference image edge')
    return parser.parse_args()

def check_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

class Videocap:
    def __init__(self, video, model_name, limit=1280):
        self.model_name = model_name
        vid = cv2.VideoCapture(video)
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = vid.get(cv2.CAP_PROP_FPS)
        self.ori_width, self.ori_height = width, height
        if not vid.isOpened() or width <= 0 or height <= 0 or self.total <= 0:
            raise RuntimeError(f"Could not open video or read metadata: {video}")

        max_edge = max(width, height)
        # Prevent GPU memory from overflowing due to excessive input size.
        scale_factor = limit / max_edge if max_edge > limit else 1.
        height = int(round(height * scale_factor))
        width = int(round(width * scale_factor))
        self.width, self.height = self.to_8s(width), self.to_8s(height)
        print(f"[video] source={self.ori_width}x{self.ori_height} fps={self.fps:.3f} frames={self.total}", flush=True)
        print(f"[video] inference={self.width}x{self.height} (max edge limit={limit})", flush=True)

        self.count = 0  # Records the number of frames entered into the queue.
        self.cap = vid
        self.ret, frame = self.cap.read()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.q = queue.Queue(maxsize=60)
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    def _reader(self):
        while True:
            # print("get me")
            self.ret, frame = self.cap.read()
            if not self.ret:
                break
            frame = np.asarray(self.process_frame(frame, self.width, self.height))
            self.q.put(frame)
            self.count+=1
        self.cap.release()

    def read(self):
        f = self.q.get()
        self.q.task_done()
        return f

    def to_8s(self, x):
        if 'tiny' in self.model_name :
            return 256 if x < 256 else x - x % 16
        else:
            return 256 if x < 256 else x - x % 8

    def process_frame(self, img, width, height):
        # OpenCV is substantially faster than constructing a PIL image per frame.
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = img / 127.5 - 1.0
        return np.expand_dims(img, axis=0)


class Cartoonizer():
    def __init__(self, arg):
        self.args = arg
        available = ort.get_available_providers()
        print(f"[ort] version={ort.__version__} device={ort.get_device()}", flush=True)
        print(f"[ort] available providers={available}", flush=True)
        if self.args.device == "gpu":
            if 'CUDAExecutionProvider' not in available:
                raise RuntimeError("GPU requested but CUDAExecutionProvider is unavailable; refusing CPU fallback.")
            requested = ['CUDAExecutionProvider']
        elif self.args.device == "trt":
            if 'TensorrtExecutionProvider' not in available:
                raise RuntimeError("TensorRT requested but TensorrtExecutionProvider is unavailable.")
            requested = ['TensorrtExecutionProvider', 'CUDAExecutionProvider']
        else:
            requested = ['CPUExecutionProvider']
        print(f"[ort] requested providers={requested}", flush=True)
        t0 = time.perf_counter()
        self.sess_land = ort.InferenceSession(self.args.model_path, providers=requested)
        print(f"[ort] active providers={self.sess_land.get_providers()}", flush=True)
        print(f"[ort] session init={time.perf_counter()-t0:.2f}s", flush=True)
        if self.args.device == "gpu" and self.sess_land.get_providers()[0] != 'CUDAExecutionProvider':
            raise RuntimeError(f"CUDA session was not activated: {self.sess_land.get_providers()}")
        self.name = os.path.basename(self.args.model_path).rsplit('.',1)[0]


    def post_precess(self, img, wh):
        img = (img.squeeze() + 1.) / 2 * 255
        img = img.clip(0, 255).astype(np.uint8)
        img = cv2.resize(img, (wh[0], wh[1]), interpolation=cv2.INTER_LINEAR)
        return img

    def __call__(self):
        # load video
        vid = Videocap(self.args.input_video_path, self.name, limit=self.args.max_edge)
        codec = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        num = vid.total
        ouput_video_path = os.path.join(self.args.output, os.path.basename(self.args.input_video_path).rsplit('.', 1)[0] + f'_{self.name}.mp4')
        ouput_videoSounds_path = os.path.join(self.args.output, os.path.basename(self.args.input_video_path).rsplit('.', 1)[0] + f'_{self.name}_sounds.mp4')

        if self.args.IfConcat == "Horizontal":
            output_size = (vid.ori_width * 2, vid.ori_height)
        elif self.args.IfConcat == "Vertical":
            output_size = (vid.ori_width, vid.ori_height * 2)
        else:
            output_size = (vid.ori_width, vid.ori_height)

        # Stream rendered BGR frames directly into ffmpeg. This removes the
        # expensive MJPEG intermediate file and the second H.264 encode pass.
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s:v", f"{output_size[0]}x{output_size[1]}",
            "-r", f"{vid.fps:.8f}", "-i", "-",
            "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            ouput_video_path
        ]
        print(f"[output] direct ffmpeg H.264 pipe={ouput_video_path}", flush=True)
        self.encoder = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        print("[render] starting frame inference...", flush=True)
        render_start = time.perf_counter()
        preprocess_wait_s = inference_s = postprocess_s = write_s = 0.0
        pbar = tqdm(total=vid.total, mininterval=1.0, file=sys.stdout)
        pbar.set_description(f"Running: {os.path.basename(ouput_video_path)}")
        num = vid.total
        input_name = self.sess_land.get_inputs()[0].name
        while num > 0:
            if vid.count < vid.total and vid.ret == False and vid.q.empty():
                pbar.close()
                if self.encoder.stdin:
                    self.encoder.stdin.close()
                self.encoder.kill()
                return "The video is broken, please upload the video again."
            t = time.perf_counter()
            frame = vid.read()
            preprocess_wait_s += time.perf_counter() - t

            t = time.perf_counter()
            fake_img = self.sess_land.run(None, {input_name: frame})[0]
            inference_s += time.perf_counter() - t

            t = time.perf_counter()
            fake_img = self.post_precess(fake_img, (vid.ori_width, vid.ori_height))
            if self.args.IfConcat == "Horizontal":
                fake_img = np.hstack((self.post_precess(frame, (vid.ori_width, vid.ori_height)), fake_img))
            elif self.args.IfConcat == "Vertical":
                fake_img = np.vstack((self.post_precess(frame, (vid.ori_width, vid.ori_height)), fake_img))
            bgr = np.ascontiguousarray(fake_img[:, :, ::-1])
            postprocess_s += time.perf_counter() - t

            t = time.perf_counter()
            try:
                self.encoder.stdin.write(bgr.tobytes())
            except BrokenPipeError:
                err = self.encoder.stderr.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"ffmpeg encoder pipe failed: {err.strip()}")
            write_s += time.perf_counter() - t
            pbar.update(1)
            num -= 1

        pbar.close()
        if self.encoder.stdin:
            self.encoder.stdin.close()
        encoder_stderr = self.encoder.stderr.read().decode("utf-8", errors="replace")
        encoder_rc = self.encoder.wait()
        if encoder_rc != 0:
            raise RuntimeError(f"ffmpeg H.264 encode failed: {encoder_stderr.strip()}")
        render_elapsed = time.perf_counter() - render_start
        print(f"[output] H.264 video={ouput_video_path}", flush=True)
        print(f"[render] frames={vid.total} elapsed={render_elapsed:.2f}s throughput={vid.total/render_elapsed:.2f} FPS", flush=True)
        measured = preprocess_wait_s + inference_s + postprocess_s + write_s
        print("[profile] stage breakdown:", flush=True)
        for label, value in [
            ("preprocess/wait", preprocess_wait_s),
            ("onnx inference", inference_s),
            ("postprocess", postprocess_s),
            ("ffmpeg pipe/write", write_s),
        ]:
            pct = (value / render_elapsed * 100.0) if render_elapsed else 0.0
            per_frame = (value / vid.total * 1000.0) if vid.total else 0.0
            print(f"[profile] {label:17s} {value:8.2f}s  {pct:5.1f}%  {per_frame:7.2f} ms/frame", flush=True)
        other = max(0.0, render_elapsed - measured)
        print(f"[profile] {'other/overlap':17s} {other:8.2f}s  {(other/render_elapsed*100.0 if render_elapsed else 0):5.1f}%", flush=True)

        # Probe first. Silent source videos are normal and should not produce an ffmpeg error.
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=index", "-of", "csv=p=0", self.args.input_video_path],
            capture_output=True, text=True
        )
        if probe.returncode != 0:
            print(f"[audio] ffprobe failed; keeping silent video: {probe.stderr.strip()}", flush=True)
            return ouput_video_path
        if not probe.stdout.strip():
            print("[audio] source has no audio stream; keeping silent video.", flush=True)
            return ouput_video_path

        print("[audio] audio stream detected; muxing without intermediate MP3...", flush=True)
        try:
            command = [
                "ffmpeg", "-loglevel", "error", "-i", ouput_video_path,
                "-i", self.args.input_video_path, "-y",
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "copy", "-c:a", "aac", "-shortest", ouput_videoSounds_path
            ]
            subprocess.check_call(command)
            print(f"[audio] final video={ouput_videoSounds_path}", flush=True)
            return ouput_videoSounds_path
        except Exception as exc:
            print(f"[audio] mux failed ({exc}); keeping silent video.", flush=True)
            return ouput_video_path


if __name__ == '__main__':
    arg = parse_args()
    check_folder(arg.output)
    func =Cartoonizer(arg)
    info = func()
    print(f'output video: {info}')