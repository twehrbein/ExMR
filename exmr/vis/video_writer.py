"""H.264 video output through ffmpeg subprocesses (GPU NVENC when available)."""

import queue
import subprocess
import tempfile
import threading
from fractions import Fraction

import torch
import torch.nn.functional as F

from exmr.preprocessing.video import h264_args

# frames wider or taller than this are scaled down: H.264 levels and players top out around 4K
MAX_SIZE = 4096
# RGB -> YUV, BT.601 limited range (ffmpeg's default for yuv420p)
RGB_TO_YUV = torch.tensor(
    [
        [0.256788, 0.504129, 0.097906],
        [-0.148223, -0.290993, 0.439216],
        [0.439216, -0.367788, -0.071427],
    ]
)


class VideoWriter:
    """Encodes RGB uint8 (H, W, 3) frames (torch tensors) into a video with an ffmpeg subprocess.
    A background thread feeds it, so write_frame returns immediately and encoding overlaps with
    rendering. Call close() at the end, or use it as a context manager.

    One video per writer (ffmpeg process) also keeps NVENC to one session per process: several
    NVENC sessions in one ffmpeg process crash intermittently when they are destroyed at exit
    (driver-side, seen with driver 610).

    Args:
        width, height: the frame size, both even (the video's YUV 4:2:0 format needs it)
    """

    def __init__(self, path, width, height, fps=30, crf=23, queue_size=8):
        self.path, self.width, self.height = str(path), width, height
        self._stderr = tempfile.TemporaryFile()  # ffmpeg's messages, shown if it fails
        self._proc = subprocess.Popen(
            self._command(fps, crf), stdin=subprocess.PIPE, stderr=self._stderr, bufsize=0
        )
        self._queue = queue.Queue(maxsize=queue_size)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        try:
            while (frame := self._queue.get()) is not None:
                self._proc.stdin.write(frame)  # the array's buffer, without a copy
        except OSError:  # ffmpeg exited (broken pipe): close() reports its error
            while self._queue.get() is not None:  # keep taking frames, so write_frame never blocks
                pass
        self._proc.stdin.close()
        self._proc.wait()

    def _command(self, fps, crf):
        """ffmpeg command: raw YUV 4:2:0 frames from stdin -> H.264 video"""
        scale = _scale_filter(self.width, self.height)
        return [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{self.width}x{self.height}",
            "-r", str(Fraction(fps).limit_denominator(1000)), "-i", "pipe:0",
            *(["-vf", *scale] if scale else []), *map(str, h264_args(crf)), self.path,
        ]  # fmt: skip

    def write_frame(self, frame):
        if frame.shape[:2] != (self.height, self.width):
            raise ValueError(f"frame {tuple(frame.shape)}, expected {self.height}x{self.width}")
        # converted to YUV 4:2:0 (the video's format) on the frame's device, e.g. the GPU: this
        # halves the bytes copied to the CPU and piped to ffmpeg, which limit 4K videos
        self._queue.put(_rgb_to_yuv420(frame).cpu().numpy())

    def close(self):
        self._queue.put(None)
        self._thread.join()
        self._stderr.seek(0)
        messages = self._stderr.read().decode(errors="replace").strip()
        self._stderr.close()
        if self._proc.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed (exit code {self._proc.returncode}) to write {self.path}:\n"
                f"{messages}"
            )
        if messages:
            print(messages)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _rgb_to_yuv420(rgb):
    """(H, W, 3) RGB uint8 -> (H * 3/2, W) uint8: the Y, U and V planes of YUV 4:2:0, with the
    colors of each 2x2 block averaged."""
    h, w, _ = rgb.shape
    rgb = rgb.permute(2, 0, 1).float()  # (3, H, W)
    M = RGB_TO_YUV.to(rgb.device)
    y = torch.einsum("ij,jhw->ihw", M[:1], rgb) + 16
    uv = torch.einsum("ij,jhw->ihw", M[1:], F.avg_pool2d(rgb, 2)) + 128
    planes = torch.cat([y.flatten(), uv.flatten()])
    return planes.round().clamp(0, 255).to(torch.uint8).view(h * 3 // 2, w)


def _scale_filter(w, h):
    """ffmpeg filter that scales w x h frames down to fit MAX_SIZE and to even sizes (yuv420p and
    libx264 need them), or none if they already do."""
    s = min(1.0, MAX_SIZE / max(w, h))
    size = 2 * round(w * s / 2), 2 * round(h * s / 2)
    return [] if size == (w, h) else [f"scale={size[0]}:{size[1]}"]
