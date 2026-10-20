"""H.264 video output through ffmpeg subprocesses (GPU NVENC when available)."""

import queue
import subprocess
import threading
from fractions import Fraction

import numpy as np
import torch
import torch.nn.functional as F

from exmr.preprocessing.video import h264_args

# NVENC's H.264 encoder rejects frames wider or taller than this
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
    """Encodes RGB uint8 (H, W, 3) frames (torch tensors, H and W even) into one or more videos,
    each the whole frame or a crop of it, with a single ffmpeg subprocess. A background thread
    feeds it, so write_frame returns immediately and encoding overlaps with rendering. Call
    close() at the end.

    Args:
        outputs: {output path: crop (x, y, w, h) of the frame, or None for the whole frame}
    """

    def __init__(self, outputs, fps=30, crf=23, queue_size=8):
        self.outputs, self.fps, self.crf = {str(p): c for p, c in outputs.items()}, fps, crf
        self._proc = None  # started with the first frame, which gives the frame size
        self._queue = queue.Queue(maxsize=queue_size)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while (frame := self._queue.get()) is not None:
            if self._proc is None:
                h, w = frame.shape
                self._proc = subprocess.Popen(
                    self._command(h * 2 // 3, w), stdin=subprocess.PIPE, bufsize=0
                )
            # write the array's buffer directly: .tobytes() would copy every frame
            self._proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8))
        if self._proc is not None:
            self._proc.stdin.close()
            self._proc.wait()

    def _command(self, h, w):
        """ffmpeg command: raw h x w YUV 4:2:0 frames from stdin, split into the outputs"""
        n = len(self.outputs)
        graph = [f"[0:v]split={n}" + "".join(f"[in{i}]" for i in range(n))]
        output_args = []
        for i, (path, crop) in enumerate(self.outputs.items()):
            x, y, cw, ch = crop or (0, 0, w, h)
            filters = ([f"crop={cw}:{ch}:{x}:{y}"] if crop else []) + _scale_filter(cw, ch)
            graph.append(f"[in{i}]{','.join(filters) or 'null'}[out{i}]")
            output_args += ["-map", f"[out{i}]", *map(str, h264_args(self.crf)), path]
        return [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{w}x{h}",
            "-r", str(Fraction(self.fps).limit_denominator(1000)), "-i", "pipe:0",
            "-filter_complex", ";".join(graph), *output_args,
        ]  # fmt: skip

    def write_frame(self, frame):
        # converted to YUV 4:2:0 (the videos' format) on the frame's device, e.g. the GPU: this
        # halves the bytes copied to the CPU and piped to ffmpeg, which limit 4K videos
        self._queue.put(_rgb_to_yuv420(frame).cpu().numpy())

    def close(self):
        self._queue.put(None)
        self._thread.join()
        if self._proc is not None and self._proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed to write {', '.join(self.outputs)}")


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
