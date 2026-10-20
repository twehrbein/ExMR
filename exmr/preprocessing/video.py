"""Video probing, frame extraction and re-encoding (ffmpeg / PyAV / TurboJPEG)."""

import functools
import re
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

import av
import cv2
import imageio.v3 as iio
from turbojpeg import TJPF_RGB, TurboJPEG


def _run_ffmpeg(args):
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-y", "-loglevel", "error", *map(str, args)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {' '.join(cmd)}\n{result.stderr}")


def extract_frames(video_path, output_dir, quality=2):
    """Write every frame to output_dir/%06d.jpg (from 000000.jpg); returns the frame count.

    quality: ffmpeg's -q:v for MJPEG, 2 (near-lossless) .. 31; lower is better.
    """
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)  # so that the returned count is exact
    output_dir.mkdir(parents=True)
    # -fps_mode passthrough: pass frames through as decoded (no duplication/dropping); same as
    # the older -vsync 0, which ffmpeg deprecated in 5.1 and newer versions reject
    _run_ffmpeg(
        ["-i", video_path, "-fps_mode", "passthrough", "-q:v", quality, "-start_number", 0,
         output_dir / "%06d.jpg"]
    )  # fmt: skip
    return sum(1 for p in output_dir.glob("*.jpg") if re.fullmatch(r"\d{6}", p.stem))


def frame_size(frame_dir):
    """(width, height) of the extracted frames."""
    img = cv2.imread(str(Path(frame_dir) / "000000.jpg"))
    if img is None:
        raise RuntimeError(f"Could not read the first frame in {frame_dir}")
    h, w = img.shape[:2]
    return w, h


def video_lwh(video_path):
    """(num_frames, width, height) from the container (PyAV via imageio)."""
    L, H, W, _ = iio.improps(video_path, plugin="pyav").shape
    return L, W, H


def video_fps(video_path):
    """Average frame rate (frames / duration); robust to VFR and missing rate metadata."""
    with av.open(str(video_path)) as c:
        s = c.streams.video[0]
        # libav's average_rate is frames / duration
        if s.average_rate and s.average_rate > 0:
            return float(s.average_rate)

        # fallback: demux (no decoding) to count frames and measure the timestamp span; pts can
        # be out of order (B-frames), so use min/max
        n, pts = 0, []
        for pkt in c.demux(s):
            if pkt.size == 0:  # flush packet
                continue
            n += 1
            if pkt.pts is not None:
                pts.append(pkt.pts)
        if n == 0:
            raise ValueError(f"no video frames in {video_path}")
        if len(pts) >= 2:
            span = (max(pts) - min(pts)) * float(s.time_base)
            if span > 0:
                return (n - 1) / span
        return float(s.base_rate or s.guessed_rate or 0.0)


@functools.lru_cache(maxsize=1)
def _turbojpeg():
    return TurboJPEG()


def read_frame(path):
    """JPEG file -> (H, W, 3) RGB uint8 (TurboJPEG; same pixels as cv2.imread + cvtColor)."""
    with open(path, "rb") as f:
        return _turbojpeg().decode(f.read(), pixel_format=TJPF_RGB)


@functools.lru_cache(maxsize=1)
def nvenc_available():
    """Whether ffmpeg can encode H.264 on the GPU (NVENC); checked with a tiny trial encode,
    since a compiled-in encoder can still fail at runtime (no GPU, old driver)."""
    listed = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True)
    if "h264_nvenc" not in listed.stdout:
        return False
    trial = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
         "-i", "color=c=black:s=256x256:d=0.1", "-c:v", "h264_nvenc", "-preset", "p4",
         "-f", "null", "-"],
        capture_output=True,
    )  # fmt: skip
    return trial.returncode == 0


def h264_args(crf):
    """ffmpeg output codec args: GPU NVENC if available, else CPU libx264, yuv420p.

    NVENC's constant-quality scale runs ~5 points "cheaper" than x264's CRF for the same file
    size, hence the offset.
    """
    if nvenc_available():
        codec = ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", crf + 5]
    else:
        codec = ["-c:v", "libx264", "-preset", "veryfast", "-crf", crf]
    return [*codec, "-pix_fmt", "yuv420p"]


def encode_frames(frame_dir, out_path, fps, crf=17):
    """frame_dir/%06d.jpg -> constant-frame-rate H.264 video with exactly that many frames.

    Used to rebuild variable-frame-rate or damaged videos, whose container frame count disagrees
    with the frames ffmpeg decodes, so that every reader sees the same frames.
    """
    fps = Fraction(fps).limit_denominator(1000)
    _run_ffmpeg(
        ["-framerate", fps, "-start_number", 0, "-i", Path(frame_dir) / "%06d.jpg",
         *h264_args(crf), out_path]
    )  # fmt: skip
    return out_path


def resample_fps(in_path, out_path, fps=30, crf=17):
    """Re-encode a video to a constant frame rate (drops/duplicates frames)."""
    _run_ffmpeg(["-i", in_path, "-vf", f"fps={fps}", "-an", *h264_args(crf), out_path])
    return out_path
