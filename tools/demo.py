"""ExMR demo: reconstruct the body, hand and foot motion of the main person in a video.

    python tools/demo.py --video path/to/video.mp4                  # moving camera
    python tools/demo.py --video path/to/video.mp4 -s               # static camera
    python tools/demo.py --video path/to/video.mp4 --mode fast      # fastest
    python tools/demo.py --video path/to/video.mp4 --mode accurate  # slowest, needs DPVO

Modes: person detector, 2D keypoints, camera motion (moving camera)
    fast       YOLO11m, RTMW-x,                SimpleVO
    balanced   YOLO11m, Sapiens2-1B 448x336,   SimpleVO   (default)
    accurate   YOLO11x, Sapiens2-1B 1024x768,  DPVO
--detector, --pose_backend, --sapiens2_resolution and --vo override single parts.

Writes results.pt / results.npz (SMPL-X parameters, see docs/OUTPUTS.md), <video name>_amass.npz
(the world-frame motion for Blender) and rendered videos to --output_dir (default:
outputs/<video name>).
"""

import argparse
import logging
from pathlib import Path

from exmr.assets import DEFAULT_CHECKPOINT_DIR, Assets
from exmr.pipeline import Options, run_video

PRESETS = {
    "fast": dict(detector="yolo11m", pose_backend="rtmw", sapiens2_resolution="448x336",
                 vo="simplevo"),
    "balanced": dict(detector="yolo11m", pose_backend="sapiens2", sapiens2_resolution="448x336",
                     vo="simplevo"),
    "accurate": dict(detector="yolo11x", pose_backend="sapiens2", sapiens2_resolution="1024x768",
                     vo="dpvo"),
}  # fmt: skip


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output_dir", type=Path, help="default: outputs/<video name>")
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help="model files, see docs/INSTALL.md (default: $EXMR_CHECKPOINT_DIR or ./checkpoints)",
    )
    parser.add_argument(
        "-s", "--static_cam", action="store_true", help="the camera does not move (skips VO)"
    )
    parser.add_argument(
        "--mode", choices=list(PRESETS), default="balanced", help="speed/accuracy preset"
    )
    parser.add_argument("--pose_backend", choices=["sapiens2", "rtmw"], help="2D keypoints")
    parser.add_argument(
        "--sapiens2_resolution",
        choices=["1024x768", "576x432", "448x336"],
        help="Sapiens2 input height x width; lower is faster",
    )
    parser.add_argument("--detector", choices=["yolo11x", "yolo11m"], help="person detector")
    parser.add_argument(
        "--vo",
        choices=["simplevo", "dpvo"],
        help="camera motion estimation for a moving camera (DPVO is an optional install)",
    )
    parser.add_argument(
        "--fps30",
        action="store_true",
        help="resample the video to 30 fps first; usually not needed. ExMR is trained on 30 fps "
        "videos, so you may try it for videos far from 30 fps (e.g. 50 fps or more), which also "
        "makes the demo faster for them (fewer frames to process). The results then "
        "follow the frames of the resampled video. Skipped for ~30 fps videos.",
    )
    parser.add_argument(
        "--snap_to_ground",
        action="store_true",
        help="assume a single flat ground and nothing to stand on (no stairs, chairs, ...): the "
        "body touches the ground at least every 0.5 s, so the world-frame motion is moved "
        "vertically to stay on the ground. Removes vertical drift (the person seems to fly or "
        "sink) in long videos; jumps with a longer flight are pulled down.",
    )
    parser.add_argument(
        "--smoothing",
        choices=["global", "both", "none"],
        default="global",
        help="temporal smoothing of the output motion (Gaussian, ~33 ms). global (default): the "
        "world-frame motion (smplx_global, _global video) only; the camera-frame motion stays "
        "raw, best aligned with the image. both: also the camera-frame motion (smplx_incam, "
        "_incam video), smoother but following fast foot motion slightly less closely in the "
        "image. none: the raw network output.",
    )
    parser.add_argument("--no_render", action="store_true", help="only save the results")
    parser.add_argument("--keep_frames", action="store_true", help="keep the extracted frames")
    args = parser.parse_args()

    # ExMR's progress at INFO; third-party libraries only warnings
    logging.basicConfig(
        level=logging.WARNING, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S"
    )
    log = logging.getLogger("exmr")
    log.setLevel(logging.INFO)
    if not args.video.exists():
        parser.error(f"video not found: {args.video}")
    # the preset, with the explicitly given parts replaced
    settings = {k: getattr(args, k) or v for k, v in PRESETS[args.mode].items()}
    keypoints = (
        f"Sapiens2-1B {settings['sapiens2_resolution']}"
        if settings["pose_backend"] == "sapiens2"
        else "RTMW-x"
    )
    vo = {"simplevo": "SimpleVO", "dpvo": "DPVO"}[settings["vo"]]
    if args.static_cam:
        vo = "static camera"
    detector = settings["detector"].replace("yolo", "YOLO")
    log.info(f"Mode {args.mode}: {detector}, {keypoints}, {vo}; smoothing: {args.smoothing}")
    options = Options(
        pose_backend=settings["pose_backend"],
        sapiens2_resolution=settings["sapiens2_resolution"],
        detector=settings["detector"],
        static_cam=args.static_cam,
        vo=settings["vo"],
        fps30=args.fps30,
        snap_to_ground=args.snap_to_ground,
        smoothing=args.smoothing,
        render=not args.no_render,
        keep_frames=args.keep_frames,
    )
    output_dir = args.output_dir or Path("outputs") / args.video.stem
    run_video(args.video, output_dir, options, Assets(args.checkpoint_dir))
    log.info(f"Done: {output_dir}")


if __name__ == "__main__":
    main()
