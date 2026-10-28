"""run_video: video -> preprocessing -> ExMR -> results (+ rendered videos).

Every preprocessing stage is cached in <output_dir>/cache together with the settings it was
computed with (including those of the stages it depends on), so rerunning with other settings
recomputes exactly the affected stages.
"""

import gc
import importlib.util
import json
import logging
import multiprocessing
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from exmr import __version__
from exmr.assets import RTMW_SHA256, Assets
from exmr.body_models.smplx_model import SmplxModel
from exmr.export import save_amass, save_bvh
from exmr.geometry.camera import bbx_xys_from_xyxy, cam_angvel_from_rotations, estimate_K
from exmr.preprocessing import video as V
from exmr.preprocessing.keypoints import VISIBILITY_THRESHOLD, detect_keypoints
from exmr.preprocessing.keypoints.mappings import COCO23, COCO63_LEFT_HAND, COCO63_RIGHT_HAND

log = logging.getLogger("exmr")

# without snap_to_ground, a hint is logged when the feet are higher above the ground (m, median)
# where they touch it, usually vertical drift of the world-frame motion in long videos
FLOATING_FEET_HINT = 0.05


@dataclass
class Options:
    pose_backend: str = "sapiens2"  # 2D keypoints: "sapiens2" or "rtmw"
    sapiens2_resolution: str = "448x336"  # or "576x432", "1024x768" (slower, similar accuracy)
    detector: str = "yolo11m"  # person detector: "yolo11m" or "yolo11x" (slower)
    static_cam: bool = False  # the camera is known to be static: no visual odometry
    vo: str = "simplevo"  # camera motion if not static_cam: "simplevo" or "dpvo" (optional)
    start: float | None = None  # only the part of the video from this time on (seconds)
    end: float | None = None  # ... up to this time (seconds; None: to its end)
    f_mm: float | None = None  # the camera's focal length (mm, full-frame equivalent), if known
    fps30: bool = False  # resample to 30 fps first (usually not needed; faster for e.g. 60 fps)
    snap_to_ground: bool = False  # flat ground, nothing to stand on: no vertical drift
    smoothing: str = "global"  # smooth the motion over time: "global" (world frame), "both", "none"
    render: bool = True
    keep_frames: bool = False  # keep the extracted JPEG frames

    def __post_init__(self):
        for name, choices in [
            ("pose_backend", ["sapiens2", "rtmw"]),
            ("sapiens2_resolution", ["1024x768", "576x432", "448x336"]),
            ("detector", ["yolo11m", "yolo11x"]),
            ("vo", ["simplevo", "dpvo"]),
            ("smoothing", ["global", "both", "none"]),
        ]:
            if getattr(self, name) not in choices:
                raise ValueError(f"Options.{name} must be one of {choices}")
        if self.start is not None and self.start < 0:
            raise ValueError("Options.start must be >= 0")
        if self.end is not None and self.end <= (self.start or 0):
            raise ValueError("Options.end must be after Options.start")
        if self.f_mm is not None and self.f_mm <= 0:
            raise ValueError("Options.f_mm must be > 0")


def run_video(video_path, output_dir, options=None, assets=None):
    """Reconstruct the main person of a video. Returns the results dict saved to
    <output_dir>/results.pt (see docs/OUTPUTS.md)."""
    options = options or Options()
    assets = assets or Assets()
    use_dpvo = options.vo == "dpvo" and not options.static_cam
    assets.check(options.pose_backend, options.detector, use_dpvo)
    if use_dpvo and importlib.util.find_spec("dpvo") is None:
        raise RuntimeError(
            "DPVO is not installed: see the optional DPVO section of docs/INSTALL.md, or use "
            "SimpleVO (--vo simplevo)"
        )
    from exmr.preprocessing.body_features import SAM_3D_BODY_ROOT

    submodule = (SAM_3D_BODY_ROOT / "sam_3d_body").is_dir()
    if not submodule and importlib.util.find_spec("sam_3d_body") is None:
        raise RuntimeError(
            "SAM 3D Body's code (the git submodule third_party/sam-3d-body) is missing, as the "
            "repository was cloned without --recursive: run git submodule update --init"
        )
    if options.render:
        _check_renderer()
    # GeForce tensor cores run FP16 matmuls with FP32 accumulation at half rate: accumulating in
    # FP16 makes the FP16 stages (Sapiens2, SAM 3D Body, HaMeR) 10-20% faster, with unchanged
    # accuracy (keypoints/tokens change at the 1e-3 level). Other GPUs
    # accumulate at full rate and gain nothing, so they keep the default.
    torch.backends.cuda.matmul.allow_fp16_accumulation = _is_geforce()
    video_path, output_dir = Path(video_path), Path(output_dir)
    if options.start is not None:
        duration = V.video_lwh(video_path)[0] / V.video_fps(video_path)
        if options.start >= duration:
            raise ValueError(f"start {options.start:g} s is past the video's end ({duration:.2f}s)")
    name = video_path.stem
    cache = output_dir / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    run_info = {"video": str(video_path), "options": asdict(options), "exmr": __version__}
    (output_dir / "run.json").write_text(json.dumps(run_info, indent=2) + "\n")

    cut_path = None
    if options.start is not None or options.end is not None:
        video_path = cut_path = _cut(video_path, output_dir, options)
    elif options.fps30:
        video_path = _resample_to_30fps(video_path, output_dir)
    frame_dir = output_dir / "frames"
    # every stage reads these frames, so all see the same images (ffmpeg applies the video's
    # rotation and other display metadata) and the same frame count (also of variable frame rate
    # videos)
    num_frames = _extract_frames(video_path, frame_dir)
    width, height = V.frame_size(frame_dir)
    fps = V.video_fps(video_path)
    log.info(f"{video_path}: {num_frames} frames, {width}x{height}, {fps:.2f} fps")

    data = _preprocess(video_path, frame_dir, num_frames, width, height, options, assets, cache)

    log.info("Running ExMR")
    results = _run_model(data, options, fps, assets)
    results["fps"] = fps
    smplx = SmplxModel(assets.smplx)
    results |= compute_joints(results, smplx)
    save_results(results, output_dir)
    save_amass(results["smplx_global"], fps, output_dir / f"{name}_amass.npz", smplx)
    save_bvh(results["smplx_global"], fps, output_dir / f"{name}.bvh", smplx)

    if options.render:
        log.info("Rendering")
        render(results, frame_dir, output_dir, name, fps, smplx.cuda())
    if not options.keep_frames:
        shutil.rmtree(frame_dir)
        if cut_path is not None:
            cut_path.unlink()  # lossless, so large; cut again in ~1 s if needed
    return results


def range_label(start, end):
    """The part start..end (seconds; None: the video's beginning / end) for file names, e.g.
    "5-15.5s" or "5s-end"."""
    return f"{start or 0:g}-{end:g}s" if end is not None else f"{start or 0:g}s-end"


def compute_joints(results, smplx):
    """The joints of results["smplx_incam"] and results["smplx_global"] (each (L, D)) and their
    projection into the image with results["K_fullimg"]: the keys joints_incam, joints_global,
    joints_2d and joint_names of results.pt (see docs/OUTPUTS.md). About a second for 1000
    frames on the CPU.

    Args:
        smplx: a SmplxModel of the parameters (on the CPU, like the results)
    """
    with torch.no_grad():
        incam, world = (smplx.joints(**results[key]) for key in ("smplx_incam", "smplx_global"))
    uvz = torch.einsum("lij,lkj->lki", results["K_fullimg"], incam)
    return {
        "joints_incam": incam,
        "joints_global": world,
        "joints_2d": uvz[..., :2] / uvz[..., 2:],
        "joint_names": smplx.joint_names,
    }


def _preprocess(video_path, frame_dir, L, width, height, options, assets, cache):
    detector = assets.yolo(options.detector).name
    # the ExMR version in every stage's settings: a new version recomputes the cached stages, whose
    # code may have changed (e.g. 0.1.4's tracker and VO read rotated videos sideways)
    tracker_settings = {
        "video": str(video_path),
        "frames": L,
        "detector": detector,
        "exmr": __version__,
    }
    backend = options.pose_backend
    kp_settings = {**tracker_settings, "pose_backend": backend}
    if backend == "sapiens2":
        kp_settings["resolution"] = options.sapiens2_resolution
    vo_cache = cache / "camera_rotations.pt"
    vo_settings = {
        "video": str(video_path),
        "frames": L,
        "odometry": options.vo,
        "f_mm": options.f_mm,
        "exmr": __version__,
    }

    # SimpleVO needs only the frames and the CPU: it runs in a separate process while the GPU
    # stages below run
    simplevo = None
    use_simplevo = options.vo == "simplevo" and not options.static_cam
    if use_simplevo and not _is_cached(vo_cache, vo_settings):
        log.info("Starting visual odometry in the background")
        simplevo = _in_background(_simplevo_rotations, frame_dir, options.f_mm)

    # RTMW's session build and warm-up run in a thread while
    # the tracker runs, which hides about half of it
    rtmw = None
    if backend == "rtmw" and not _is_cached(cache / "keypoints.pt", kp_settings):
        rtmw = _in_thread(_warm_rtmw, assets.rtmw)

    def track():
        from exmr.preprocessing.tracker import Tracker

        return Tracker(assets.yolo(options.detector)).get_main_track(frame_dir).float()

    bbx_xyxy = _cached(cache / "tracks.pt", tracker_settings, track)  # (L, 4)
    K = estimate_K(width, height, options.f_mm)[None].repeat(L, 1, 1)  # (L, 3, 3)

    def keypoints():
        if backend == "sapiens2":
            return detect_keypoints(
                backend, assets.sapiens2, frame_dir, bbx_xyxy,
                resolution=options.sapiens2_resolution,
            )  # fmt: skip
        return rtmw.result()(frame_dir, bbx_xyxy)

    kp2d = _cached(cache / "keypoints.pt", kp_settings, keypoints)  # (L, 63, 3)

    def body_features():
        from exmr.preprocessing.body_features import SamBodyFeatures

        # the backbone profits from FP16 accumulation only in float16 (see run_video)
        fp16 = torch.backends.cuda.matmul.allow_fp16_accumulation
        return SamBodyFeatures(assets.sam_3d_body, fp16=fp16)(frame_dir, bbx_xyxy, K)

    body_settings = {**tracker_settings, "f_mm": options.f_mm}
    body_feat = _cached(cache / "body_features.pt", body_settings, body_features)

    def hand_features():
        from exmr.preprocessing.hand_features import HamerHandFeatures

        return HamerHandFeatures(assets.hamer)(frame_dir, hand_boxes(kp2d, backend))

    hand_feat = _cached(cache / "hand_features.pt", kp_settings, hand_features)

    if options.static_cam:
        R_w2c = torch.eye(3).repeat(L, 1, 1)
    else:

        def camera_rotations():
            from exmr.preprocessing.visual_odometry import DPVOOdometry

            if options.vo == "dpvo":
                return DPVOOdometry(assets.dpvo)(frame_dir, K[0])
            return torch.from_numpy(simplevo.result())  # waits for SimpleVO to finish

        R_w2c = _cached(vo_cache, vo_settings, camera_rotations)

    inputs = model_inputs(kp2d, bbx_xyxy, K, R_w2c, body_feat, hand_feat, backend)
    return {**inputs, "bbx_xyxy": bbx_xyxy}


def hand_boxes(kp2d, backend):
    """(L, 2, 4) boxes [left, right] of the hands for HaMeR, from the (L, 63, 3) keypoints of the
    pose backend ("sapiens2" or "rtmw")."""
    from exmr.preprocessing.hand_features import hand_boxes_from_keypoints

    threshold = VISIBILITY_THRESHOLD[backend]
    return torch.stack(
        [
            hand_boxes_from_keypoints(kp2d[:, COCO63_LEFT_HAND], threshold),
            hand_boxes_from_keypoints(kp2d[:, COCO63_RIGHT_HAND], threshold),
        ],
        dim=1,
    )


def model_inputs(kp2d, bbx_xyxy, K_fullimg, R_w2c, body_feat, hand_feat, backend):
    """ExMR's inputs (see ExMR.forward, without the batch dimension) from the preprocessing
    stages' outputs of one video: (L, 63, 3) keypoints of the pose backend ("sapiens2" or
    "rtmw"), (L, 4) person boxes, (L, 3, 3) intrinsics, (L, 3, 3) world-to-camera rotations,
    (L, 1024) SAM 3D Body and (L, 2, 1024) HaMeR features."""
    # body + foot keypoints; RTMW's raw scores become a binary visibility for the network
    kp23 = kp2d[:, COCO23].clone()
    if backend == "rtmw":
        kp23[..., 2] = (kp23[..., 2] > VISIBILITY_THRESHOLD[backend]).to(kp23.dtype)
    return {
        "kp2d": kp23,
        "bbx_xys": bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2).float(),
        "K_fullimg": K_fullimg,
        "cam_angvel": cam_angvel_from_rotations(R_w2c),
        "body_feat": body_feat,
        "hand_feat": hand_feat,
    }


def _simplevo_rotations(frame_dir, f_mm=None):
    """SimpleVO's (L, 3, 3) camera rotations, as a numpy array: torch tensors would be returned
    from the separate process through shared memory. f_mm: the focal length (SimpleVO's default
    if None)."""
    from exmr.preprocessing.visual_odometry import SimpleVO

    vo = SimpleVO(progress=False) if f_mm is None else SimpleVO(f_mm=f_mm, progress=False)
    return vo(frame_dir).numpy()


def _run_model(data, options, fps, assets):
    from exmr.model.exmr import ExMR

    model = ExMR.from_pretrained(assets.exmr, assets.smplx)
    inputs = {
        k: data[k][None].cuda()
        for k in ("kp2d", "bbx_xys", "K_fullimg", "cam_angvel", "body_feat", "hand_feat")
    }
    out = model(
        **inputs,
        static_cam=options.static_cam,
        snap_to_ground=options.snap_to_ground,
        smooth_incam=options.smoothing == "both",
        smooth_global=options.smoothing != "none",
        fps=fps,
    )
    from exmr.model.global_traj import foot_contacts

    contact, foot_height = foot_contacts(
        out["smplx_global"], out["static_conf_logits"], model.body_model
    )
    if not options.snap_to_ground:
        _hint_floating_feet(foot_height[contact])
    del model
    _free_gpu_memory()
    return {
        "smplx_incam": {k: v[0].cpu() for k, v in out["smplx_incam"].items()},
        "smplx_global": {k: v[0].cpu() for k, v in out["smplx_global"].items()},
        "static_conf_logits": out["static_conf_logits"][0].cpu(),
        "foot_contact": contact[0].cpu(),
        "K_fullimg": data["K_fullimg"],
        "bbx_xyxy": data["bbx_xyxy"],
    }


def _hint_floating_feet(height):
    """height: of the feet where they touch the ground (all frames, both feet)"""
    if len(height) and height.median() > FLOATING_FEET_HINT:
        log.warning(
            f"In the world frame, the feet are {height.median() * 100:.0f} cm above the ground "
            "(median) where they touch it, probably vertical drift. If the scene has a single "
            "flat ground and nothing to stand on, snap_to_ground (demo: --snap_to_ground) keeps "
            "the body on it"
        )


def save_results(results, output_dir):
    """Save the results of run_video as <output_dir>/results.pt and results.npz (the same values
    with flattened keys, e.g. smplx_global_transl), see docs/OUTPUTS.md."""
    _atomic_save(results, output_dir / "results.pt")
    flat = {}
    for key, value in results.items():
        if isinstance(value, dict):
            flat |= {f"{key}_{k}": v.numpy() for k, v in value.items()}
        else:
            flat[key] = np.asarray(value)
    np.savez(output_dir / "results.npz", **flat)


def _is_geforce():
    return torch.cuda.is_available() and "GeForce" in torch.cuda.get_device_name()


def _check_renderer():
    """Render a triangle with nvdiffrast, so that a missing or broken install (e.g. built for
    another torch version or GPU) stops the run now rather than after the reconstruction."""
    try:
        import nvdiffrast.torch as dr

        triangle = torch.tensor([[[-1.0, -1, 0, 1], [1, -1, 0, 1], [0, 1, 0, 1]]], device="cuda")
        faces = torch.tensor([[0, 1, 2]], dtype=torch.int32, device="cuda")
        dr.rasterize(dr.RasterizeCudaContext(), triangle, faces, (8, 8))
    except Exception as e:
        raise RuntimeError(
            f"nvdiffrast, which renders the videos, does not work: {e}\nInstall it as in "
            "docs/INSTALL.md, or skip the videos (--no_render); the other outputs need no "
            "nvdiffrast"
        ) from e


def render(
    results, frame_dir, output_dir, name, fps, smplx, videos=("incam", "global", "incam_global")
):
    """Render the results of run_video over the frames (frame_dir/%06d.jpg) and in the world
    frame: <output_dir>/<name>_<video>.mp4 for the videos "incam" (camera view), "global" (world
    view) and "incam_global" (both side by side); only the views needed are rendered. smplx: a
    SmplxModel on the GPU."""
    from exmr.vis.render import render_videos

    with torch.no_grad():
        verts_incam = smplx(**{k: v.cuda() for k, v in results["smplx_incam"].items()})
        verts_global = smplx(**{k: v.cuda() for k, v in results["smplx_global"].items()})
    render_videos(
        frame_dir, verts_incam, verts_global, smplx.faces, smplx.J_regressor.cuda(),
        results["K_fullimg"][0], fps,
        {view: output_dir / f"{name}_{view}.mp4" for view in videos},
    )  # fmt: skip


def _warm_rtmw(checkpoint):
    from exmr.preprocessing.keypoints.rtmw import RTMWKeypoints

    model = RTMWKeypoints(checkpoint, sha256=RTMW_SHA256)
    model.warm_up()
    return model


def _in_thread(fn, *args):
    """The future result of fn(*args), computed in a new thread."""
    executor = ThreadPoolExecutor(1)
    future = executor.submit(fn, *args)
    executor.shutdown(wait=False)
    return future


def _in_background(fn, *args):
    """The future result of fn(*args), computed in a new process. The process is started with
    "spawn": a forked process would inherit this process's state (e.g. CUDA, OpenCV's threads)."""
    executor = ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn"))
    future = executor.submit(fn, *args)
    executor.shutdown(wait=False)  # the process exits once fn has returned
    return future


def _is_cached(path, settings):
    """Whether path holds a value cached with these settings (see _cached)."""
    if not path.exists():
        return False
    return torch.load(path, map_location="cpu", weights_only=True)["settings"] == settings


def _cached(path, settings, compute):
    """compute(), cached in path together with settings; recomputed when the settings differ."""
    if path.exists():
        cached = torch.load(path, map_location="cpu", weights_only=True)
        if cached["settings"] == settings:
            log.info(f"Using cached {path.name}")
            return cached["value"]
    log.info(f"Computing {path.stem}")
    value = compute()
    _atomic_save({"settings": settings, "value": value}, path)
    _free_gpu_memory()
    return value


def _atomic_save(obj, path):
    """Save to a temporary file first, so that an interrupted run leaves no partial file."""
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _extract_frames(video_path, frame_dir):
    marker = frame_dir / ".extracted"
    if marker.exists() and marker.read_text().split("\n")[0] == str(video_path):
        return int(marker.read_text().split("\n")[1])
    log.info(f"Extracting frames to {frame_dir}")
    num_frames = V.extract_frames(video_path, frame_dir)
    marker.write_text(f"{video_path}\n{num_frames}")
    return num_frames


def _cut(video_path, output_dir, options):
    """The part of the video from options.start to options.end, in output_dir; with options.fps30
    also resampled to 30 fps (unless it has ~30 fps already, as _resample_to_30fps)."""
    resample = options.fps30 and abs(V.video_fps(video_path) - 30) >= 0.5
    label = range_label(options.start, options.end)
    out = output_dir / f"{video_path.stem}_{label}{'_30fps' if resample else ''}.mp4"
    if not out.exists():
        log.info(f"Cutting {label}" + (", resampled to 30 fps" if resample else ""))
        tmp = out.with_name(out.stem + ".tmp.mp4")  # no complete-looking file if interrupted
        V.cut(video_path, tmp, options.start, options.end, fps=30 if resample else None)
        os.replace(tmp, out)
    return out


def _resample_to_30fps(video_path, output_dir):
    fps = V.video_fps(video_path)
    if abs(fps - 30) < 0.5:
        return video_path
    out = output_dir / f"{video_path.stem}_30fps.mp4"
    if not out.exists():
        log.info(f"Resampling {fps:.2f} fps to 30 fps")
        V.resample_fps(video_path, out, fps=30)
    return out


def _free_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()
