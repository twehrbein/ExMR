# Demo outputs

`python tools/demo.py --video <video>` writes to `--output_dir` (default `outputs/<video name>`):

| File | Content |
|---|---|
| `results.pt` | reconstruction (PyTorch, see below) |
| `results.npz` | the same as NumPy arrays, with flattened keys (`smplx_incam_body_pose`, ...) |
| `<name>_amass.npz` | the world-frame motion as an AMASS SMPL-X file, e.g. for Blender, see [below](#blender-and-amass) |
| `<name>_incam.mp4` | SMPL-X mesh rendered over the input video |
| `<name>_global.mp4` | SMPL-X mesh in the world frame, with a ground plane, seen by a static camera |
| `<name>_incam_global.mp4` | both side by side (scaled down to at most 4096 pixels wide) |
| `cache/` | cached preprocessing results (reused when the demo is run again with the same settings) |
| `run.json` | the video and options of the run (e.g. `smoothing`) |

## results.pt

```python
results = torch.load("results.pt")
```

For a video with `L` frames:

| Key | Shape | Description |
|---|---|---|
| `smplx_incam` | dict | SMPL-X parameters in the camera frame (x right, y down, z forward); not smoothed over time unless `--smoothing both` |
| `smplx_global` | dict | SMPL-X parameters in a gravity-aligned world frame (y up); the motion starts at the origin (x, z) and stands on the ground y=0: the lowest mesh vertex over all frames is at y=0 (with `--snap_to_ground`: the lowest vertex within every 0.5 s); smoothed over time unless `--smoothing none` |
| `static_conf_logits` | `(L, 6)` | logits of [left ankle, left foot, right ankle, right foot, left wrist, right wrist] being static (in contact) |
| `K_fullimg` | `(L, 3, 3)` | camera intrinsics used (estimated from the image size) |
| `bbx_xyxy` | `(L, 4)` | tracked bounding box of the person, `[x1, y1, x2, y2]` in pixels |
| `fps` | float | frame rate of the processed video |

Both SMPL-X parameter dicts contain, per frame, axis-angle rotations:

| Key | Shape | Description |
|---|---|---|
| `global_orient` | `(L, 3)` | root rotation |
| `body_pose` | `(L, 63)` | 21 body joints, relative to their parent |
| `left_hand_pose`, `right_hand_pose` | `(L, 45)` | 15 finger joints per hand (full axis-angle, no PCA, flat hand mean) |
| `betas` | `(L, 10)` | body shape (the same for all frames) |
| `transl` | `(L, 3)` | root translation in meters |

Face parameters (jaw, eyes, expression) are not estimated; use zeros.

To get the mesh with the [smplx](https://github.com/vchoutas/smplx) package:

```python
import smplx

params = results["smplx_incam"]  # or results["smplx_global"]
L = len(params["body_pose"])
model = smplx.SMPLX(
    "SMPLX_NEUTRAL.npz", use_pca=False, flat_hand_mean=True, num_betas=10, batch_size=L
)  # batch_size: the face parameters, which ExMR does not estimate, default to zeros of that size
vertices = model(**params).vertices  # (L, 10475, 3)
```

`exmr.body_models.smplx_model.SmplxModel` gives the same vertices.

## Blender and AMASS

`<name>_amass.npz` holds the world-frame motion (`smplx_global`) in the layout of the SMPL-X files
of [AMASS](https://amass.is.tue.mpg.de), which the SMPL-X Blender add-on imports:

1. Download "SMPL-X for Blender" from the [SMPL-X website](https://smpl-x.is.tue.mpg.de)
   (registration, Blender 4.5 or newer) and install the zip in Blender (Edit > Preferences > Get
   Extensions > Install from Disk).
2. In the 3D viewport's sidebar (View > Sidebar), tab "SMPL Models", click "Add Animation" and
   select the file. The default settings fit (Format: AMASS, Hand pose reference: Flat).
3. Set "Target framerate" to the file's frame rate (`mocap_frame_rate`) or a divisor of it, e.g. 25
   for a 25 fps video: the add-on rejects targets above the frame rate and otherwise keeps every
   (frame rate / target)-th frame, rounded down. Its default of 30 suits 30 and 60 fps videos.
4. For the exact SMPL-X mesh, enable "Use keyframed corrective pose weights" (slower import and
   playback). Without it, the skeleton moves the same, but the mesh lacks SMPL-X's pose-dependent
   shape corrections.

The add-on's default body model is SMPL-X "locked head" (without the head bun of SMPL-X v1.1,
which ExMR uses). The joints and the body agree within about a millimeter; only the head differs.

As in AMASS, the world frame is z-up with the ground at z=0: the motion is `smplx_global` rotated
by 90 degrees about the x-axis, i.e. the point (x, y, z) of `smplx_global` is at (x, -z, y). The
hand poses are relative to the flat hand, as in `results.pt`.

| Key | Shape | Description |
|---|---|---|
| `poses` | `(L, 165)` | axis-angle rotations of the 55 SMPL-X joints: root, 21 body joints, jaw, 2 eyes, 15 + 15 finger joints (jaw and eyes are zero) |
| `root_orient`, `pose_body`, `pose_jaw`, `pose_eye`, `pose_hand` | `(L, 3)`, `(L, 63)`, `(L, 3)`, `(L, 6)`, `(L, 90)` | the same, split into parts |
| `trans` | `(L, 3)` | root translation in meters |
| `betas` | `(10,)` | body shape (`num_betas`: 10) |
| `gender`, `surface_model_type` | | `neutral`, `smplx` |
| `mocap_frame_rate` | | frame rate, rounded to an integer (the Blender add-on reads it as one); the exact rate is `fps` in `results.pt` |

`exmr.export.save_amass` writes the file from `results.pt`, e.g. for results of an earlier ExMR
version:

```python
from exmr.export import save_amass

save_amass(results["smplx_global"], results["fps"], "motion_amass.npz", "SMPLX_NEUTRAL.npz")
```
