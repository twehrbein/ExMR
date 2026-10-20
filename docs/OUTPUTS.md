# Demo outputs

`python tools/demo.py --video <video>` writes to `--output_dir` (default `outputs/<video name>`):

| File | Content |
|---|---|
| `results.pt` | reconstruction (PyTorch, see below) |
| `results.npz` | the same as NumPy arrays, with flattened keys (`smplx_incam_body_pose`, ...) |
| `<name>_incam.mp4` | SMPL-X mesh rendered over the input video |
| `<name>_global.mp4` | SMPL-X mesh in the world frame, with a ground plane, seen by a static camera |
| `<name>_incam_global.mp4` | both side by side (scaled down to at most 4096 pixels wide) |
| `cache/` | cached preprocessing results (reused when the demo is run again with the same settings) |
| `run.json` | the video and options of the run |

## results.pt

```python
results = torch.load("results.pt")
```

For a video with `L` frames:

| Key | Shape | Description |
|---|---|---|
| `smplx_incam` | dict | SMPL-X parameters in the camera frame (x right, y down, z forward) |
| `smplx_global` | dict | SMPL-X parameters in a gravity-aligned world frame (y up); the motion starts at the origin |
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
