# Third-party notices

ExMR is a derivative work of [GVHMR](https://github.com/zju3dv/GVHMR) (see [LICENSE](LICENSE)) and
uses the third-party models, code and data listed below. Each component remains under its own
license; using ExMR means accepting those terms as well.

## Models and code used at runtime

| Component | Used for | License | Notes |
|---|---|---|---|
| [SAM 3D Body](https://github.com/facebookresearch/sam-3d-body) | body image features | SAM License | git submodule `third_party/sam-3d-body` (b5c765a); gated weights on Hugging Face; acknowledge SAM in publications |
| [HaMeR](https://github.com/geopavlakos/hamer) | hand image features | MIT | checkpoint requires the MANO model |
| [Sapiens2](https://github.com/facebookresearch/sapiens2) | 2D whole-body keypoints | Sapiens2 License | use restrictions apply; license text must be shipped with adapted code |
| [RTMW](https://github.com/open-mmlab/mmpose) via [rtmlib](https://github.com/Tau-J/rtmlib) | 2D whole-body keypoints (alternative) | Apache-2.0 | |
| [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) | person detection and tracking | AGPL-3.0 | |
| [nvdiffrast](https://github.com/NVlabs/nvdiffrast) | rendering | NVIDIA Source Code License | non-commercial use only (research or evaluation purposes) |
| [DPVO](https://github.com/princeton-vl/DPVO) (optional) | visual odometry, alternative to SimpleVO | MIT | git submodule `third_party/DPVO` (c0c5a10) with a small build patch for torch >= 2.7 (`third_party/patches/DPVO-torch2.7.patch`) |

## Body models (not distributed, register and download separately)

| Model | License |
|---|---|
| [SMPL-X](https://smpl-x.is.tue.mpg.de/) | non-commercial research license |
| [MANO](https://mano.is.tue.mpg.de/) | non-commercial research license |

## Adapted source files

| File | Origin | License |
|---|---|---|
| `exmr/geometry/rotation_conversions.py` | [PyTorch3D](https://github.com/facebookresearch/pytorch3d) 0.7.9, `pytorch3d/transforms/rotation_conversions.py` (subset) | BSD-3-Clause, see [licenses/PyTorch3D_LICENSE.txt](licenses/PyTorch3D_LICENSE.txt) |
| `exmr/geometry/quaternion.py` | GVHMR `hmr4d/utils/geo/quaternion.py` (modified), which builds on `common/quaternion.py` of [QuaterNet](https://github.com/facebookresearch/QuaterNet) (`qmul`, `qrot`; Copyright (c) 2018-present, Facebook, Inc.) and of [HumanML3D](https://github.com/EricGuo5513/HumanML3D) (`qinv`, `qnormalize`, `qpow`, `qslerp`, `qbetween`; Copyright (c) 2022 Chuan Guo) | QuaterNet: CC BY-NC 4.0, see [licenses/QuaterNet_LICENSE.txt](licenses/QuaterNet_LICENSE.txt); HumanML3D: MIT, see [licenses/HumanML3D_LICENSE.txt](licenses/HumanML3D_LICENSE.txt) |
| `exmr/model/*`, `exmr/geometry/{kinematics,camera,smoothing}.py`, `exmr/body_models/smplx_skeleton.py` | adapted from GVHMR (`hmr4d/`) | GVHMR license, see [LICENSE](LICENSE) |
| `exmr/preprocessing/{tracker,visual_odometry,video}.py` | adapted from GVHMR (`hmr4d/utils/preproc`, SimpleVO) | GVHMR license, see [LICENSE](LICENSE) |
| `exmr/preprocessing/keypoints/sapiens2.py` | model definition and UDP heatmap decoding adapted from [Sapiens2](https://github.com/facebookresearch/sapiens2) | Sapiens2 License, see [licenses/Sapiens2_LICENSE.md](licenses/Sapiens2_LICENSE.md) |
| `exmr/preprocessing/keypoints/rtmw.py` | crop and SimCC decoding adapted from [RTMLib](https://github.com/Tau-J/rtmlib) | Apache-2.0, see [licenses/RTMLib_LICENSE.txt](licenses/RTMLib_LICENSE.txt) |
| `exmr/preprocessing/hand_features.py` | hand crop adapted from [HaMeR](https://github.com/geopavlakos/hamer) `hamer/datasets/utils.py` | MIT (Copyright (c) 2023 UC Regents, Georgios Pavlakos) |

`exmr/model/layers.py` re-implements timm's `Mlp` layout (same parameter names) so that the model
does not depend on timm.
