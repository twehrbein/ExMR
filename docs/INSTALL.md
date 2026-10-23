# Installation

Tested on Linux with an NVIDIA RTX 4090, CUDA 12.8 (driver >= 570), Python 3.12. The demo needs
about 8 GB of GPU memory (7.4 GB peak measured on a 1080x1920 video).

**Windows:** not tested. Use [WSL2](https://learn.microsoft.com/windows/wsl/install) (Ubuntu, with
the NVIDIA driver >= 570 installed in Windows) and follow these instructions inside it. Native
Windows is not supported.

> [!IMPORTANT]
> Request access to [SAM 3D Body](https://huggingface.co/facebook/sam-3d-body-dinov3) on Hugging Face
> before you start: its weights are gated and access is approved manually, which can take a while.
> SMPL-X and MANO need a (free) registration too, see [model files](#3-model-files).

## 1. Code

```bash
git clone --recursive https://github.com/twehrbein/ExMR.git
cd ExMR
```

`third_party/` holds pinned git submodules: SAM 3D Body (required) and DPVO (optional).

## 2. Environment

```bash
conda create -y -n exmr python=3.12
conda activate exmr
# ignore packages in ~/.local, which would otherwise shadow the environment's (set once, kept
# for every activation of this environment)
conda env config vars set PYTHONNOUSERSITE=1
conda activate exmr

pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
conda install -y -c conda-forge ffmpeg=7.1.1 libjpeg-turbo
# constraints.txt pins the indirect dependencies too, to the validated versions
pip install -r requirements.txt -c constraints.txt
# HaMeR and Sapiens2 at the evaluated versions, without their (training) dependencies
# (`pip check` lists those as missing; they are not needed)
pip install --no-deps \
    "git+https://github.com/geopavlakos/hamer.git@3a01849f4148352e9260b69bf28b65d1671a4905" \
    "git+https://github.com/facebookresearch/sapiens2.git@7e5bae88456ac418ff0e58e74106c9fe192055d4"
# nvdiffrast renders the result videos: a prebuilt wheel for exactly this torch version
pip install "https://huggingface.co/twehrbein/ExMR/resolve/03e36730ba7466ef09fcd98523fec6d9c2da8a0a/wheels/nvdiffrast-0.4.0%2Bpt2.8.0cu128-cp312-cp312-linux_x86_64.whl#sha256=ca26d44f4f198e8fb460f86f355c5fdaeaac6319e15a5551ea31397719c0d563"
pip install -e .
```

The wheel is [nvdiffrast](https://github.com/NVlabs/nvdiffrast) v0.4.0, built by us for this torch
version and hosted with the ExMR weights (see the
[model card](https://huggingface.co/twehrbein/ExMR)); the hash pins the tested file. With another
torch version, [build nvdiffrast](#optional-build-nvdiffrast-from-source) instead. The demo tries
nvdiffrast before it starts; without it, everything but the videos works (`--no_render`).

### Optional: build nvdiffrast from source

nvdiffrast is compiled against the installed torch, with nvcc 12.x and gcc <= 14. Install both
into the environment, so that the build does not depend on the system's CUDA toolkit (DPVO below
needs the same):

```bash
conda install -y -c conda-forge cuda-version=12.8 cuda-nvcc cuda-cudart-dev cuda-cccl \
    libcublas-dev libcusparse-dev libcusolver-dev ninja
# re-activate: the compiler packages just installed point CC/CXX to the environment's gcc 14
# only on activation (a newer system gcc, e.g. 15, is rejected by nvcc 12.8)
conda activate exmr
# build for the GPUs that the torch build above supports (the compiler packages set a longer
# list, with older GPUs that DPVO cannot be built for); fewer build faster, see the table below
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0;10.0;12.0"
CUDA_HOME=$CONDA_PREFIX pip install --no-build-isolation \
    "git+https://github.com/NVlabs/nvdiffrast.git@v0.4.0"
```

`TORCH_CUDA_ARCH_LIST` takes the compute capabilities of the GPUs to build for, separated by
semicolons (e.g. `"8.6;8.9"`); `nvidia-smi --query-gpu=name,compute_cap --format=csv` prints them.
The list above covers all GPUs that the torch build supports (RTX 40 series GPUs run the code
built for `8.6`):

| GPU | `TORCH_CUDA_ARCH_LIST` |
|---|---|
| V100 | `7.0` |
| RTX 20 series, T4 | `7.5` |
| A100 | `8.0` |
| RTX 30 series, A40, RTX A6000 | `8.6` |
| RTX 40 series, L4, L40, RTX 6000 Ada | `8.9` |
| H100, H200 | `9.0` |
| B200 | `10.0` |
| RTX 50 series, RTX PRO 6000 Blackwell | `12.0` |

### Optional: DPVO

DPVO, a learned visual odometry method, is an alternative to the default SimpleVO for estimating
the motion of a moving camera (`--vo dpvo`); `--mode accurate` uses it. See
[camera motion](../README.md#camera-motion) for the differences. It is compiled with the
compiler from [build nvdiffrast](#optional-build-nvdiffrast-from-source) (its `conda install` and
re-activation).

```bash
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0;10.0;12.0"  # as for nvdiffrast, see above
git -C third_party/DPVO apply ../patches/DPVO-torch2.7.patch  # build fix for torch >= 2.7
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
unzip eigen-3.4.0.zip -d third_party/DPVO/thirdparty && rm eigen-3.4.0.zip
pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.8.0+cu128.html \
    numba==0.68.0 pypose==0.7.2 -c constraints.txt
CUDA_HOME=$CONDA_PREFIX pip install --no-build-isolation --no-deps -e third_party/DPVO
```

## 3. Model files

Three models need a (free) registration or access request first:

* **SMPL-X**: register at [smpl-x.is.tue.mpg.de](https://smpl-x.is.tue.mpg.de) and download
  `models_smplx_v1_1.zip` ("SMPL-X v1.1 (NPZ+PKL)").
* **MANO**: register at [mano.is.tue.mpg.de](https://mano.is.tue.mpg.de) and download
  `mano_v1_2.zip` ("Models & Code").
* **SAM 3D Body** is gated on Hugging Face: request access at
  [facebook/sam-3d-body-dinov3](https://huggingface.co/facebook/sam-3d-body-dinov3) and, once
  approved, log in with `hf auth login`.

Then download everything else and unpack the two zips (~12 GB, about 10 minutes):

```bash
python tools/download_models.py --smplx_zip models_smplx_v1_1.zip --mano_zip mano_v1_2.zip
# optional DPVO weights: add --dpvo
```

The script checks every file against the version ExMR was tested with, skips files that are already
present (rerun it after an interrupted download) and prints each model's license; downloading a
model means accepting its license. It also stores the DINOv3 code that SAM 3D Body loads through
`torch.hub`, so that the demo runs offline afterwards.

All model files go into one directory, `checkpoints/` by default (or set `EXMR_CHECKPOINT_DIR`, or
pass `--checkpoint_dir` to both scripts):

```
checkpoints/
├── exmr/exmr.safetensors                  ExMR
├── body_models/smplx/SMPLX_NEUTRAL.npz    SMPL-X (register at smpl-x.is.tue.mpg.de)
├── yolo/yolo11x.pt, yolo11m.pt            person detector (Ultralytics)
├── sapiens2/sapiens2_1b_pose.safetensors  2D keypoints (default)
├── rtmw/rtmw-x_384x288_fp16.onnx          2D keypoints (--mode fast)
├── sam-3d-body-dinov3/                    body features: model.ckpt, model_config.yaml,
│                                          assets/mhr_model.pt
├── hamer/                                 hand features (HaMeR's hamer_demo_data):
│   ├── hamer_ckpts/                       checkpoints/hamer.ckpt, model_config.yaml
│   └── data/                              mano_mean_params.npz,
│                                          mano/MANO_RIGHT.pkl (register at mano.is.tue.mpg.de)
└── dpvo/dpvo.pth                          optional, for --mode accurate / --vo dpvo
```

The demo checks for all files it needs before it starts and lists the missing ones.
