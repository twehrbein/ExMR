"""Download the model files into the checkpoint directory (layout: exmr/assets.py, docs/INSTALL.md).

    python tools/download_models.py --smplx_zip models_smplx_v1_1.zip --mano_zip mano_v1_2.zip
    python tools/download_models.py --only sapiens2 rtmw     # selected components only
    python tools/download_models.py --dpvo                   # also the optional DPVO weights

SMPL-X and MANO require registration: download models_smplx_v1_1.zip ("SMPL-X v1.1 (NPZ+PKL)") from
https://smpl-x.is.tue.mpg.de and mano_v1_2.zip from https://mano.is.tue.mpg.de and pass the zips.
SAM 3D Body is gated: request access at https://huggingface.co/facebook/sam-3d-body-dinov3, then
log in with `hf auth login`.

Every file is checked against the SHA256 of the file ExMR was tested with. Files that are already
present are skipped (re-check them with --verify). Each model stays under its own license, printed
before its download; by downloading it you accept those terms.
"""

import argparse
import hashlib
import io
import pickle
import shutil
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import scipy.sparse
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exmr.assets import DEFAULT_CHECKPOINT_DIR, RTMW_SHA256  # noqa: E402

EXMR_REPO, EXMR_REVISION = "twehrbein/ExMR", "8debd2e34262fcc63384f7e04d39ca784b0aec21"
SAPIENS2_REPO, SAPIENS2_REVISION = (
    "facebook/sapiens2-pose-1b",
    "f5fed8b97b99698d5eea1d14ff0855d0b4c3f000",
)
SAM3D_REPO, SAM3D_REVISION = (
    "facebook/sam-3d-body-dinov3",
    "11aaa346c7204874a1cbafe3d39a979080b2c55a",
)
# DINOv3 code for SAM 3D Body's backbone (torch.hub; the weights come with SAM 3D Body)
DINOV3_COMMIT = "6876159a11b4df116f30f667f8c9888617df0751"
DINOV3_TREE_SHA256 = "0389aafef4614add2090f2907f210698df48f16d3f2b845b6002c4ebb6aea0f8"
YOLO_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/{}"
HAMER_URL = "https://www.cs.utexas.edu/~pavlakos/hamer/data/hamer_demo_data.tar.gz"
DPVO_URL = (
    "https://www.dropbox.com/scl/fo/2g6e134ruh1acmaye5bmy/AK0DY9WLZ25elq-7N7sJcB0/models.zip"
    "?rlkey=4j336vcc4h324j4p4ji3ppl0t&dl=1"
)

SHA256 = {
    "exmr/exmr.safetensors": "8d3947310c30444ec59dae4662574d7a32bf83af30320eeac87146529ff2fda5",
    "exmr/config.json": "b29ad86d7916a7f54154c1a20fb07d3bc9d794da4d671c88ba0cf75a5cec8c43",
    "body_models/smplx/SMPLX_NEUTRAL.npz": (
        "376021446ddc86e99acacd795182bbef903e61d33b76b9d8b359c2b0865bd992"
    ),
    "yolo/yolo11m.pt": "d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95",
    "yolo/yolo11x.pt": "7bc158aa95c0ebfdd87f70f01653c1131b93e92522dbe15c228bcd742e773a24",
    "sapiens2/sapiens2_1b_pose.safetensors": (
        "2dab7014a17e99e460c18817325a71dd7a81ce48d87027f01c2ee7d7b3af969f"
    ),
    "rtmw/rtmw-x_384x288_fp16.onnx": RTMW_SHA256,
    "sam-3d-body-dinov3/model.ckpt": (
        "b5a2f9d305dd02626b967aa2e86021fba07065df66ce7a7e00ffb9664f150abf"
    ),
    "sam-3d-body-dinov3/model_config.yaml": (
        "1012fc3f39cb5e90e3f8fbadf7bded31604bfafdce0321d17a7c1a2d3f08b88d"
    ),
    "sam-3d-body-dinov3/assets/mhr_model.pt": (
        "352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc"
    ),
    "hamer/hamer_ckpts/checkpoints/hamer.ckpt": (
        "e5cc06f294d88a92dee24e603480aab04de532b49f0e08200804ee7d90e16f53"
    ),
    "hamer/hamer_ckpts/model_config.yaml": (
        "0e5eeb82752e47dfd01db8e13ccc4c5eba9bf83f53da8285523b8d3e87247aa3"
    ),
    "hamer/hamer_ckpts/dataset_config.yaml": (
        "392d10aece296152769328f14bb1c0137d6ea2f50f7e69f716ee00e9ad6ac343"
    ),
    "hamer/data/mano_mean_params.npz": (
        "efc0ec58e4a5cef78f3abfb4e8f91623b8950be9eff8b8e0dbb0d036ebc63988"
    ),
    "dpvo/dpvo.pth": "30d02dc2b88a321cf99aad8e4ea1152a44d791b5b65bf95ad036922819c0ff12",
}
# the converted MANO pickle is not byte-identical across Python versions: check its arrays instead
MANO_RIGHT_ARRAYS_SHA256 = "a7a63a5c0b352fbf7e1b8a565e353e126c63066de0cc0c9f8af031dff59d98c2"

LICENSES = {
    "exmr": "ExMR license (non-commercial research), https://github.com/twehrbein/ExMR/blob/main/LICENSE",
    "smplx": "SMPL-X license (non-commercial research), https://smpl-x.is.tue.mpg.de/modellicense.html",
    "mano": "MANO license (non-commercial research), https://mano.is.tue.mpg.de/license.html",
    "yolo": "AGPL-3.0 (Ultralytics), https://github.com/ultralytics/ultralytics/blob/main/LICENSE",
    "sapiens2": "Sapiens2 License, https://github.com/facebookresearch/sapiens2/blob/main/LICENSE.md",
    "rtmw": "Apache-2.0 (RTMW, OpenMMLab), https://github.com/open-mmlab/mmpose/blob/main/LICENSE",
    "sam3d": "SAM License, https://huggingface.co/facebook/sam-3d-body-dinov3/blob/main/LICENSE; "
    "DINOv3 code: DINOv3 License, https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md",
    "hamer": "MIT (HaMeR), https://github.com/geopavlakos/hamer/blob/main/LICENSE.md",
    "dpvo": "MIT (DPVO), https://github.com/princeton-vl/DPVO/blob/main/LICENSE",
}
DEFAULT_COMPONENTS = ["exmr", "smplx", "mano", "yolo", "sapiens2", "rtmw", "sam3d", "hamer"]


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_sha256(path, expected):
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            f"Checksum mismatch for {path}:\n  expected {expected}\n  got      {actual}"
        )


class Downloader:
    def __init__(self, root, verify):
        self.root, self.verify = Path(root), verify

    def todo(self, rel, sha256=None):
        """Whether root/rel still has to be fetched (with --verify: re-check present files)."""
        path = self.root / rel
        if not path.exists():
            return True
        if self.verify:
            check_sha256(path, sha256 or SHA256[rel])
            print(f"  ok       {rel}")
        else:
            print(f"  present  {rel}")
        return False

    def place(self, tmp, rel):
        """Check a fetched file (SHA256 table) and move it to root/rel."""
        check_sha256(tmp, SHA256[rel])
        dest = self.root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(tmp, dest)
        print(f"  fetched  {rel}")

    def url(self, url, rel):
        if not self.todo(rel):
            return
        with tempfile.TemporaryDirectory(dir=self.root) as tmp:
            tmp = Path(tmp) / "download"
            with _open(url) as response, open(tmp, "wb") as f:
                _copy(response, f, total=response.length, desc=rel)
            self.place(tmp, rel)

    def hf(self, repo, revision, filename, rel):
        """A file of a Hugging Face repo (uses the token of `hf auth login` for gated repos)."""
        if not self.todo(rel):
            return
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import GatedRepoError

        with tempfile.TemporaryDirectory(dir=self.root) as tmp:
            try:
                path = hf_hub_download(repo, filename, revision=revision, local_dir=tmp)
            except GatedRepoError:
                raise RuntimeError(
                    f"{repo} is gated: request access at https://huggingface.co/{repo}, wait for "
                    "the approval, then log in with `hf auth login`"
                ) from None
            self.place(path, rel)


def _open(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "exmr"}))


def _copy(src, dst, total, desc):
    with tqdm(
        total=total, unit="B", unit_scale=True, unit_divisor=1024, desc=desc, leave=False
    ) as bar:
        for chunk in iter(lambda: src.read(1 << 20), b""):
            dst.write(chunk)
            bar.update(len(chunk))


def fetch_exmr(d, args):
    d.hf(EXMR_REPO, EXMR_REVISION, "exmr.safetensors", "exmr/exmr.safetensors")
    # not needed to run ExMR; the Hugging Face Hub counts the downloads of this file
    d.hf(EXMR_REPO, EXMR_REVISION, "config.json", "exmr/config.json")


def fetch_rtmw(d, args):
    # FP16 conversion of RTMW-x (Models_Pose2Sim RTMW_x_lastest.onnx), see the ExMR model card
    d.hf(EXMR_REPO, EXMR_REVISION, "rtmw-x_384x288_fp16.onnx", "rtmw/rtmw-x_384x288_fp16.onnx")


def fetch_sapiens2(d, args):
    d.hf(SAPIENS2_REPO, SAPIENS2_REVISION, "sapiens2_1b_pose.safetensors",
         "sapiens2/sapiens2_1b_pose.safetensors")  # fmt: skip


def fetch_yolo(d, args):
    for name in ("yolo11x.pt", "yolo11m.pt"):
        d.url(YOLO_URL.format(name), f"yolo/{name}")


def fetch_sam3d(d, args):
    for name in ("model.ckpt", "model_config.yaml", "assets/mhr_model.pt"):
        d.hf(SAM3D_REPO, SAM3D_REVISION, name, f"sam-3d-body-dinov3/{name}")
    fetch_dinov3_code(args.verify)


def fetch_dinov3_code(verify):
    """Pre-fill torch.hub's cache with the pinned DINOv3 code: SAM 3D Body loads its backbone with
    torch.hub.load("facebookresearch/dinov3", ...), which otherwise fetches the latest code from
    GitHub on first use."""
    repo_dir = Path(torch.hub.get_dir()) / "facebookresearch_dinov3_main"
    if repo_dir.exists():
        if verify and _tree_sha256(repo_dir) != DINOV3_TREE_SHA256:
            raise RuntimeError(
                f"{repo_dir} holds another DINOv3 version than the tested commit "
                f"{DINOV3_COMMIT[:12]}; delete it and rerun"
            )
        print(f"  {'ok' if verify else 'present':8} {repo_dir}")
        return
    url = f"https://github.com/facebookresearch/dinov3/archive/{DINOV3_COMMIT}.zip"
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=repo_dir.parent) as tmp:
        with _open(url) as response:
            data = io.BytesIO(response.read())
        with zipfile.ZipFile(data) as z:
            z.extractall(tmp)
        src = Path(tmp) / f"dinov3-{DINOV3_COMMIT}"
        if _tree_sha256(src) != DINOV3_TREE_SHA256:
            raise RuntimeError(f"Unexpected content of {url}")
        shutil.move(src, repo_dir)
    print(f"  fetched  {repo_dir} (DINOv3 code @ {DINOV3_COMMIT[:12]})")


def _tree_sha256(root):
    digest = hashlib.sha256()
    for path in sorted(
        p for p in Path(root).rglob("*") if p.is_file() and "__pycache__" not in p.parts
    ):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def fetch_hamer(d, args):
    """HaMeR's demo data: streamed, only the HaMeR checkpoint and MANO mean parameters are kept
    (the archive's 3.8 GB of ViTPose weights are not needed)."""
    members = {
        f"_DATA/{rel.removeprefix('hamer/')}": rel
        for rel in SHA256
        if rel.startswith("hamer/") and d.todo(rel)
    }
    if not members:
        return
    with tempfile.TemporaryDirectory(dir=d.root) as tmp, _open(HAMER_URL) as response:
        with tqdm(total=response.length, unit="B", unit_scale=True, unit_divisor=1024,
                  desc="hamer_demo_data.tar.gz", leave=False) as bar:  # fmt: skip
            stream = _ProgressReader(response, bar)
            with tarfile.open(fileobj=stream, mode="r|gz") as tar:
                for member in tar:
                    rel = members.get(member.name)
                    if rel is None or not member.isfile():
                        continue
                    out = Path(tmp) / Path(rel).name
                    with tar.extractfile(member) as src, open(out, "wb") as f:
                        shutil.copyfileobj(src, f, 1 << 20)
                    d.place(out, rel)
                    del members[member.name]
    if members:
        raise RuntimeError(f"Not found in {HAMER_URL}: {sorted(members)}")


class _ProgressReader:
    def __init__(self, f, bar):
        self.f, self.bar = f, bar

    def read(self, n=-1):
        data = self.f.read(n)
        self.bar.update(len(data))
        return data


def fetch_dpvo(d, args):
    rel = "dpvo/dpvo.pth"
    if not d.todo(rel):
        return
    with tempfile.TemporaryDirectory(dir=d.root) as tmp, _open(DPVO_URL) as response:
        with zipfile.ZipFile(io.BytesIO(response.read())) as z:
            out = Path(z.extract("dpvo.pth", tmp))
        d.place(out, rel)


def fetch_smplx(d, args):
    rel = "body_models/smplx/SMPLX_NEUTRAL.npz"
    if not d.todo(rel):
        return
    if args.smplx_zip is None:
        raise RuntimeError(
            'SMPL-X needs registration: download models_smplx_v1_1.zip ("SMPL-X v1.1 (NPZ+PKL)") '
            "from https://smpl-x.is.tue.mpg.de and pass it with --smplx_zip"
        )
    with tempfile.TemporaryDirectory(dir=d.root) as tmp, zipfile.ZipFile(args.smplx_zip) as z:
        d.place(Path(z.extract(_member(z, "smplx/SMPLX_NEUTRAL.npz"), tmp)), rel)


def fetch_mano(d, args):
    rel = "hamer/data/mano/MANO_RIGHT.pkl"
    path = d.root / rel
    if path.exists():
        if args.verify:
            _check_mano(_load_mano(path), path)
        print(f"  {'ok' if args.verify else 'present':8} {rel}")
        return
    if args.mano_zip is None:
        raise RuntimeError(
            'MANO needs registration: download mano_v1_2.zip ("Models & Code") from '
            "https://mano.is.tue.mpg.de and pass it with --mano_zip"
        )
    with zipfile.ZipFile(args.mano_zip) as z:
        model = _load_mano(io.BytesIO(z.read(_member(z, "models/MANO_RIGHT.pkl"))))
    _check_mano(model, args.mano_zip)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"  fetched  {rel} (converted, without chumpy)")


def _member(z, suffix):
    names = [n for n in z.namelist() if n.endswith(suffix)]
    if len(names) != 1:
        raise RuntimeError(f"Expected one */{suffix} in {z.filename}, found {names}")
    return names[0]


class _ChumpyStub:
    """Stands in for chumpy objects (chumpy does not work on Python 3.12 / numpy 2)."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        self.state = state


class _ManoUnpickler(pickle.Unpickler):
    ALLOWED = {
        ("numpy", "ndarray"), ("numpy", "dtype"), ("numpy.core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "_reconstruct"), ("scipy.sparse.csc", "csc_matrix"),
        ("scipy.sparse._csc", "csc_matrix"), ("__builtin__", "set"), ("builtins", "set"),
        ("__builtin__", "object"), ("builtins", "object"), ("copy_reg", "_reconstructor"),
        ("copyreg", "_reconstructor"),
    }  # fmt: skip

    def find_class(self, module, name):
        if module in ("chumpy.ch", "chumpy.reordering"):
            return type(name, (_ChumpyStub,), {})
        if (module, name) not in self.ALLOWED:
            raise pickle.UnpicklingError(f"Unexpected object in the MANO model: {module}.{name}")
        if name == "csc_matrix":  # pickled under the deprecated module scipy.sparse.csc
            return scipy.sparse.csc_matrix
        return super().find_class(module, name)


def _load_mano(f):
    """MANO pickle -> dict of numpy arrays / scipy matrices / strings, chumpy objects resolved.

    In the official MANO_RIGHT.pkl, shapedirs is a chumpy Select (selection from a Ch array) and
    the other arrays are plain numpy; this only resolves what that file uses."""
    if isinstance(f, (str, Path)):
        with open(f, "rb") as fh:
            return _load_mano(fh)
    model = _ManoUnpickler(f, encoding="latin1").load()
    for key, value in model.items():
        if isinstance(value, _ChumpyStub):
            model[key] = _resolve_chumpy(value)
    return model


def _resolve_chumpy(obj):
    kind, state = type(obj).__name__, obj.state
    if kind == "Ch":
        return np.asarray(state["x"])
    if kind == "Select":
        a = _resolve_chumpy(state["a"])
        return a.ravel()[np.asarray(state["idxs"]).ravel()].reshape(state["preferred_shape"])
    raise RuntimeError(f"Unsupported chumpy object in the MANO model: {kind}")


def _mano_arrays_sha256(model):
    digest = hashlib.sha256()
    for key in sorted(model):
        value = model[key]
        if scipy.sparse.issparse(value):
            value = value.toarray()
        value = np.ascontiguousarray(value) if isinstance(value, np.ndarray) else str(value)
        digest.update(key.encode() + b"\0")
        if isinstance(value, np.ndarray):
            digest.update(f"{value.dtype.str}{value.shape}".encode() + value.tobytes())
        else:
            digest.update(value.encode())
    return digest.hexdigest()


def _check_mano(model, source):
    actual = _mano_arrays_sha256(model)
    if actual != MANO_RIGHT_ARRAYS_SHA256:
        raise RuntimeError(
            f"MANO_RIGHT.pkl from {source} differs from the tested model (MANO v1.2):\n"
            f"  expected {MANO_RIGHT_ARRAYS_SHA256}\n  got      {actual}"
        )


FETCH = {
    "exmr": fetch_exmr, "smplx": fetch_smplx, "mano": fetch_mano, "yolo": fetch_yolo,
    "sapiens2": fetch_sapiens2, "rtmw": fetch_rtmw, "sam3d": fetch_sam3d, "hamer": fetch_hamer,
    "dpvo": fetch_dpvo,
}  # fmt: skip


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help="target directory (default: $EXMR_CHECKPOINT_DIR or ./checkpoints)",
    )
    parser.add_argument("--only", nargs="+", choices=list(FETCH), help="only these components")
    parser.add_argument("--dpvo", action="store_true", help="also the optional DPVO weights")
    parser.add_argument("--smplx_zip", type=Path, help="models_smplx_v1_1.zip (registration)")
    parser.add_argument("--mano_zip", type=Path, help="mano_v1_2.zip (registration)")
    parser.add_argument("--verify", action="store_true", help="re-check files already present")
    args = parser.parse_args()

    components = args.only or DEFAULT_COMPONENTS + (["dpvo"] if args.dpvo else [])
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    d = Downloader(args.checkpoint_dir, args.verify)
    failed = []
    for name in components:
        print(f"[{name}] {LICENSES[name]}")
        try:
            FETCH[name](d, args)
        except Exception as e:  # report every failed component at the end
            print(f"  FAILED: {e}")
            failed.append(name)
    if failed:
        sys.exit(f"\nFailed: {', '.join(failed)} (see above); rerun to retry")
    print(f"\nAll model files are in {args.checkpoint_dir}")


if __name__ == "__main__":
    main()
