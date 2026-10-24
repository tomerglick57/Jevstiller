"""Frozen text encoders behind one interface."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Encoder(Protocol):
    id: str      # name + content hash; changing it starts a new lineage
    dim: int

    def encode(self, texts: Sequence[str]) -> np.ndarray:  # (n, dim) float32, L2-normalised
        ...


from ._batching import BatchingEncoder  # noqa: E402
from ._hashing import HashEncoder  # noqa: E402

# tier -> (torch checkpoint, onnx repo)
TIERS = {
    "small": ("BAAI/bge-small-en-v1.5", "Xenova/bge-small-en-v1.5"),
    "base": ("BAAI/bge-base-en-v1.5", "Xenova/bge-base-en-v1.5"),
    "large": ("BAAI/bge-large-en-v1.5", "Xenova/bge-large-en-v1.5"),
}


_HINT = {
    # part of `pip install jevstiller` where ONNX Runtime ships wheels; missing means no build for this platform
    "onnxruntime": "ONNX Runtime isn't installed (it has no build for every platform). Use PyTorch instead "
                   "(pip install \"jevstiller[gpu]\"), or the hash encoder (--encoder hash), which needs nothing",
    "torch": "the PyTorch encoders need pip install \"jevstiller[gpu]\"",
    "PIL": "questions about images need pip install \"jevstiller[images]\" (Pillow)",
    "transformers": "the PyTorch encoders need pip install \"jevstiller[gpu]\"",
}


def _missing(err: ModuleNotFoundError) -> Exception:
    """A missing optional package, as an error that says what to install."""
    hint = _HINT.get((err.name or "").split(".")[0])
    return ImportError(f"this encoder needs {err.name}: {hint}") if hint else err


def load_encoder(spec: str = "base", backend: str = "auto", device: str = "auto", **kw) -> Encoder:
    """spec: 'hash' | 'hash:<dim>' | a tier name | 'torch:<hf model>' | 'onnx:<hf repo>'.

    The ONNX encoders come with `pip install jevstiller` (where ONNX Runtime has a build); the PyTorch ones need
    `jevstiller[gpu]`. A missing backend raises an ImportError that says what to install."""
    try:
        return _load(spec, backend, device, **kw)
    except ModuleNotFoundError as e:
        raise _missing(e) from None


def _load(spec: str, backend: str, device: str, **kw) -> Encoder:
    if spec == "hash" or spec.startswith("hash:"):
        dim = int(spec.split(":")[1]) if ":" in spec else 512
        return HashEncoder(dim=dim, **kw)
    if spec.startswith("torch:"):
        from ._hf import TorchEncoder
        return TorchEncoder(spec[6:], device=device, **kw)
    if spec.startswith("onnx:"):
        from ._onnx import OnnxEncoder
        return OnnxEncoder(spec[5:], device=device, **kw)
    if spec in TIERS:
        torch_name, onnx_repo = TIERS[spec]
        if backend == "auto":
            try:
                import torch  # noqa: F401
                backend = "torch" if torch.cuda.is_available() and device in ("auto", "cuda") else "onnx"
            except ImportError:
                backend = "onnx"
        if backend == "torch":
            from ._hf import TorchEncoder
            return TorchEncoder(torch_name, device=device, **kw)
        from ._onnx import OnnxEncoder
        return OnnxEncoder(onnx_repo, device=device, **kw)
    raise ValueError(f"unknown encoder spec {spec!r}")


# image encoders by name: their PyTorch checkpoint (the GPU path) and their ONNX export (the CPU path), None where there
# is none. Measured against Jev's answers in experiments/README.md ("Images: which encoder"); so400m is the best overall
AUTO_IMAGE_GPU, AUTO_IMAGE_CPU = "siglip-so400m", "siglip"
IMAGE_MODELS = {
    "siglip": {"torch": "google/siglip-base-patch16-224",
               "onnx": ("Xenova/siglip-base-patch16-224", "onnx/vision_model.onnx")},
    "siglip-384": {"torch": "google/siglip-base-patch16-384",
                   "onnx": ("Xenova/siglip-base-patch16-384", "onnx/vision_model.onnx")},
    "siglip-large": {"torch": "google/siglip-large-patch16-256",
                     "onnx": ("Xenova/siglip-large-patch16-256", "onnx/vision_model.onnx")},
    "siglip-so400m": {"torch": "google/siglip-so400m-patch14-384", "onnx": None},
    "clip": {"torch": None, "onnx": ("Xenova/clip-vit-base-patch32", "onnx/vision_model.onnx")},
}


def image_choice(spec: str, backend: str = "auto", cuda: bool = False, torch: bool = False):
    """(name, backend, weights) for an image encoder spec, without loading anything.

    `auto`: the best overall model on a GPU (so400m, through PyTorch), SigLIP B/16 otherwise (ONNX on CPU), as text
    picks PyTorch on a GPU and ONNX elsewhere. A named model takes PyTorch on a GPU when it has a checkpoint, else its
    ONNX export. `onnx:<repo>`: that repo's onnx/vision_model.onnx."""
    if spec.startswith("onnx:") and len(spec) > 5:
        return spec, "onnx", (spec[5:], "onnx/vision_model.onnx")
    gpu = cuda and torch and backend in ("auto", "torch")
    if spec == "auto":
        spec = AUTO_IMAGE_GPU if gpu else AUTO_IMAGE_CPU
    if spec not in IMAGE_MODELS:
        raise ValueError(f"unknown image encoder {spec!r}: auto, {', '.join(IMAGE_MODELS)} or onnx:<repo>")
    m = IMAGE_MODELS[spec]
    if backend == "auto":
        backend = "torch" if gpu and m["torch"] else "onnx" if m["onnx"] else "torch"
    if backend == "torch" and not m["torch"]:
        raise ValueError(f"image encoder {spec!r} has no PyTorch checkpoint here: use backend onnx")
    if backend == "onnx" and not m["onnx"]:
        raise ValueError(f"image encoder {spec!r} has no ONNX export: it needs PyTorch (pip install "
                         f"\"jevstiller[gpu,images]\"), and a GPU to be faster than Jev")
    return spec, backend, m[backend]


def load_image_encoder(spec: str = "auto", backend: str = "auto", device: str = "auto", **kw):
    """An image encoder (`image_choice`). Needs `jevstiller[images]`, and `jevstiller[gpu]` for PyTorch."""
    cuda = has_torch = False
    if device in ("auto", "cuda") and backend in ("auto", "torch"):
        try:
            import torch
            has_torch, cuda = True, torch.cuda.is_available()
        except ImportError:
            pass
    name, backend, weights = image_choice(spec, backend, cuda, has_torch)
    try:
        import PIL  # noqa: F401
        if backend == "torch":
            from ._image import TorchImageEncoder
            enc = TorchImageEncoder(weights, device="cuda" if cuda and device != "cpu" else "cpu", **kw)
        else:
            from ._image import OnnxImageEncoder
            enc = OnnxImageEncoder(*weights, device=device, **kw)
    except ModuleNotFoundError as e:
        raise _missing(e) from None
    enc.name = name
    return enc


__all__ = ["Encoder", "HashEncoder", "BatchingEncoder", "load_encoder", "load_image_encoder", "image_choice",
           "IMAGE_MODELS", "TIERS"]
