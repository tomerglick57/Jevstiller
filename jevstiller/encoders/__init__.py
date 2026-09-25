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


__all__ = ["Encoder", "HashEncoder", "BatchingEncoder", "load_encoder", "TIERS"]
