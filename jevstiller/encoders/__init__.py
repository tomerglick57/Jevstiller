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


from .hashing import HashEncoder  # noqa: E402

# tier -> (torch checkpoint, onnx repo)
TIERS = {
    "small": ("BAAI/bge-small-en-v1.5", "Xenova/bge-small-en-v1.5"),
    "base": ("BAAI/bge-base-en-v1.5", "Xenova/bge-base-en-v1.5"),
    "large": ("BAAI/bge-large-en-v1.5", "Xenova/bge-large-en-v1.5"),
}


def load_encoder(spec: str = "base", backend: str = "auto", device: str = "auto", **kw) -> Encoder:
    """spec: 'hash' | 'hash:<dim>' | a tier name | 'torch:<hf model>' | 'onnx:<hf repo>'."""
    if spec == "hash" or spec.startswith("hash:"):
        dim = int(spec.split(":")[1]) if ":" in spec else 512
        return HashEncoder(dim=dim, **kw)
    if spec.startswith("torch:"):
        from .hf import TorchEncoder
        return TorchEncoder(spec[6:], device=device, **kw)
    if spec.startswith("onnx:"):
        from .onnx import OnnxEncoder
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
            from .hf import TorchEncoder
            return TorchEncoder(torch_name, device=device, **kw)
        from .onnx import OnnxEncoder
        return OnnxEncoder(onnx_repo, device=device, **kw)
    raise ValueError(f"unknown encoder spec {spec!r}")


__all__ = ["Encoder", "HashEncoder", "load_encoder", "TIERS"]
