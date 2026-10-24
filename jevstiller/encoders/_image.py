"""Frozen image encoders: an ONNX vision tower (CPU) or a SigLIP one in PyTorch (GPU), both with the model's own
preprocessing redone in Pillow and NumPy.

Images come from callers, so nothing is decoded before `check_image` has read its header and found it inside Jev's
own limits (PNG or JPEG, still, 280-1,568 px a side, 78,400-2,458,624 px, 16 MiB): a request Jev would refuse is
never decoded here, and a decompression bomb can't be one Jev accepts. Needs `pip install "jevstiller[images]"`.
"""
from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Jev's limits for an image (docs.typesafe.ai/go/images, 2026-10-14)
MIN_SIDE, MAX_SIDE = 280, 1568
MIN_PIXELS, MAX_PIXELS = 78_400, 2_458_624
MAX_FILE_BYTES = 16 * 2**20
FORMATS = ("PNG", "JPEG")
# more than Jev's rules: a PNG's chunks before its pixel data, and its compressed text and colour profiles, which
# Pillow reads (and inflates, up to 1 MiB each) before a header check can answer. An image has a handful; 325,000
# empty chunks cost 0.7 s of CPU, and 63 zTXt chunks 64 MiB, per check (security audit run 5).
MAX_PNG_CHUNKS = 256
MAX_PNG_COMPRESSED = 8
_PNG = b"\x89PNG\r\n\x1a\n"


@dataclass(frozen=True)
class ImageCheck:
    ok: bool
    reason: str = ""                     # why not, when not ok
    width: int = 0
    height: int = 0
    format: str = ""


def _png_lean(data: bytes) -> bool:
    """Few chunks before the pixel data (MAX_PNG_CHUNKS), few of them compressed (MAX_PNG_COMPRESSED)."""
    pos, n, compressed = len(_PNG), 0, 0
    while pos + 8 <= len(data):
        kind = data[pos + 4:pos + 8]
        if kind == b"IDAT":
            return True
        n += 1
        compressed += kind in (b"zTXt", b"iTXt", b"iCCP")
        if n > MAX_PNG_CHUNKS or compressed > MAX_PNG_COMPRESSED:
            return False
        pos += 12 + int.from_bytes(data[pos:pos + 4], "big")
    return True                                         # no pixel data: Pillow refuses it


def check_image(data: bytes) -> ImageCheck:
    """Whether Jev would take this image, from its header only (nothing is decoded). PNG and JPEG are the only
    formats Pillow tries; PNGs with an unusual number of chunks before their pixel data are refused too."""
    from PIL import Image, UnidentifiedImageError
    if not data:
        return ImageCheck(False, "empty")
    if len(data) > MAX_FILE_BYTES:
        return ImageCheck(False, "file too large")
    if data.startswith(_PNG) and not _png_lean(data):
        return ImageCheck(False, "too many chunks", format="PNG")
    try:
        with Image.open(io.BytesIO(data), formats=FORMATS) as im:   # reads the header; pixels stay undecoded
            fmt, (w, h) = im.format or "", im.size
            animated = bool(getattr(im, "is_animated", False)) or getattr(im, "n_frames", 1) > 1
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        return ImageCheck(False, "not an image")
    if fmt not in FORMATS:
        return ImageCheck(False, f"format {fmt or 'unknown'}", w, h, fmt)
    if animated:
        return ImageCheck(False, "animated", w, h, fmt)
    if not (MIN_SIDE <= w <= MAX_SIDE and MIN_SIDE <= h <= MAX_SIDE):
        return ImageCheck(False, "side out of range", w, h, fmt)
    if not MIN_PIXELS <= w * h <= MAX_PIXELS:
        return ImageCheck(False, "pixels out of range", w, h, fmt)
    return ImageCheck(True, "", w, h, fmt)


class _Preprocess:
    """A model's `preprocessor_config.json` (resize, optional centre crop, rescale, normalise), done the way the
    transformers processors do it with Pillow, without transformers or torchvision."""

    def __init__(self, cfg: dict):
        size = cfg.get("size", {})
        if "shortest_edge" in size:                   # CLIP: shorter side to N, then a centre crop
            self.resize_short, self.resize_hw = int(size["shortest_edge"]), None
        else:                                         # SigLIP: straight to H x W
            self.resize_short, self.resize_hw = None, (int(size["height"]), int(size["width"]))
        crop = cfg.get("crop_size") if cfg.get("do_center_crop") else None
        self.crop_hw = (int(crop["height"]), int(crop["width"])) if crop else None
        self.resample = int(cfg.get("resample", 3))
        self.scale = float(cfg.get("rescale_factor", 1 / 255)) if cfg.get("do_rescale", True) else 1.0
        norm = cfg.get("do_normalize", True)
        self.mean = np.array(cfg.get("image_mean", [0, 0, 0]) if norm else [0, 0, 0], np.float32)[:, None, None]
        self.std = np.array(cfg.get("image_std", [1, 1, 1]) if norm else [1, 1, 1], np.float32)[:, None, None]
        self.hash = hashlib.sha256(json.dumps([self.resize_short, self.resize_hw, self.crop_hw, self.resample,
                                               self.scale, self.mean.ravel().tolist(), self.std.ravel().tolist()]
                                              ).encode()).hexdigest()[:6]

    def __call__(self, data: bytes) -> np.ndarray:
        """(3, H, W) float32, as the model's own processor makes it. Call `check_image` first."""
        from PIL import Image
        with Image.open(io.BytesIO(data), formats=FORMATS) as im:
            im = im.convert("RGB")
            if self.resize_hw is not None:
                im = im.resize((self.resize_hw[1], self.resize_hw[0]), resample=self.resample)
            else:
                w, h = im.size
                s = self.resize_short
                nw, nh = (s, int(s * h / w)) if w <= h else (int(s * w / h), s)
                im = im.resize((nw, nh), resample=self.resample)
            if self.crop_hw is not None:
                ch, cw = self.crop_hw
                w, h = im.size
                top, left = (h - ch) // 2, (w - cw) // 2
                im = im.crop((left, top, left + cw, top + ch))
            x = np.asarray(im, dtype=np.float32).transpose(2, 0, 1) * self.scale
        return (x - self.mean) / self.std


def _preprocessor(repo: str) -> _Preprocess:
    from huggingface_hub import hf_hub_download
    return _Preprocess(json.loads(Path(hf_hub_download(repo, "preprocessor_config.json")).read_text()))


def _no_text(items: Sequence) -> list:
    if any(isinstance(x, str) for x in items):
        raise TypeError("an image encoder embeds image bytes, not text")
    return list(items)


class OnnxImageEncoder:
    """Embeds images (raw PNG/JPEG bytes) with an ONNX vision model from the Hugging Face Hub.

    Preprocessing follows the model's `preprocessor_config.json` (resize, optional centre crop, rescale, normalise),
    the same steps transformers takes with Pillow, so stored and served embeddings agree with the model's own.
    `id` covers the model file and the preprocessing: changing either starts a new lineage."""

    def __init__(self, repo: str = "Xenova/siglip-base-patch16-224", file: str = "onnx/vision_model.onnx",
                 device: str = "auto", batch_size: int = 16, threads: int = 0):
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        self.repo, self.file, self.batch_size = repo, file, batch_size
        model_path = Path(hf_hub_download(repo, file))
        for extra in (f"{file}_data", f"{file}.data"):  # large models keep their weights beside the graph
            try:
                hf_hub_download(repo, extra)
            except Exception:
                pass
        self.preprocess = _preprocessor(repo)
        avail = ort.get_available_providers()
        cpu = ["CPUExecutionProvider"]
        want = ["CUDAExecutionProvider", *cpu] if device in ("auto", "cuda") else cpu
        providers = [p for p in want if p in avail] or ["CPUExecutionProvider"]
        so = ort.SessionOptions()
        so.log_severity_level = 3
        if threads:
            so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(model_path), so, providers=providers)
        self.device = "cuda" if providers[0].startswith("CUDA") else "cpu"
        self.input_name = self.sess.get_inputs()[0].name
        outs = [o.name for o in self.sess.get_outputs()]
        self.output_name = next((o for o in ("image_embeds", "pooler_output") if o in outs), outs[0])
        h = hashlib.sha256()
        with open(model_path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        self.id = f"onnx-image:{repo}/{file}:{self.output_name}:{self.preprocess.hash}:{h.hexdigest()[:8]}"
        self.dim = int(self.sess.get_outputs()[outs.index(self.output_name)].shape[-1])

    def encode_images(self, images: Sequence[bytes]) -> np.ndarray:
        """(n, dim) float32, L2-normalised. Call `check_image` first."""
        chunks = []
        for s in range(0, len(images), self.batch_size):
            batch = np.stack([self.preprocess(d) for d in images[s:s + self.batch_size]])
            e = self.sess.run([self.output_name], {self.input_name: batch})[0].astype(np.float32)
            if e.ndim == 3:                     # a model without a pooled output (DINOv2): its first token,
                e = e[:, 0]                     # which is what its pooler returns
            chunks.append(e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12))
        return np.concatenate(chunks) if chunks else np.zeros((0, self.dim), np.float32)

    def encode(self, items: Sequence[bytes]) -> np.ndarray:
        """The Encoder interface, for images: `items` are image file bytes, never text."""
        return self.encode_images(_no_text(items))


class TorchImageEncoder:
    """The GPU path for images, as TorchEncoder is for text: a Hugging Face SigLIP vision tower in PyTorch, fp16
    autocast on CUDA, with the same Pillow/NumPy preprocessing as OnnxImageEncoder (transformers 5's own image
    processors need torchvision). Needs `pip install "jevstiller[gpu,images]"`."""

    def __init__(self, model_name: str = "google/siglip-so400m-patch14-384", device: str = "auto",
                 batch_size: int = 16):
        import torch
        from transformers import SiglipVisionModel
        from transformers.utils import logging as hf_logging
        self.torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device, self.batch_size, self.model_name = device, batch_size, model_name
        self.preprocess = _preprocessor(model_name)
        level = hf_logging.get_verbosity()
        hf_logging.set_verbosity_error()      # the checkpoint's unused text tower would be reported, key by key
        try:
            self.model = SiglipVisionModel.from_pretrained(model_name, use_safetensors=True).to(device).eval()
        finally:
            hf_logging.set_verbosity(level)
        self.dim = int(self.model.config.hidden_size)
        n_params = sum(p.numel() for p in self.model.parameters())
        h = hashlib.sha256((self.model.config.to_json_string() + str(n_params)).encode()).hexdigest()[:8]
        self.id = f"torch-image:{model_name}:{self.preprocess.hash}:{h}"

    def encode_images(self, images: Sequence[bytes]) -> np.ndarray:
        """(n, dim) float32, L2-normalised. Call `check_image` first."""
        torch = self.torch
        out = np.empty((len(images), self.dim), dtype=np.float32)
        with torch.inference_mode():
            for s in range(0, len(images), self.batch_size):
                batch = np.stack([self.preprocess(d) for d in images[s:s + self.batch_size]])
                x = torch.from_numpy(batch).to(self.device)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.device.startswith("cuda")):
                    e = self.model(pixel_values=x).pooler_output
                e = torch.nn.functional.normalize(e.float(), dim=-1)
                out[s:s + len(e)] = e.cpu().numpy()
        return out

    def encode(self, items: Sequence[bytes]) -> np.ndarray:
        """The Encoder interface, for images: `items` are image file bytes, never text."""
        return self.encode_images(_no_text(items))
