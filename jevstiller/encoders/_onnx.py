"""ONNX Runtime encoder. One artifact for CPU / CUDA / ARM; picks the best available provider."""
from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ._hf import default_pooling, length_batches


class OnnxEncoder:
    def __init__(self, repo: str = "Xenova/bge-small-en-v1.5", file: str = "onnx/model.onnx",
                 device: str = "auto", pooling: str | None = None, max_length: int = 256,
                 batch_size: int | None = None):
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer
        self.repo, self.file = repo, file
        self.pooling = pooling or default_pooling(repo)
        self.max_length = max_length
        model_path = Path(hf_hub_download(repo, file))
        for extra in ("onnx/model.onnx_data", "onnx/model.onnx.data"):   # large models keep weights beside the graph
            try:
                hf_hub_download(repo, extra)
            except Exception:
                pass
        self.tok = Tokenizer.from_file(hf_hub_download(repo, "tokenizer.json"))
        self.tok.enable_truncation(max_length)
        self.tok.enable_padding()
        avail = ort.get_available_providers()
        cpu = ["CPUExecutionProvider"]
        want = ["CUDAExecutionProvider", *cpu] if device in ("auto", "cuda") else cpu
        providers = [p for p in want if p in avail] or ["CPUExecutionProvider"]
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(str(model_path), so, providers=providers)
        self.device = "cuda" if providers[0].startswith("CUDA") else "cpu"
        # On CPU, length-sorted batches of 8 beat 16 and 64 on real text at 16-256 concurrent callers (bge-small,
        # benchmarks/encoders.py, docs/benchmarks.md); a GPU wants bigger batches
        self.batch_size = batch_size or (64 if self.device == "cuda" else 8)
        self.input_names = [i.name for i in self.sess.get_inputs()]
        self.output_name = self.sess.get_outputs()[0].name
        h = hashlib.sha256()
        with open(model_path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        self.id = f"onnx:{repo}/{file}:{self.pooling}:{max_length}:{h.hexdigest()[:8]}"
        self.dim = int(self.encode(["probe"]).shape[1]) if not hasattr(self, "dim") else self.dim

    def _run(self, texts: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        enc = self.tok.encode_batch(list(texts))
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self.input_names:
            feed["token_type_ids"] = np.zeros_like(ids)
        h = self.sess.run([self.output_name], feed)[0]
        return h, mask

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = None
        for idx in length_batches(texts, self.batch_size):
            h, mask = self._run([texts[i] for i in idx])
            if self.pooling == "cls":
                e = h[:, 0]
            else:
                m = mask[..., None].astype(h.dtype)
                e = (h * m).sum(1) / np.maximum(m.sum(1), 1e-6)
            e = e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)
            if out is None:
                out = np.empty((len(texts), e.shape[1]), np.float32)
            out[idx] = e
        if out is None:
            out = np.zeros((0, getattr(self, "dim", 0)), np.float32)
        if not hasattr(self, "dim"):
            self.dim = int(out.shape[1])
        return out
