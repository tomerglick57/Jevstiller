"""PyTorch encoder over Hugging Face checkpoints. CUDA if available."""
from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np


def default_pooling(model_name: str) -> str:
    return "cls" if "bge" in model_name.lower() else "mean"


class TorchEncoder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", device: str = "auto",
                 pooling: str | None = None, max_length: int = 256, batch_size: int = 64):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.model_name = model_name
        self.pooling = pooling or default_pooling(model_name)
        self.max_length, self.batch_size = max_length, batch_size
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.dim = int(self.model.config.hidden_size)
        n_params = sum(p.numel() for p in self.model.parameters())
        h = hashlib.sha256((self.model.config.to_json_string() + str(n_params)).encode()).hexdigest()[:8]
        self.id = f"torch:{model_name}:{self.pooling}:{max_length}:{h}"

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        torch = self.torch
        out = np.empty((len(texts), self.dim), dtype=np.float32)
        with torch.inference_mode():
            for s in range(0, len(texts), self.batch_size):
                b = self.tok(list(texts[s:s + self.batch_size]), padding=True, truncation=True,
                             max_length=self.max_length, return_tensors="pt").to(self.device)
                with torch.autocast(device_type="cuda", enabled=self.device.startswith("cuda")):
                    h = self.model(**b).last_hidden_state
                if self.pooling == "cls":
                    e = h[:, 0]
                else:
                    m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
                    e = (h * m).sum(1) / m.sum(1).clamp(min=1e-6)
                e = torch.nn.functional.normalize(e.float(), dim=-1)
                out[s:s + len(e)] = e.cpu().numpy()
        return out
