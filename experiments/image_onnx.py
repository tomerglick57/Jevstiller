"""The product's image encoder (jevstiller.encoders._image, ONNX Runtime, preprocessing in Pillow and NumPy) against
the PyTorch embeddings the image benchmark used (image_embed.py), and its cost per image on CPU.

    python3 experiments/image_onnx.py --dataset food101 --n 300

For each model: the cosine between its ONNX embedding and the reference one for the same image (same weights should
give ~1.0), and the time per image on CPU with 4 threads, header check, decoding and preprocessing included, one
image at a time and in batches of 16.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from image_data import ROOT as DATA  # noqa: E402
from image_data import iter_images  # noqa: E402

from jevstiller.encoders._image import OnnxImageEncoder, check_image  # noqa: E402

MODELS = [  # (name, repo, file, the reference embeddings to compare with)
    ("siglip-b16", "Xenova/siglip-base-patch16-224", "onnx/vision_model.onnx", "siglip-b16"),
    ("siglip-b16-int8", "Xenova/siglip-base-patch16-224", "onnx/vision_model_int8.onnx", "siglip-b16"),
    ("clip-b32-openai", "Xenova/clip-vit-base-patch32", "onnx/vision_model.onnx", None),
    ("clip-b32-openai-int8", "Xenova/clip-vit-base-patch32", "onnx/vision_model_int8.onnx", None),
    ("siglip-b16-384", "Xenova/siglip-base-patch16-384", "onnx/vision_model.onnx", None),
    ("siglip-l16-256", "Xenova/siglip-large-patch16-256", "onnx/vision_model.onnx", None),
    ("siglip2-b16", "onnx-community/siglip2-base-patch16-224-ONNX", "onnx/vision_model.onnx", None),
    ("dinov2-b", "Xenova/dinov2-base", "onnx/model.onnx", None),
    ("dinov3-s", "onnx-community/dinov3-vits16-pretrain-lvd1689m-ONNX", "onnx/model.onnx", None),
    ("dinov3-b", "onnx-community/dinov3-vitb16-pretrain-lvd1689m-ONNX", "onnx/model.onnx", None),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="food101")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--only", default="", help="comma-separated model names (default: all)")
    a = ap.parse_args()
    ids, images = [], []
    for i, data, _ in iter_images(a.dataset, decode=False):
        ids.append(i)
        images.append(data)
        if len(ids) == a.n:
            break
    checks = [check_image(d) for d in images]
    out = {"dataset": a.dataset, "n": len(ids), "jev_would_refuse": sum(not c.ok for c in checks), "models": {}}
    first: dict[str, np.ndarray] = {}
    for name, repo, file, ref in MODELS:
        if a.only and name not in a.only.split(","):
            continue
        try:
            enc = OnnxImageEncoder(repo, file, device="cpu", threads=4)
        except Exception as e:                       # e.g. an int8 graph whose ops this ONNX Runtime lacks on CPU
            out["models"][name] = {"error": f"{type(e).__name__}: {str(e)[:160]}"}
            print(name, "does not load:", out["models"][name]["error"], flush=True)
            continue
        X = enc.encode_images(images)
        r = {"id": enc.id, "dim": enc.dim}
        if ref:
            z = np.load(DATA / a.dataset / f"emb-{ref}.npz")
            idx = {k: j for j, k in enumerate(z["ids"].tolist())}
            R = z["X"][[idx[i] for i in ids]]
            cos = (X * R).sum(1)
            r["cosine_to_pytorch"] = {"mean": round(float(cos.mean()), 5), "min": round(float(cos.min()), 5)}
        base = first.setdefault(repo, X)
        if base is not X:
            cos = (X * base).sum(1)
            r["cosine_to_fp32_onnx"] = {"mean": round(float(cos.mean()), 5), "min": round(float(cos.min()), 5)}
        enc.encode_images(images[:2])                                     # warm-up
        t0 = time.perf_counter()
        for d in images[:64]:
            check_image(d)
            enc.encode_images([d])
        r["ms_per_image_batch1"] = round((time.perf_counter() - t0) * 1000 / 64, 1)
        t0 = time.perf_counter()
        for s in range(0, 64, 16):
            [check_image(d) for d in images[s:s + 16]]
            enc.encode_images(images[s:s + 16])
        r["ms_per_image_batch16"] = round((time.perf_counter() - t0) * 1000 / 64, 1)
        out["models"][name] = r
        print(name, json.dumps(r), flush=True)
    (ROOT / "experiments" / "results").mkdir(exist_ok=True)
    (ROOT / "experiments" / "results" / f"image-onnx-{a.dataset}.json").write_text(json.dumps(out, indent=1))
    print("Jev would refuse", out["jev_would_refuse"], "of", out["n"], "as stored (before resizing to its limits)")


if __name__ == "__main__":
    main()
