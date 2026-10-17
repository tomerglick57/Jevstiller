"""Embed a dataset's images with a frozen image encoder: one .npz of (ids, L2-normalised embeddings) per dataset and
encoder, next to the images (image_data.ROOT/<dataset>/emb-<encoder>.npz).

    python3 experiments/image_embed.py --dataset crack --encoder clip-b32

Also times the encoder on CPU (PyTorch, not ONNX Runtime) on 64 images: one at a time with 4 threads, and in
batches of 32 with every core. That is roughly what a CPU deployment would pay per image, decoding included.
Weights are loaded from safetensors files only.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from image_data import ROOT, iter_images  # noqa: E402

ENCODERS = {
    "clip-b32": ("open_clip", "hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K"),   # 88M-parameter vision tower, 224 px
    "siglip-b16": ("hf", "google/siglip-base-patch16-224"),                      # 93M, 224 px, 4x the patches
    "dinov2-s": ("hf", "facebook/dinov2-small"),                                 # 22M, 224 px, no text tower
}


def build(name: str, device: str):
    """(preprocess: PIL image -> tensor, embed: batch tensor -> (n, dim) normalised numpy)."""
    import torch
    kind, ref = ENCODERS[name]
    if kind == "open_clip":
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(ref)    # prefers open_clip_model.safetensors
        model = model.eval().to(device)
        forward = model.encode_image
    else:
        from transformers import AutoImageProcessor, AutoModel, SiglipVisionModel
        proc = AutoImageProcessor.from_pretrained(ref)
        cls = SiglipVisionModel if "siglip" in ref else AutoModel    # SigLIP: the vision tower and its pooling head
        model = cls.from_pretrained(ref, use_safetensors=True).eval().to(device)

        def preprocess(im):
            return proc(images=im, return_tensors="pt")["pixel_values"][0]

        def forward(x):
            return model(pixel_values=x).pooler_output

    @torch.no_grad()
    def embed(x):
        x = x.to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
            f = forward(x).float()
        f = torch.nn.functional.normalize(f, dim=-1)
        return f.cpu().numpy().astype(np.float32)
    return preprocess, embed


def cpu_timing(name: str, images: list) -> dict:
    import torch
    out = {}
    for threads, batch in ((4, 1), (torch.get_num_threads(), 32)):
        torch.set_num_threads(threads)
        preprocess, embed = build(name, "cpu")
        embed(torch.stack([preprocess(images[0])]))                         # warm-up
        t0 = time.perf_counter()
        for s in range(0, len(images), batch):
            embed(torch.stack([preprocess(im) for im in images[s:s + batch]]))
        out[f"ms_per_image_threads{threads}_batch{batch}"] = round((time.perf_counter() - t0) * 1000 / len(images), 1)
    return out


def main() -> None:
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--encoder", required=True, choices=list(ENCODERS))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=256)
    a = ap.parse_args()
    out = ROOT / a.dataset / f"emb-{a.encoder}.npz"
    preprocess, embed = build(a.encoder, a.device)
    ids, chunks, sample = [], [], []
    t0 = time.perf_counter()
    with ThreadPoolExecutor(8) as pool:                  # decoding and resizing in threads keep the GPU busy
        batch_ids, batch_ims = [], []
        for i, im, _ in iter_images(a.dataset):
            batch_ids.append(i)
            batch_ims.append(im)
            if len(sample) < 64:
                sample.append(im)
            if len(batch_ims) == a.batch:
                chunks.append(embed(torch.stack(list(pool.map(preprocess, batch_ims)))))
                ids += batch_ids
                batch_ids, batch_ims = [], []
        if batch_ims:
            chunks.append(embed(torch.stack(list(pool.map(preprocess, batch_ims)))))
            ids += batch_ids
    X = np.concatenate(chunks)
    dt = time.perf_counter() - t0
    np.savez(out, ids=np.array(ids), X=X)
    info = {"dataset": a.dataset, "encoder": a.encoder, "ref": ENCODERS[a.encoder][1], "n": len(ids), "dim": X.shape[1],
            "device": a.device, "seconds": round(dt, 1), "images_per_s": round(len(ids) / dt, 1),
            "cpu": cpu_timing(a.encoder, sample)}
    out.with_suffix(".json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info))


if __name__ == "__main__":
    main()
