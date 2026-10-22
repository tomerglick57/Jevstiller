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
    "siglip-b16-384": ("hf", "google/siglip-base-patch16-384"),                  # the same at 384 px: more detail
    "siglip-l16-256": ("hf", "google/siglip-large-patch16-256"),                 # 316M, 256 px
    "siglip-so400m": ("hf", "google/siglip-so400m-patch14-384"),                 # 428M, 384 px: GPU-class
    "dinov2-b": ("hf", "facebook/dinov2-base"),                                  # 86M, 224 px
    # the product's own encoder (jevstiller.encoders._image), ONNX Runtime on CPU, from the raw image bytes
    "clip-b32-openai": ("onnx", ("Xenova/clip-vit-base-patch32", "onnx/vision_model.onnx")),
    "siglip2-b16": ("onnx", ("onnx-community/siglip2-base-patch16-224-ONNX", "onnx/vision_model.onnx")),
    # DINOv3 (Meta, DINOv3 License: accepted by the user for testing only, 2026-10-18)
    "dinov3-s": ("onnx", ("onnx-community/dinov3-vits16-pretrain-lvd1689m-ONNX", "onnx/model.onnx")),
    "dinov3-b": ("onnx", ("onnx-community/dinov3-vitb16-pretrain-lvd1689m-ONNX", "onnx/model.onnx")),
}


def build(name: str, device: str):
    """(preprocess: PIL image -> tensor, embed: batch tensor -> (n, dim) normalised numpy)."""
    import torch
    kind, ref = ENCODERS[name]
    if kind == "onnx":
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from jevstiller.encoders._image import OnnxImageEncoder
        enc = OnnxImageEncoder(*ref, device="cpu")

        def embed_bytes(items):
            return enc.encode_images(list(items))
        return None, embed_bytes
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
    if ENCODERS[name][0] == "onnx":                      # measured by experiments/image_onnx.py
        return {}
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
    ap.add_argument("--only-cached", default=None, metavar="JEV_MODEL",
                    help="embed only the images with this Jev model's recorded answer (image_jev.cache_path)")
    a = ap.parse_args()
    only = None
    if a.only_cached:
        from image_jev import cache_path
        only = {json.loads(line)["key"].split("\t", 1)[1] for line in open(cache_path(a.dataset, a.only_cached))}
    out = ROOT / a.dataset / f"emb-{a.encoder}.npz"
    preprocess, embed = build(a.encoder, a.device)
    ids, chunks, sample = [], [], []
    t0 = time.perf_counter()
    raw = preprocess is None                              # the ONNX path embeds the image files themselves
    with ThreadPoolExecutor(8) as pool:                  # decoding and resizing in threads keep the GPU busy
        batch_ids, batch_ims = [], []
        for i, im, _ in iter_images(a.dataset, decode=not raw):
            if only is not None and i not in only:
                continue
            batch_ids.append(i)
            batch_ims.append(im)
            if len(sample) < 64:
                sample.append(im)
            if len(batch_ims) == a.batch:
                chunks.append(embed(batch_ims) if raw else embed(torch.stack(list(pool.map(preprocess, batch_ims)))))
                ids += batch_ids
                batch_ids, batch_ims = [], []
        if batch_ims:
            chunks.append(embed(batch_ims) if raw else embed(torch.stack(list(pool.map(preprocess, batch_ims)))))
            ids += batch_ids
    X = np.concatenate(chunks)
    dt = time.perf_counter() - t0
    np.savez(out, ids=np.array(ids), X=X)
    info = {"dataset": a.dataset, "encoder": a.encoder, "ref": ENCODERS[a.encoder][1], "n": len(ids), "dim": X.shape[1],
            "device": a.device, "seconds": round(dt, 1), "images_per_s": round(len(ids) / dt, 1),
            "cpu": cpu_timing(a.encoder, sample) if not raw else {}}
    out.with_suffix(".json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info))


if __name__ == "__main__":
    main()
