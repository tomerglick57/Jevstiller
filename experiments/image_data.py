"""Public image datasets for the image benchmarks, shaped like the text ones in datasets.py.

    python3 experiments/image_data.py fetch food101      # download (idempotent)
    python3 experiments/image_data.py sizes food101      # image size stats (what Jev would be sent)

Each dataset is one repeated question about many images, as an adopter would ask Jev:
- food101: what dish is in a photo (101 classes; Food-101's validation split, 25,250 photos);
- rvl_cdip: what type of document a scanned page is (16 classes; part of RVL-CDIP's test split);
- crack: does a concrete surface have a crack (yes/no; 24,000 photos, CC BY 4.0).

Files are downloaded from the Hugging Face Hub as parquet or zip, never with a dataset loading script, into
`$JEVSTILLER_IMAGE_DATA` (default ~/.cache/jevstiller/images/<dataset>/), outside the repository. Zip members are
read in memory, never extracted. A state is an image's id, `<dataset>/<n>`: embeddings are precomputed per id
(image_embed.py) and the replay (run.py) sees only ids.
"""
from __future__ import annotations

import io
import json
import os
import sys
import zipfile
from collections.abc import Iterator
from pathlib import Path

ROOT = Path(os.environ.get("JEVSTILLER_IMAGE_DATA", Path.home() / ".cache" / "jevstiller" / "images"))

# chainyo/rvl-cdip numbers the classes alphabetically (its README), not in RVL-CDIP's original order, and its
# parquet files carry no names
RVL_CLASSES = ["advertisement", "budget", "email", "file folder", "form", "handwritten", "invoice", "letter", "memo",
               "news article", "presentation", "questionnaire", "resume", "scientific publication",
               "scientific report", "specification"]

SPECS = {
    "food101": {"repo": "ethz/food101", "kind": "parquet",
                "files": [f"data/validation-0000{i}-of-00003.parquet" for i in range(3)],
                "instructions": "What dish is shown in this photo?"},
    "rvl_cdip": {"repo": "chainyo/rvl-cdip", "kind": "parquet",           # shards are sorted by label: all 15,
                 "files": [f"data/test-{i:05d}-of-00015.parquet" for i in range(15)],   # 1,000 of each class
                 "instructions": "What type of document is this page?", "names": RVL_CLASSES, "per_class": 1000},
    "crack": {"repo": "mohammadnajeeb/concrete_crack_images", "kind": "zip", "files": ["data/train.zip"],
              "instructions": "Does this concrete surface have a crack?",
              "folders": {"positive": "crack", "negative": "no crack"}},
}


def fetch(name: str) -> Path:
    from huggingface_hub import hf_hub_download
    spec, raw = SPECS[name], ROOT / name / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for f in spec["files"]:
        hf_hub_download(spec["repo"], f, repo_type="dataset", local_dir=raw)
    return raw


def _parquet_names(path: Path) -> list[str] | None:
    """Class names from the Hugging Face features stored in a parquet file's schema metadata."""
    import pyarrow.parquet as pq
    meta = pq.read_schema(path).metadata or {}
    hf = json.loads(meta.get(b"huggingface", b"{}"))
    label = (hf.get("info", {}).get("features") or hf.get("features") or {}).get("label", {})
    return label.get("names")


def _class_names(name: str) -> list[str]:
    spec, raw = SPECS[name], ROOT / name / "raw"
    if "names" in spec:
        return spec["names"]
    if spec["kind"] == "zip":
        return list(dict.fromkeys(spec["folders"].values()))
    names = _parquet_names(raw / spec["files"][0])
    if not names:
        raise SystemExit(f"{name}: no class names in the parquet metadata")
    return [n.replace("_", " ") for n in names]


def iter_images(name: str, decode: bool = True) -> Iterator[tuple[str, object, str]]:
    """(id, image, class name) for every image, in a fixed order. `image` is an RGB PIL image, or the raw bytes
    with decode=False."""
    from PIL import Image
    spec, raw = SPECS[name], ROOT / name / "raw"
    names = _class_names(name)
    n, taken, cap = 0, {}, spec.get("per_class")
    if spec["kind"] == "parquet":
        import pyarrow.parquet as pq
        for f in spec["files"]:
            pf = pq.ParquetFile(raw / f)
            for batch in pf.iter_batches(batch_size=256, columns=["image", "label"]):
                for img, label in zip(batch.column("image").to_pylist(), batch.column("label").to_pylist(), strict=True):
                    if cap is not None:
                        if taken.get(label, 0) >= cap:
                            continue
                        taken[label] = taken.get(label, 0) + 1
                    data = img["bytes"]
                    yield f"{name}/{n}", (Image.open(io.BytesIO(data)).convert("RGB") if decode else data), names[label]
                    n += 1
    else:
        folders = spec["folders"]
        for f in spec["files"]:
            with zipfile.ZipFile(raw / f) as z:
                for m in sorted(z.namelist()):
                    if not m.lower().endswith((".jpg", ".jpeg", ".png")):
                        continue
                    parts = [p.lower() for p in Path(m).parts[:-1]]
                    label = next((folders[p] for p in parts if p in folders), None)
                    if label is None:
                        continue
                    data = z.read(m)
                    yield f"{name}/{n}", (Image.open(io.BytesIO(data)).convert("RGB") if decode else data), label
                    n += 1


def load(name: str) -> dict:
    """The dataset as datasets.LOADERS returns one: rows of (state, class name), the classes, the question.
    The state is the image's id. Labels and class names come from a small index written next to the embeddings,
    so a replay needs neither the images nor pyarrow."""
    index = ROOT / name / "labels.json"
    d = json.loads(index.read_text()) if index.exists() else None
    if not isinstance(d, dict):
        d = {"classes": _class_names(name), "rows": [[i, c] for i, _, c in iter_images(name, decode=False)]}
        index.write_text(json.dumps(d))
    return {"rows": [tuple(r) for r in d["rows"]], "classes": {c: "" for c in d["classes"]},
            "instructions": SPECS[name]["instructions"]}


IMAGE_LOADERS = {f"img_{n}": (lambda n=n: load(n)) for n in SPECS}


class PrecomputedEncoder:
    """The Encoder interface over image_embed.py's embeddings: a state is an image id, its embedding is looked up."""

    def __init__(self, dataset: str, encoder: str):
        import hashlib

        import numpy as np
        z = np.load(ROOT / dataset / f"emb-{encoder}.npz")
        self.X = z["X"].astype(np.float32)
        self.index = {i: k for k, i in enumerate(z["ids"].tolist())}
        self.dim = self.X.shape[1]
        self.id = f"precomputed:{encoder}:{hashlib.sha256(self.X.tobytes()).hexdigest()[:12]}"
        self.device = "precomputed"

    def encode(self, texts):
        return self.X[[self.index[t] for t in texts]]


def sizes(name: str) -> dict:
    """Width/height stats, and the share Jev would refuse as they are (each side 280-1568 px)."""
    import numpy as np
    from PIL import Image
    w, h = [], []
    for _, data, _ in iter_images(name, decode=False):
        with Image.open(io.BytesIO(data)) as im:
            w.append(im.width)
            h.append(im.height)
    w, h = np.array(w), np.array(h)
    short = np.minimum(w, h)
    return {"n": len(w), "width_median": int(np.median(w)), "height_median": int(np.median(h)),
            "short_side_p5": int(np.percentile(short, 5)), "short_side_median": int(np.median(short)),
            "too_small_for_jev": float((short < 280).mean()), "too_large_for_jev": float((np.maximum(w, h) > 1568).mean())}


if __name__ == "__main__":
    cmd, name = sys.argv[1], sys.argv[2]
    if cmd == "fetch":
        print(fetch(name))
        d = load(name)
        counts: dict[str, int] = {}
        for _, c in d["rows"]:
            counts[c] = counts.get(c, 0) + 1
        print(f"{name}: {len(d['rows'])} images, {len(counts)} classes, per class {min(counts.values())}-{max(counts.values())}")
    elif cmd == "sizes":
        print(json.dumps(sizes(name), indent=1))
    else:
        raise SystemExit(__doc__)
