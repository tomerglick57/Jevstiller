"""Image encoders compared on Jev's own answers, with no Jev call.

    python3 experiments/image_compare.py [--run]

Each dataset is cut down to the images that have Jev 1.14's recorded answer (step 2's cache, `run.py --only-cached`),
so whatever any encoder sends to the teacher is in the cache: every encoder sees the same images and the same answers.
The stream leans towards the harder images (step 2's loop answered the easy ones itself and never asked Jev about
them), so the numbers are lower than step 2's; the comparison between encoders is what counts. Embeddings come from
image_embed.py (`--only-cached jev-1.14.0` is enough); CPU costs from image_onnx.py.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from image_data import ROOT as DATA  # noqa: E402

DATASETS = ["crack", "food101", "rvl_cdip"]
ENCODERS = ["clip-b32", "clip-b32-openai", "siglip-b16", "siglip2-b16", "siglip-b16-384", "siglip-l16-256",
            "siglip-so400m", "dinov2-s", "dinov2-b", "dinov3-s", "dinov3-b"]


def result(d: str, e: str) -> dict | None:
    p = ROOT / "experiments" / "results" / f"img_{d}-cached-{e}-probs-t98-cmp" / "results.json"
    if not p.exists():
        return None
    r = json.loads(p.read_text())
    evals = [c for c in r["checkpoints"] if c.get("eval")]
    f = evals[-1]["eval"] if evals else {}
    first = next((c["examples"] for c in r["checkpoints"] if c.get("production")), None)
    return {"coverage": f.get("coverage"), "agreement": f.get("system_agreement"), "first": first,
            "stream_local": r["final_status"]["student_share"], "n_stream": r["n_stream"]}


def run(d: str, e: str) -> str:
    if result(d, e) is not None or not (DATA / d / f"emb-{e}.npz").exists():
        return f"skip {d} {e}"
    cmd = [sys.executable, str(ROOT / "experiments" / "run.py"), "--dataset", f"img_{d}", "--teacher", "cached",
           "--only-cached", "--encoder", e, "--eval-size", "1000", "--tag", "cmp"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    return f"{'done' if p.returncode == 0 else 'FAILED'} {d} {e}" + ("" if p.returncode == 0 else p.stderr[-300:])


def baseline(d: str, e: str) -> dict | None:
    """baselines.py on the same images: Jevstiller's rule (soft labels, the bound, the OOD gate) over fixed splits."""
    p = ROOT / "experiments" / "results" / f"baselines-img_{d}-{e}.json"
    if not p.exists():
        return None
    runs = json.loads(p.read_text())["runs"]["soft+bound"]
    import numpy as np
    cov = np.array([r["coverage"] for r in runs])
    dis = np.array([r["disagreement"] for r in runs])
    return {"coverage": cov.mean(), "sd": cov.std(), "worst": dis.max(), "violated": sum(r["violated"] for r in runs),
            "n": len(runs)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="run the replays that are missing (4 at a time)")
    a = ap.parse_args()
    if a.run:
        with ThreadPoolExecutor(4) as ex:
            for line in ex.map(lambda de: run(*de), [(d, e) for d in DATASETS for e in ENCODERS]):
                print(line, flush=True)
    print("| Encoder | " + " | ".join(DATASETS) + " |")
    print("|---|" + "---|" * len(DATASETS))
    for e in ENCODERS:
        cells = []
        for d in DATASETS:
            r = result(d, e)
            cells.append("–" if r is None or r["coverage"] is None else
                         f"{r['coverage']:.1%} @ {r['agreement']:.1%}")
        print(f"| {e} | " + " | ".join(cells) + " |")
    print("\nThreshold rule on fixed splits (baselines.py, soft labels + bound + OOD gate): mean coverage ± sd, "
          "worst disagreement, violations of the 2% budget")
    print("| Encoder | " + " | ".join(DATASETS) + " |")
    print("|---|" + "---|" * len(DATASETS))
    for e in ENCODERS:
        cells = []
        for d in DATASETS:
            b = baseline(d, e)
            cells.append("–" if b is None else
                         f"{b['coverage']:.1%} ± {b['sd']:.1%}, worst {b['worst']:.2%}, {b['violated']}/{b['n']}")
        print(f"| {e} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
