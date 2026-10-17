"""Step 1 of the image benchmark: a table of the image replays.

    for d in crack food101 rvl_cdip; do for e in clip-b32 siglip-b16 dinov2-s; do
        python3 experiments/run.py --dataset img_$d --teacher oracle --encoder $e --tag img; done; done
    python3 experiments/image_bench.py

Each image dataset is replayed through the loop with its own labels as the teacher (`--teacher oracle`), no Jev
call. That measures whether a frozen image encoder separates the classes well enough for the bound to let a
student answer. It is not an upper bound on agreement with Jev: on Banking77 the same small text encoder covered
56% against the labels and 72% against Jev's own answers, which are easier to imitate. The other columns size
step 2 (recording Jev's answers): when the first student was promoted, and the encoder's cost per image on CPU.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from image_data import ROOT as DATA  # noqa: E402
from image_data import SPECS  # noqa: E402

ENCODERS = ["clip-b32", "siglip-b16", "dinov2-s"]


def pct(x: float | None) -> str:
    return f"{x:.1%}" if x is not None else "–"


def row(d: str, e: str) -> str | None:
    p = ROOT / "experiments" / "results" / f"img_{d}-oracle-{e}-probs-t98-img" / "results.json"
    if not p.exists():
        return None
    r = json.loads(p.read_text())
    evals = [c for c in r["checkpoints"] if c.get("eval")]
    first = next((c["teacher_calls"] for c in evals), None)
    f = evals[-1]["eval"] if evals else {}
    emb = json.loads((DATA / d / f"emb-{e}.json").read_text())
    cpu = emb["cpu"].get("ms_per_image_threads4_batch1")
    first_s = f"{first:,}" if first is not None else "none"
    return (f"| {d} | {e} | {r['n_classes']} | {r['n_stream']:,} | {first_s} | "
            f"{pct(f.get('coverage'))} | {pct(f.get('system_agreement'))} | {pct(f.get('student_agreement_all'))} | "
            f"{r['final_status']['teacher_calls']:,} | {cpu} |")


def main() -> None:
    print("| Dataset | Encoder | Classes | Stream | Labels before the first student | Coverage (held out) | "
          "Agreement (held out) | Student alone | Labels used in all | CPU ms/image |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for d in SPECS:
        for e in ENCODERS:
            line = row(d, e)
            if line:
                print(line)


if __name__ == "__main__":
    main()
