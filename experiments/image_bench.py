"""Tables of the image benchmark's replays (experiments/README.md, "Images").

    for d in crack food101 rvl_cdip; do for e in clip-b32 siglip-b16 dinov2-s; do
        python3 experiments/run.py --dataset img_$d --teacher oracle --encoder $e --tag img; done; done
    python3 experiments/image_bench.py

Step 2's rows come from the replays against Jev's answers (`--teacher jev`, or `--teacher cached` from the
recorded answers), with the settings in experiments/README.md.

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


def jev_row(d: str, e: str) -> str | None:
    """Step 2: the replay against Jev's answers, from the cache if it was replayed from it, else the live run."""
    for teacher, tag in (("cached", "img-cached"), ("cached", "img"), ("jev", "img")):
        p = ROOT / "experiments" / "results" / f"img_{d}-{teacher}-{e}-probs-t98-{tag}" / "results.json"
        if p.exists():
            break
    else:
        return None
    r = json.loads(p.read_text())
    evals = [c for c in r["checkpoints"] if c.get("eval")]
    f = evals[-1]["eval"] if evals else {}
    tp, fs = r["teacher_profile"], r["final_status"]
    calls = fs["teacher_calls"] + tp["n"]                  # the stream's teacher calls and the held-out rows
    cost = fs["teacher_cost_usd"] + tp["cost_usd"]
    return (f"| {d} | {e} | {pct(tp['teacher_accuracy_vs_truth'])} | {pct(tp['share_below_0.6'])} | "
            f"{r['n_eval']:,} | {r['n_stream']:,} | {pct(f.get('coverage'))} | {pct(f.get('system_agreement'))} | "
            f"{calls:,} | {cost / 0.042:.1f}M | ${cost:.2f} |")


def main() -> None:
    print("Step 1, the labels as the teacher:\n")
    print("| Dataset | Encoder | Classes | Stream | Labels before the first student | Coverage (held out) | "
          "Agreement (held out) | Student alone | Labels used in all | CPU ms/image |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for d in SPECS:
        for e in ENCODERS:
            line = row(d, e)
            if line:
                print(line)
    print("\nStep 2, Jev's answers (jev-1.14.0):\n")
    print("| Dataset | Encoder | Jev vs labels | Jev below 0.6 | Held out | Stream | Coverage (held out) | "
          "Agreement with Jev (held out) | Jev calls | Tokens | Cost |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for d in SPECS:
        for e in ENCODERS:
            line = jev_row(d, e)
            if line:
                print(line)


if __name__ == "__main__":
    main()
