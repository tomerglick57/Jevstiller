"""What a caller's confidence check sees, with and without Task.confidence_floor.

    python3 experiments/confidence_floor.py --dataset banking77 [--seeds 5] [--floors 0.6,0.8] [--target 0.98]

Uses the recorded Jev answers (experiments/cache/<dataset>.jsonl; no API key), and the splits of baselines.py: per
seed a train / calibration / test split, one soft-label head, the OOD gate from the training data. The routing
policy is fitted twice on the same calibration rows:

- `label`: today's contract without a floor: a row disagrees when the label differs.
- `floor`: Task.confidence_floor: a row also disagrees when Jev's own confidence was below the floor.

A caller checks each answer's `confidence` against the floor ("below it: unsure, send to review"). On the test
split this reports:

- coverage: the share of requests answered locally.
- flags lost: of the requests Jev would have flagged (confidence below the floor), the share answered locally with
  a confidence at or above it, so the caller no longer flags them.
- flags added: requests Jev was sure about that got a local answer below the floor (share of all requests).
- outcome disagreement: the share of all requests where the caller's outcome changed (a different label, or the
  other side of the floor), and how many splits put it above the budget.
- local confidence: the lowest local confidence, and how often the student's (Jev's definition, before the floor)
  was below Jev's own on the same request.

Writes experiments/results/confidence-floor-<dataset>.json.
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
from datasets import LOADERS  # noqa: E402

from jevstiller import Config, Task, load_encoder  # noqa: E402
from jevstiller._calibrate import fit_policy, threshold_grid  # noqa: E402
from jevstiller._ood import KnnOOD  # noqa: E402
from jevstiller._student import LinearStudent  # noqa: E402
from jevstiller.teachers._replay import state_text  # noqa: E402


def peakedness(P: np.ndarray) -> np.ndarray:
    """Jev's confidence, row by row: (K * max - 1) / (K - 1)."""
    k = P.shape[1]
    return np.maximum(0.0, (k * P.max(axis=1) - 1) / (k - 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="banking77", choices=[d for d in LOADERS if d != "synthetic"])
    ap.add_argument("--encoder", default="small")
    ap.add_argument("--backend", default="onnx")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--floors", default="0.6,0.8")
    ap.add_argument("--target", type=float, default=0.98)
    ap.add_argument("--test-size", type=int, default=2000)
    ap.add_argument("--calib-fraction", type=float, default=0.2)
    a = ap.parse_args()
    floors = [float(f) for f in a.floors.split(",")]
    cfg = Config()
    ds = LOADERS[a.dataset]()
    task = Task(name=a.dataset, instructions=ds["instructions"], classes=ds["classes"], target_agreement=a.target)
    labels = list(task.labels)
    idx = {c: i for i, c in enumerate(labels)}
    cache = {}
    for line in open(ROOT / "experiments" / "cache" / f"{a.dataset}.jsonl"):
        d = json.loads(line)
        cache[d["key"]] = d
    rows = [t for t, _ in ds["rows"] if f"{task.version}\t{state_text(t)}" in cache]
    P = np.zeros((len(rows), len(labels)))
    J = np.zeros(len(rows))                              # Jev's own confidence, as it reported it
    for i, t in enumerate(rows):
        d = cache[f"{task.version}\t{state_text(t)}"]
        J[i] = d["confidence"]
        for c, p in d["probs"].items():
            if c in idx:
                P[i, idx[c]] = p
    P /= np.maximum(P.sum(axis=1, keepdims=True), 1e-9)
    jev = P.argmax(axis=1)
    enc = load_encoder(a.encoder, backend=a.backend, device=a.device)
    t0 = time.perf_counter()
    X = enc.encode(rows)
    print(f"{a.dataset}: {len(rows):,} rows, {len(labels)} classes, encoded in {time.perf_counter() - t0:.0f} s "
          f"({enc.id})", flush=True)

    budget, delta = task.budget, 1 - cfg.confidence
    out = {"dataset": a.dataset, "encoder": enc.id, "rows": len(rows), "classes": len(labels), "seeds": a.seeds,
           "target": a.target, "budget": budget, "date": time.strftime("%Y-%m-%d"),
           "runs": {str(f): {"label": [], "floor": []} for f in floors}}
    for seed in range(a.seeds):
        perm = np.random.default_rng(seed).permutation(len(rows))
        te, rest = perm[:a.test_size], perm[a.test_size:]
        n_cal = int(len(rest) * a.calib_fraction)
        cal, tr = rest[:n_cal], rest[n_cal:]
        student = LinearStudent(X.shape[1], len(labels))
        student.fit(X[tr], P[tr], epochs=cfg.student_epochs, l2=cfg.student_l2, seed=seed,
                    patience=cfg.student_patience)
        ood = KnnOOD(cfg.ood_k)
        ood.fit(X[tr], max_ref=cfg.ood_max_ref, seed=seed, y=P[tr].argmax(axis=1))
        ood_thr = ood.threshold(cfg.ood_quantile, seed=seed)
        Pc, Pt = student.predict_proba(X[cal]), student.predict_proba(X[te])
        oc, ot = ood.score(X[cal]), ood.score(X[te])
        same_label = Pc.argmax(axis=1) == jev[cal]
        pred, s = Pt.argmax(axis=1), peakedness(Pt)
        for floor in floors:
            for kind in ("label", "floor"):
                agree = same_label & (J[cal] >= floor) if kind == "floor" else same_label
                pol = fit_policy(Pc.max(axis=1), agree, oc, budget * (1 - cfg.fit_headroom), delta,
                                 ood_threshold=ood_thr, candidates=threshold_grid(len(labels)),
                                 confidence_floor=floor if kind == "floor" else None)
                loc = pol.accepts(Pt.max(axis=1), ot) if pol.usable else np.zeros(len(te), dtype=bool)
                reported = np.maximum(s, floor) if kind == "floor" else s
                jflag, ours = J[te] < floor, reported < floor
                changed = loc & ((pred != jev[te]) | (ours != jflag))
                out["runs"][str(floor)][kind].append({
                    "seed": seed, "coverage": float(loc.mean()),
                    "flags_lost": float((loc & jflag & ~ours).sum() / max(jflag.sum(), 1)),
                    "flags_added": float((loc & ~jflag & ours).mean()),
                    "outcome_disagreement": float(changed.mean()), "violated": bool(changed.mean() > budget),
                    "label_disagreement": float((loc & (pred != jev[te])).mean()),
                    "jev_flagged": float(jflag.mean()),
                    "local_confidence_min": float(reported[loc].min()) if loc.any() else None,
                    "student_below_jev": float((s[loc] < J[te][loc]).mean()) if loc.any() else None,
                })
        line = "  ".join(f"{f}: cov {out['runs'][str(f)]['label'][-1]['coverage']:.1%} -> "
                         f"{out['runs'][str(f)]['floor'][-1]['coverage']:.1%}" for f in floors)
        print(f"seed {seed:2d}  {line}", flush=True)

    print(f"\n{a.dataset}: {a.seeds} seeds, budget {budget:.1%} of requests, test {a.test_size:,} rows each")
    print(f"{'floor':>5} {'policy':<6} {'coverage':>9} {'flags lost':>11} {'flags added':>12} {'outcome dis':>12} "
          f"{'over budget':>12} {'Jev flagged':>12} {'lowest local conf':>18}")
    for f in floors:
        for kind in ("label", "floor"):
            runs = out["runs"][str(f)][kind]
            m = {k: np.mean([r[k] for r in runs if r[k] is not None]) for k in runs[0] if k not in ("seed", "violated")}
            low = min((r["local_confidence_min"] for r in runs if r["local_confidence_min"] is not None), default=None)
            print(f"{f:>5} {kind:<6} {m['coverage']:9.1%} {m['flags_lost']:11.1%} {m['flags_added']:12.1%} "
                  f"{m['outcome_disagreement']:12.2%} {sum(r['violated'] for r in runs):>7}/{len(runs):<4} "
                  f"{m['jev_flagged']:12.1%} {low if low is None else round(low, 3):>18}")
    dst = ROOT / "experiments" / "results" / f"confidence-floor-{a.dataset}.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1))
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
