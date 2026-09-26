"""Threshold rules and label targets, compared on identical data: what the bound and the soft labels buy.

    python3 experiments/baselines.py --dataset banking77 [--seeds 20] [--encoder small]

Uses the recorded Jev answers (experiments/cache/<dataset>.jsonl, see record_answers.py; no API key). For each
seed the rows are split into train / calibration / test; a head is trained on the train split and a confidence
threshold is picked on the calibration split, then the test split measures what actually happened:

- coverage: share of test requests the student answered
- disagreement: share of *all* test requests answered by the student with a label different from Jev's
  (the quantity the contract bounds), and whether it exceeded the budget
- accuracy of the system (student where it answers, Jev elsewhere) against the dataset's own labels

Four rules are compared. `bound` is Jevstiller's (Clopper-Pearson, fixed-sequence testing, OOD gate from the
training data); `point` is the usual recipe (loosest threshold whose empirical disagreement on the calibration
split is within the budget, no OOD gate). Each is trained on Jev's probability distributions (`soft`) or on its
argmax only (`hard`). Writes experiments/results/baselines-<dataset>.json.
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
from jevstiller._calibrate import clopper_pearson_upper, fit_policy, threshold_grid  # noqa: E402
from jevstiller._ood import KnnOOD  # noqa: E402
from jevstiller._student import LinearStudent  # noqa: E402
from jevstiller.teachers._replay import state_text  # noqa: E402

RULES = ("soft+bound", "hard+bound", "soft+point", "hard+point")


def point_threshold(conf: np.ndarray, agree: np.ndarray, budget: float) -> float | None:
    """The loosest threshold whose empirical disagreement rate over all calibration rows is within budget."""
    best = None
    for t in sorted(threshold_grid(2), reverse=True):
        sel = conf >= t
        if not sel.any():
            continue
        if (sel & ~agree).mean() <= budget:
            best = float(t)
        else:
            break
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="banking77", choices=[d for d in LOADERS if d != "synthetic"])
    ap.add_argument("--encoder", default="small")
    ap.add_argument("--backend", default="onnx")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--target", type=float, default=0.98)
    ap.add_argument("--limit", type=int, default=0, help="use only this many rows (0 = all)")
    ap.add_argument("--test-size", type=int, default=2000)
    ap.add_argument("--calib-fraction", type=float, default=0.2)
    a = ap.parse_args()
    cfg = Config()
    ds = LOADERS[a.dataset]()
    task = Task(name=a.dataset, instructions=ds["instructions"], classes=ds["classes"], target_agreement=a.target)
    labels = list(task.labels)
    idx = {c: i for i, c in enumerate(labels)}

    cache = {}
    for line in open(ROOT / "experiments" / "cache" / f"{a.dataset}.jsonl"):
        d = json.loads(line)
        cache[d["key"]] = d
    rows = [(t, c) for t, c in ds["rows"] if f"{task.version}\t{state_text(t)}" in cache]
    missing = len(ds["rows"]) - len(rows)
    if missing:
        print(f"{missing:,} rows have no recorded Jev answer and are skipped (record_answers.py fills them)")
    if a.limit:
        rows = rows[:a.limit]
    texts = [t for t, _ in rows]
    truth = np.array([idx[c] for _, c in rows])
    P = np.zeros((len(rows), len(labels)))
    for i, (t, _) in enumerate(rows):
        d = cache[f"{task.version}\t{state_text(t)}"]
        for c, p in d["probs"].items():
            if c in idx:
                P[i, idx[c]] = p
    P /= np.maximum(P.sum(axis=1, keepdims=True), 1e-9)
    jev = P.argmax(axis=1)
    print(f"{len(rows):,} rows, {len(labels)} classes; Jev accuracy vs labels {(jev == truth).mean():.1%}")

    enc = load_encoder(a.encoder, backend=a.backend, device=a.device)
    t0 = time.perf_counter()
    X = enc.encode(texts)
    print(f"encoded in {time.perf_counter() - t0:.0f} s ({enc.id})", flush=True)

    budget = task.budget
    out = {"dataset": a.dataset, "encoder": enc.id, "rows": len(rows), "classes": len(labels), "seeds": a.seeds,
           "target": a.target, "budget": budget, "jev_accuracy": float((jev == truth).mean()),
           "date": time.strftime("%Y-%m-%d"), "runs": {r: [] for r in RULES}}
    for seed in range(a.seeds):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(rows))
        te, rest = perm[:a.test_size], perm[a.test_size:]
        n_cal = int(len(rest) * a.calib_fraction)
        cal, tr = rest[:n_cal], rest[n_cal:]
        for target_kind in ("soft", "hard"):
            Y = P[tr] if target_kind == "soft" else np.eye(len(labels))[jev[tr]]
            student = LinearStudent(X.shape[1], len(labels))
            student.fit(X[tr], Y, epochs=cfg.student_epochs, l2=cfg.student_l2, seed=seed,
                        patience=cfg.student_patience)
            Pc, Pt = student.predict_proba(X[cal]), student.predict_proba(X[te])
            conf_c, conf_t = Pc.max(axis=1), Pt.max(axis=1)
            agree_c = Pc.argmax(axis=1) == jev[cal]
            pred_t = Pt.argmax(axis=1)
            for rule in ("bound", "point"):
                if rule == "bound":
                    ood = KnnOOD(cfg.ood_k)
                    ood.fit(X[tr], max_ref=cfg.ood_max_ref, seed=seed, y=Y.argmax(axis=1))
                    pol = fit_policy(conf_c, agree_c, ood.score(X[cal]), budget * (1 - cfg.fit_headroom),
                                     1 - cfg.confidence, ood_threshold=ood.threshold(cfg.ood_quantile, seed=seed),
                                     candidates=threshold_grid(len(labels)))
                    acc = pol.accepts(conf_t, ood.score(X[te])) if pol.usable else np.zeros(len(te), bool)
                    thr = pol.conf_threshold
                else:
                    thr = point_threshold(conf_c, agree_c, budget)
                    acc = (conf_t >= thr) if thr is not None else np.zeros(len(te), bool)
                dis = float((acc & (pred_t != jev[te])).mean())
                system = np.where(acc, pred_t, jev[te])
                run = {"seed": seed, "threshold": thr, "coverage": float(acc.mean()), "disagreement": dis,
                       "violated": bool(dis > budget), "system_accuracy": float((system == truth[te]).mean()),
                       "calib_upper_bound": float(clopper_pearson_upper(int((acc & (pred_t != jev[te])).sum()),
                                                                        len(te), 1 - cfg.confidence))}
                out["runs"][f"{target_kind}+{rule}"].append(run)
        line = "  ".join(f"{r}: cov {out['runs'][r][-1]['coverage']:.1%} dis {out['runs'][r][-1]['disagreement']:.2%}"
                         for r in RULES)
        print(f"seed {seed:2d}  {line}", flush=True)

    print(f"\n{a.dataset}: {a.seeds} seeds, budget {budget:.1%} of requests, test {a.test_size:,} rows each")
    print(f"{'rule':<11} {'coverage':>9} {'disagreement':>13} {'violations':>11} {'system acc':>11}")
    for r in RULES:
        runs = out["runs"][r]
        cov = np.mean([x["coverage"] for x in runs])
        dis = np.mean([x["disagreement"] for x in runs])
        viol = sum(x["violated"] for x in runs)
        sacc = np.mean([x["system_accuracy"] for x in runs])
        print(f"{r:<11} {cov:9.1%} {dis:13.2%} {viol:>6}/{len(runs):<4} {sacc:11.1%}")
    dst = ROOT / "experiments" / "results" / f"baselines-{a.dataset}.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1))
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
