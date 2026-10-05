"""What a local head can promise for a yes/no (`noul`) question, measured on Jev's recorded probabilities.

    python3 experiments/noul_curve.py --task offensive [--seeds 10] [--target 0.98]

Uses experiments/cache/noul_<task>.jsonl (record_noul.py; no API key). For each seed the rows are split into
train / calibration / test; a two-class head is trained on Jev's probability as a soft target [1-p, p], and its
own probability p_hat is compared with Jev's p on the test split under two kinds of contract:

- bucket: the caller declares cut-offs (one: yes/no; two: no / unsure / yes). "Agrees" means p_hat falls in the
  same bucket as p. The routing score is p_hat's distance from the nearest cut-off.
- tolerance: "agrees" means |p_hat - p| <= eps. The routing score is how extreme p_hat is.

The threshold on the routing score is chosen exactly as for `choice` (Clopper-Pearson bound with the loop's
headroom, fixed-sequence testing on the fixed grid, OOD cutoff from the training data). Reported per contract:
coverage, disagreement over all requests, and how many seeds exceeded the budget.
Writes experiments/results/noul-curve-<task>.json.
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
from noul_tasks import NOUL_TASKS, rows_for  # noqa: E402

from jevstiller import Config, load_encoder  # noqa: E402
from jevstiller._calibrate import fit_policy, threshold_grid  # noqa: E402
from jevstiller._ood import KnnOOD  # noqa: E402
from jevstiller._student import LinearStudent  # noqa: E402

BUCKETS = {"cut 0.5": [0.5], "cut 0.8": [0.8], "cut 0.2": [0.2], "band 0.4-0.6": [0.4, 0.6], "band 0.2-0.8": [0.2, 0.8]}
TOLERANCES = [0.05, 0.1, 0.2]


def bucket(p: np.ndarray, cuts: list[float]) -> np.ndarray:
    return np.searchsorted(np.array(cuts), p, side="right")


def dist(v: np.ndarray, cuts: list[float]) -> np.ndarray:
    """Distance of each probability from the nearest cut-off."""
    return np.min(np.abs(v[:, None] - np.array(cuts)[None, :]), axis=1)


def judge(score_c, score_t, agree_c, agree_t, ood_c, ood_t, ood_thr, fit_budget, delta, budget) -> dict:
    """Pick the routing threshold on the calibration rows as the loop does, then measure it on the test rows."""
    pol = fit_policy(score_c, agree_c, ood_c, fit_budget, delta, ood_threshold=ood_thr, candidates=threshold_grid(2))
    acc = pol.accepts(score_t, ood_t) if pol.usable else np.zeros(len(score_t), bool)
    dis = float((acc & ~agree_t).mean())
    return {"coverage": float(acc.mean()), "disagreement": dis, "violated": bool(dis > budget),
            "threshold": pol.conf_threshold}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="offensive", choices=list(NOUL_TASKS))
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--target", type=float, default=0.98)
    ap.add_argument("--test-size", type=int, default=2000)
    ap.add_argument("--encoder", default="small")
    a = ap.parse_args()
    cfg = Config()
    budget = 1 - a.target
    rec = {}
    for line in open(ROOT / "experiments" / "cache" / f"noul_{a.task}.jsonl"):
        d = json.loads(line)
        rec[d["text"]] = d["noul"]
    rows = [(t, y) for t, y in rows_for(a.task) if t in rec]
    texts = [t for t, _ in rows]
    truth = np.array([y for _, y in rows])
    p = np.array([rec[t] for t in texts])
    print(f"{a.task}: {len(rows):,} rows; dataset says yes {truth.mean():.1%}; Jev p>=0.5 on {(p >= 0.5).mean():.1%}; "
          f"Jev accuracy at 0.5 {((p >= 0.5) == truth).mean():.1%}")
    q = np.quantile(p, [0.05, 0.25, 0.5, 0.75, 0.95])
    print(f"  Jev's p: quantiles 5/25/50/75/95% = {np.round(q, 2).tolist()}; within 0.1 of 0 or 1: "
          f"{((p <= 0.1) | (p >= 0.9)).mean():.1%}; between 0.3 and 0.7: {((p > 0.3) & (p < 0.7)).mean():.1%}")
    enc = load_encoder(a.encoder, backend="onnx", device="cpu")
    t0 = time.perf_counter()
    X = enc.encode(texts)
    print(f"  encoded in {time.perf_counter() - t0:.0f} s", flush=True)

    names = [f"bucket {k}" for k in BUCKETS] + [f"within {e}" for e in TOLERANCES]
    runs = {n: [] for n in names}
    fit = []
    for seed in range(a.seeds):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(rows))
        te, rest = perm[:a.test_size], perm[a.test_size:]
        n_cal = int(len(rest) * cfg.calib_fraction)
        cal, tr = rest[:n_cal], rest[n_cal:]
        Y = np.stack([1 - p[tr], p[tr]], axis=1)
        head = LinearStudent(X.shape[1], 2)
        head.fit(X[tr], Y, epochs=cfg.student_epochs, l2=cfg.student_l2, seed=seed, patience=cfg.student_patience)
        ph_c, ph_t = head.predict_proba(X[cal])[:, 1], head.predict_proba(X[te])[:, 1]
        ood = KnnOOD(cfg.ood_k)
        ood.fit(X[tr], max_ref=cfg.ood_max_ref, seed=seed, y=(p[tr] >= 0.5).astype(int))
        ood_c, ood_t, ood_thr = ood.score(X[cal]), ood.score(X[te]), ood.threshold(cfg.ood_quantile, seed=seed)
        fit.append({"mae": float(np.abs(ph_t - p[te]).mean()), "corr": float(np.corrcoef(ph_t, p[te])[0, 1]),
                    "side_agree": float(((ph_t >= 0.5) == (p[te] >= 0.5)).mean()),
                    "acc_jev": float(((p[te] >= 0.5) == truth[te]).mean()),
                    "acc_head": float(((ph_t >= 0.5) == truth[te]).mean())})

        gate = (ood_c, ood_t, ood_thr, budget * (1 - cfg.fit_headroom), 1 - cfg.confidence, budget)
        for k, cuts in BUCKETS.items():
            runs[f"bucket {k}"].append(judge(0.5 + np.minimum(dist(ph_c, cuts), 0.5), 0.5 + np.minimum(dist(ph_t, cuts), 0.5),
                                             bucket(ph_c, cuts) == bucket(p[cal], cuts),
                                             bucket(ph_t, cuts) == bucket(p[te], cuts), *gate))
        for eps in TOLERANCES:
            runs[f"within {eps}"].append(judge(np.maximum(ph_c, 1 - ph_c), np.maximum(ph_t, 1 - ph_t),
                                               np.abs(ph_c - p[cal]) <= eps, np.abs(ph_t - p[te]) <= eps, *gate))
        print(f"  seed {seed}: MAE {fit[-1]['mae']:.3f}  same side of 0.5 {fit[-1]['side_agree']:.1%}  "
              + "  ".join(f"{n.split()[-1]}:{runs[n][-1]['coverage']:.0%}" for n in names), flush=True)

    m = lambda k: float(np.mean([f[k] for f in fit]))  # noqa: E731
    print(f"\n{a.task}: head vs Jev over {a.seeds} seeds: MAE {m('mae'):.3f}, correlation {m('corr'):.3f}, "
          f"same side of 0.5 without any routing {m('side_agree'):.1%}; accuracy vs dataset: Jev {m('acc_jev'):.1%}, "
          f"head {m('acc_head'):.1%}")
    print(f"target {a.target:.0%} (budget {budget:.0%} of requests)")
    print(f"{'contract':<22} {'coverage':>9} {'disagreement':>13} {'worst':>7} {'violations':>11}")
    summary = {}
    for n in names:
        r = runs[n]
        cov, dis = np.mean([x["coverage"] for x in r]), np.mean([x["disagreement"] for x in r])
        worst, viol = max(x["disagreement"] for x in r), sum(x["violated"] for x in r)
        summary[n] = {"coverage": float(cov), "disagreement": float(dis), "worst": float(worst), "violations": viol}
        print(f"{n:<22} {cov:9.1%} {dis:13.2%} {worst:7.2%} {viol:>6}/{len(r)}")
    out = {"task": a.task, "rows": len(rows), "seeds": a.seeds, "target": a.target, "fit": fit, "runs": runs,
           "summary": summary, "date": time.strftime("%Y-%m-%d")}
    dst = ROOT / "experiments" / "results" / f"noul-curve-{a.task}.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1))
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
