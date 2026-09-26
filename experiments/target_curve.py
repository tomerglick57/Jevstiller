"""What the agreement target buys: coverage and agreement at several targets, for each benchmarked task.

    python3 experiments/target_curve.py [--tag bench] [--targets 0.99,0.98,0.97,0.95,0.93,0.90]

For each task with a replay under experiments/results/<task>-cached-small-probs-t98-<tag>/, takes the
production student of that replay and its calibration rows from the replay's sample store, refits the routing
policy at each target exactly as the loop does (Clopper-Pearson bound with the loop's headroom, fixed-sequence
testing on the fixed grid, the version's own OOD cutoff), and measures the result on the 2,000 held-out rows:
coverage, agreement with Jev, and accuracy against the dataset's labels. Writes experiments/results/target-curve.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from datasets import LOADERS  # noqa: E402
from run import CacheOnlyTeacher  # noqa: E402

from jevstiller import Config, Task, load_encoder  # noqa: E402
from jevstiller._calibrate import RoutingPolicy, fit_policy, threshold_grid  # noqa: E402
from jevstiller._ood import KnnOOD  # noqa: E402
from jevstiller._store import SampleStore  # noqa: E402
from jevstiller._student import LinearStudent  # noqa: E402
from jevstiller.teachers import CachedTeacher  # noqa: E402


def one(dataset: str, tag: str, targets: list[float], enc) -> dict | None:
    res = ROOT / "experiments" / "results" / f"{dataset}-cached-small-probs-t98-{tag}"
    if not (res / "results.json").exists():
        return None
    r = json.loads((res / "results.json").read_text())
    a = r["args"]
    state = res / "state" / dataset
    reg = json.loads((state / "versions" / "registry.json").read_text())
    if not reg.get("production"):
        return None
    vdir = Path(next(v["dir"] for v in reg["versions"] if v["name"] == reg["production"]))
    meta = json.loads((vdir / "meta.json").read_text())
    student, ood = LinearStudent.load(vdir / "head.npz"), KnnOOD.load(vdir / "ood.npz")
    pol = RoutingPolicy.load(vdir / "policy.json")
    cfg = Config()

    ds = LOADERS[dataset]()
    rows = list(ds["rows"])
    rng = np.random.default_rng(a["seed"])
    rng.shuffle(rows)
    ev = rows[:a["eval_size"]]
    task = Task(name=dataset, instructions=ds["instructions"], classes=ds["classes"], target_agreement=a["target"])
    labels = list(task.labels)
    cache = ROOT / "experiments" / "cache" / f"{dataset}.jsonl"
    teacher = CachedTeacher(CacheOnlyTeacher(cache), cache)
    texts = [t for t, _ in ev]
    truth = np.array([c for _, c in ev])
    jev = np.array([o.label for o in teacher.classify(texts, task)])

    store = SampleStore(state / "samples.sqlite", read_only=True)
    try:
        Xc, _, yc, _ = store.calib_set(meta["task_version"], meta["encoder"], labels, student.W.shape[0],
                                       meta.get("teacher_model"), meta.get("since_id", 0))
    finally:
        store.close()
    Pc = student.predict_proba(Xc)
    conf_c, agree_c, ood_c = Pc.max(axis=1), Pc.argmax(axis=1) == yc, ood.score(Xc)

    X = enc.encode(texts)
    P = student.predict_proba(X)
    conf, pred, oods = P.max(axis=1), np.array(labels, dtype=object)[P.argmax(axis=1)], ood.score(X)
    out = {"version": reg["production"], "n_calib": int(len(Xc)), "n_eval": len(ev),
           "jev_accuracy": float((jev == truth).mean()), "points": []}
    for target in targets:
        budget = 1 - target
        p = fit_policy(conf_c, agree_c, ood_c, budget * (1 - cfg.fit_headroom), 1 - cfg.confidence,
                       ood_threshold=pol.ood_threshold, candidates=threshold_grid(len(labels)))
        acc = p.accepts(conf, oods) if p.usable else np.zeros(len(ev), bool)
        system = np.where(acc, pred, jev)
        out["points"].append({"target": target, "threshold": p.conf_threshold, "coverage": float(acc.mean()),
                              "agreement": float(1 - (acc & (pred != jev)).mean()),
                              "accuracy": float((system == truth).mean()),
                              "bound": p.disagreement_ub})
        pt = out["points"][-1]
        print(f"  target {target:.0%}: threshold {pt['threshold'] if pt['threshold'] is not None else float('nan'):.3f}"
              f"  coverage {pt['coverage']:.1%}  agreement {pt['agreement']:.2%}  accuracy {pt['accuracy']:.1%}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="bench")
    ap.add_argument("--targets", default="0.99,0.98,0.97,0.95,0.93,0.90")
    a = ap.parse_args()
    targets = [float(t) for t in a.targets.split(",")]
    enc = load_encoder("small", backend="onnx", device="cpu")
    out = {}
    for d in [d for d in LOADERS if d != "synthetic"]:
        print(d, flush=True)
        r = one(d, a.tag, targets, enc)
        if r:
            out[d] = r
    dst = ROOT / "experiments" / "results" / "target-curve.json"
    dst.write_text(json.dumps(out, indent=1))
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
