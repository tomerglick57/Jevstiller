"""Replay a labelled dataset through Jevstiller with the labels hidden; report the curves from DESIGN.md §15.

    python3 experiments/run.py --dataset banking77 --teacher jev --encoder small
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets import LOADERS  # noqa: E402

from jevstiller import Config, Jevstiller, Task, load_encoder  # noqa: E402
from jevstiller.teachers import CachedTeacher, SyntheticTeacher  # noqa: E402


class CacheOnlyTeacher:
    """Stands in for Jev when every answer is already in the cache (`--teacher cached`, no API key needed)."""
    name = "jev:jev-1.13.0"

    def __init__(self, cache: Path):
        self.cache = cache

    def classify(self, texts, task):
        raise SystemExit(f"{len(texts)} message(s) are not in {self.cache}. Unpack experiments/cache/banking77.jsonl.gz "
                         f"(see experiments/reproduce.sh), or run with --teacher jev and TYPESAFE_API_KEY to record them.")


class OracleTeacher:
    """Returns the dataset's own label with probability `p_true`, the rest spread uniformly. Upper bound, not a product path."""
    name = "oracle"

    def __init__(self, truth: dict, p_true: float = 0.9):
        self.truth, self.p_true = truth, p_true

    def classify(self, texts, task):
        from jevstiller.teachers import TeacherOutput, peakedness
        out = []
        K = len(task.labels)
        for t in texts:
            y = self.truth[t]
            probs = {c: (self.p_true if c == y else (1 - self.p_true) / (K - 1)) for c in task.labels}
            out.append(TeacherOutput(y, probs, peakedness(probs), input_tokens=len(t.split()) + 40, cost_usd=0.0))
        return out


CHECKPOINTS = [250, 500, 1000, 2000, 3000, 5000, 7500, 10000, 15000, 20000, 30000, 50000, 75000, 100000]
JEV_RPM = 1200


def load_env(path: Path) -> None:
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="banking77", choices=list(LOADERS))
    ap.add_argument("--teacher", default="jev", choices=["jev", "cached", "synthetic", "oracle"],
                    help="cached: Jev's recorded answers only (experiments/cache/), no API key")
    ap.add_argument("--encoder", default="base")
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--label-target", default="probs", choices=["probs", "hard"])
    ap.add_argument("--target", type=float, default=0.98)
    ap.add_argument("--limit", type=int, default=0, help="max streamed rows (0 = all)")
    ap.add_argument("--eval-size", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--audit", type=float, default=0.02)
    ap.add_argument("--min-train", type=int, default=500)
    ap.add_argument("--min-per-class", type=int, default=5)
    ap.add_argument("--min-calib", type=int, default=200)
    ap.add_argument("--min-new", type=int, default=1000)
    ap.add_argument("--shadow-min", type=int, default=300)
    ap.add_argument("--ood-max-ref", type=int, default=5000)
    ap.add_argument("--rare-classes", default=None, choices=["wait", "defer"],
                    help="classes below --min-per-class: wait for them (default) or train and defer them to the teacher")
    ap.add_argument("--out", default="experiments/results")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    load_env(root / ".env")
    ds = LOADERS[a.dataset]()
    rng = np.random.default_rng(a.seed)
    rows = list(ds["rows"])
    rng.shuffle(rows)
    ev, stream = rows[:a.eval_size], rows[a.eval_size:]
    if a.limit:
        stream = stream[:a.limit]
    task = Task(name=a.dataset, instructions=ds["instructions"], classes=ds["classes"], target_agreement=a.target)

    if a.teacher in ("jev", "cached"):
        cache = root / "experiments" / "cache" / f"{a.dataset}.jsonl"
        if a.teacher == "cached":
            inner = CacheOnlyTeacher(cache)
        else:
            from jevstiller.teachers.jev import JevTeacher
            inner = JevTeacher()
        teacher = CachedTeacher(inner, cache)
    elif a.teacher == "oracle":
        teacher = OracleTeacher(dict(ds["rows"]))          # dry run: the hidden labels as a perfect teacher
    else:
        assert a.dataset == "synthetic", "the synthetic teacher only makes sense with the synthetic dataset"
        teacher = SyntheticTeacher(ds["world"])

    name = f"{a.dataset}-{a.teacher}-{a.encoder}-{a.label_target}-t{int(a.target * 100)}" + (f"-{a.tag}" if a.tag else "")
    out = root / a.out / name
    out.mkdir(parents=True, exist_ok=True)
    data_dir = out / "state"
    if data_dir.exists():
        import shutil
        shutil.rmtree(data_dir)

    t0 = time.perf_counter()
    enc = load_encoder(a.encoder, backend=a.backend, device=a.device)
    ev_texts = [t for t, _ in ev]
    ev_truth = np.array([c for _, c in ev])
    enc.encode(ev_texts[:64])                                   # warm-up
    t1 = time.perf_counter()
    X_ev = enc.encode(ev_texts)
    enc_ms = (time.perf_counter() - t1) * 1000 / len(ev_texts)
    print(f"encoder {enc.id} on {getattr(enc, 'device', 'cpu')}: {enc_ms:.2f} ms/text (batched)")

    # --- teacher profile on the held-out slice (also gives eval labels) --------------------------------
    print(f"teacher-labelling {len(ev_texts)} held-out rows ...")
    outs = []
    for s in range(0, len(ev_texts), a.batch):
        outs += teacher.classify(ev_texts[s:s + a.batch], task)
    ev_teacher = np.array([o.label for o in outs])
    conf = np.array([o.confidence for o in outs])
    profile = {"n": len(outs), "confidence_quantiles": {q: float(np.quantile(conf, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
               "share_below_0.6": float((conf < 0.6).mean()), "share_below_0.9": float((conf < 0.9).mean()),
               "teacher_accuracy_vs_truth": float((ev_teacher == ev_truth).mean()),
               "mean_latency_ms": float(np.mean([o.latency_ms for o in outs if o.latency_ms])) if any(o.latency_ms for o in outs) else None,
               "mean_input_tokens": float(np.mean([o.input_tokens for o in outs])),
               "cost_usd": float(sum(o.cost_usd for o in outs))}
    print(f"teacher profile: acc vs truth {profile['teacher_accuracy_vs_truth']:.3f}, "
          f"conf median {profile['confidence_quantiles'][0.5]:.3f}, below 0.6: {profile['share_below_0.6']:.1%}, "
          f"latency {profile['mean_latency_ms']} ms")

    cfg = Config(audit_rate=a.audit, min_train_samples=a.min_train, min_samples_per_class=a.min_per_class,
                 min_calib_samples=a.min_calib, min_new_samples=a.min_new, shadow_min_samples=a.shadow_min,
                 seed=a.seed, label_target=a.label_target, ood_max_ref=a.ood_max_ref,
                 training="inline",                     # replays must not depend on thread timing
                 **({"rare_classes": a.rare_classes} if a.rare_classes else {}))
    js = Jevstiller(task, teacher, data_dir, encoder=enc, config=cfg)

    # --- replay ----------------------------------------------------------------------------------------
    checkpoints = []
    window = []          # (served_by_student, ) for the last 1000 requests
    seen = 0
    next_cp = [c for c in CHECKPOINTS if c <= len(stream)] + [len(stream)]
    texts = [t for t, _ in stream]
    for s in range(0, len(texts), a.batch):
        res = js.classify_batch(texts[s:s + a.batch])
        window += [r.source != "teacher" for r in res]
        window = window[-1000:]
        seen += len(res)
        if next_cp and seen >= next_cp[0]:
            while next_cp and seen >= next_cp[0]:
                next_cp.pop(0)
            st = js.status()
            cp = {"examples": seen, "teacher_calls": st.teacher_calls, "production": st.production,
                  "student_share_cumulative": st.student_share, "student_share_window": float(np.mean(window)),
                  "audit_agreement": st.audit_agreement, "audit_agreement_lb": st.audit_agreement_lb,
                  "audit_n": st.audit_n, "policy": st.policy, "elapsed_s": time.perf_counter() - t0}
            if st.production:
                e = js.evaluate(ev_texts, ev_teacher, X=X_ev)
                acc = e["accepted"]
                sys_label = np.where(acc, e["student_label"], ev_teacher)
                cp["eval"] = {"coverage": e["coverage"], "selective_disagreement": e["selective_disagreement"],
                              "system_agreement": e["system_agreement"],
                              "student_agreement_all": e["student_agreement_all"],
                              "student_accuracy_on_accepted": float((e["student_label"][acc] == ev_truth[acc]).mean()) if acc.any() else None,
                              "system_accuracy_vs_truth": float((sys_label == ev_truth).mean()),
                              "teacher_accuracy_vs_truth": profile["teacher_accuracy_vs_truth"]}
            checkpoints.append(cp)
            ev_s = cp.get("eval", {})
            print(f"[{seen:>7,}] prod={st.production or '-':>11} window student share {cp['student_share_window']:.1%}  "
                  f"eval coverage {ev_s.get('coverage', float('nan')):.1%}  system agreement {ev_s.get('system_agreement', float('nan')):.2%}  "
                  f"teacher calls {st.teacher_calls:,}")

    # --- throughput -------------------------------------------------------------------------------------
    st = js.status()
    burst = {}
    if st.production:
        t1 = time.perf_counter()
        js.evaluate(ev_texts, ev_teacher)          # encode + student + ood + policy, no store writes
        dt = time.perf_counter() - t1
        burst = {"rows": len(ev_texts), "student_seconds": dt, "student_rows_per_s": len(ev_texts) / dt,
                 "jev_seconds_at_rate_limit": len(ev_texts) / JEV_RPM * 60,
                 "jev_seconds_sequential": len(ev_texts) * (profile["mean_latency_ms"] or 350) / 1000}

    final = js.status()
    results = {"args": vars(a), "task_version": task.version, "encoder": enc.id, "encoder_device": getattr(enc, "device", "cpu"),
               "encoder_ms_per_text": enc_ms, "n_stream": len(stream), "n_eval": len(ev), "n_classes": len(task.labels),
               "teacher_profile": profile, "checkpoints": checkpoints, "burst": burst,
               "final_status": {k: v for k, v in final.__dict__.items() if k != "events"}, "events": final.events,
               "teacher_cache": {"hits": getattr(teacher, "hits", None), "misses": getattr(teacher, "misses", None)}}
    (out / "results.json").write_text(json.dumps(results, indent=2, default=float))
    write_report(out, results)
    try:
        plot(out, results)
    except Exception as e:  # matplotlib optional
        print("plot skipped:", e)
    print("\n" + final.report())
    print(f"\nwrote {out}")


def write_report(out: Path, r: dict) -> None:
    p = r["teacher_profile"]
    L = [f"# {out.name}", "",
         f"stream {r['n_stream']:,} rows · eval {r['n_eval']:,} rows · {r['n_classes']} classes · target agreement {r['args']['target']:.0%}",
         f"encoder `{r['encoder']}` on {r['encoder_device']}: {r['encoder_ms_per_text']:.2f} ms/text", "",
         "## Teacher profile (held-out slice)", "",
         f"- accuracy vs hidden truth: **{p['teacher_accuracy_vs_truth']:.1%}**",
         f"- confidence median {p['confidence_quantiles'][0.5]:.3f}; share below 0.6: {p['share_below_0.6']:.1%}; below 0.9: {p['share_below_0.9']:.1%}",
         f"- mean latency {p['mean_latency_ms']} ms · mean input tokens {p['mean_input_tokens']:.0f} · cost ${p['cost_usd']:.4f}", "",
         "## Curve", "",
         "| examples | production | Jev share (window) | eval coverage | selective disagreement | system agreement | student acc (accepted) | system acc | teacher acc |",
         "|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for c in r["checkpoints"]:
        e = c.get("eval")
        if e:
            L.append(f"| {c['examples']:,} | {c['production']} | {1 - c['student_share_window']:.1%} | {e['coverage']:.1%} | "
                     f"{e['selective_disagreement']:.2%} | {e['system_agreement']:.2%} | "
                     f"{(e['student_accuracy_on_accepted'] or 0):.1%} | {e['system_accuracy_vs_truth']:.1%} | {e['teacher_accuracy_vs_truth']:.1%} |")
        else:
            L.append(f"| {c['examples']:,} | - | 100% | - | - | - | - | - | - |")
    b = r.get("burst")
    if b:
        L += ["", "## Throughput (held-out slice, student path)", "",
              f"- student: **{b['student_rows_per_s']:,.0f} rows/s** ({b['rows']:,} rows in {b['student_seconds']:.2f} s)",
              f"- Jev at the 1,200/min rate limit: {b['jev_seconds_at_rate_limit']:.0f} s",
              f"- Jev sequential at measured latency: {b['jev_seconds_sequential']:.0f} s"]
    L += ["", "## Events", ""] + [f"- {e['kind']}: " + ", ".join(f"{k}={v}" for k, v in e.items() if k not in ("kind", "ts")) for e in r["events"]]
    L += ["", "> Agreement with the teacher is not accuracy. Accuracy columns use the hidden labels and are research-only."]
    (out / "report.md").write_text("\n".join(L) + "\n")


def plot(out: Path, r: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cps = r["checkpoints"]
    x = [c["examples"] for c in cps]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(x, [100 * (1 - c["student_share_window"]) for c in cps], marker="o")
    ax[0].set_title("Jev share of traffic (last 1,000)")
    ax[0].set_ylim(0, 105)
    ax[0].set_ylabel("%")
    ev = [(c["examples"], c["eval"]) for c in cps if c.get("eval")]
    if ev:
        ax[1].plot([e[0] for e in ev], [100 * e[1]["coverage"] for e in ev], marker="o", label="student coverage")
        ax[1].plot([e[0] for e in ev], [100 * e[1]["system_agreement"] for e in ev], marker="s", label="system agreement")
        ax[1].axhline(100 * r["args"]["target"], ls="--", c="grey", label="target")
        ax[1].set_ylim(0, 105)
        ax[1].legend()
        ax[1].set_title("held-out evaluation")
        ax[2].plot([e[0] for e in ev], [100 * e[1]["system_accuracy_vs_truth"] for e in ev], marker="o", label="system vs truth")
        ax[2].axhline(100 * r["teacher_profile"]["teacher_accuracy_vs_truth"], ls="--", c="grey", label="teacher vs truth")
        ax[2].legend()
        ax[2].set_title("accuracy vs hidden labels (research)")
    for a_ in ax:
        a_.set_xscale("log")
        a_.set_xlabel("examples streamed")
        a_.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "curves.png", dpi=120)


if __name__ == "__main__":
    main()
