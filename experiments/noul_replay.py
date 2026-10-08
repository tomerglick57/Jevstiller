"""Replay a yes/no (`noul`) task through the loop, with Jev's recorded probabilities as the teacher.

    python3 experiments/noul_replay.py --task sports [--cutoffs 0.5] [--target 0.98]

The stream goes through `Jevstiller` exactly as a proxy task would see it (bootstrap, training, shadow,
promotion, audit), with experiments/cache/noul_<task>.jsonl standing in for Jev (record_noul.py; no API key).
2,000 rows are held out and scored at the end with `evaluate`: coverage, and agreement with Jev on the
outcome its probability stands for under the cut-offs. Writes experiments/results/noul-replay-<task>.json.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from noul_tasks import NOUL_TASKS, rows_for  # noqa: E402

from jevstiller import Config, Jevstiller, Task, load_encoder  # noqa: E402
from jevstiller._task import noul_labels  # noqa: E402
from jevstiller.teachers import TeacherOutput  # noqa: E402


class RecordedNoul:
    """Jev's recorded yes probability for each text, in the shape a teacher returns."""
    name = "jev:jev-1.13.0"

    def __init__(self, rec: dict[str, dict]):
        self.rec = rec

    def classify(self, texts, task):
        out = []
        for t in texts:
            d = self.rec[t]
            p = float(d["noul"])
            out.append(TeacherOutput("true" if p >= 0.5 else "false", {"false": 1 - p, "true": p}, abs(2 * p - 1),
                                     input_tokens=d["input_tokens"], cost_usd=d["input_tokens"] * 0.042 / 1e6,
                                     latency_ms=d["latency_ms"], model=f"jev:{d['model']}"))
        return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="sports", choices=list(NOUL_TASKS))
    ap.add_argument("--cutoffs", default="0.5", help="one cut-off, or two for an unsure band: 0.4,0.6")
    ap.add_argument("--target", type=float, default=0.98)
    ap.add_argument("--eval-size", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--encoder", default="small")
    a = ap.parse_args()
    cutoffs = tuple(float(x) for x in a.cutoffs.split(","))
    spec = NOUL_TASKS[a.task]
    rec = {}
    for line in open(ROOT / "experiments" / "cache" / f"noul_{a.task}.jsonl"):
        d = json.loads(line)
        rec[d["text"]] = d
    rows = [(t, y) for t, y in rows_for(a.task) if t in rec]
    rng = np.random.default_rng(a.seed)
    rng.shuffle(rows)
    ev, stream = rows[:a.eval_size], rows[a.eval_size:]
    task = Task.noul(a.task, spec["instructions"], spec["criteria"], cutoffs=cutoffs, target_agreement=a.target)
    tag = a.cutoffs.replace(",", "-")
    out = ROOT / "experiments" / "results" / f"noul-replay-{a.task}-c{tag}"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    enc = load_encoder(a.encoder, backend="onnx", device="cpu")
    js = Jevstiller(task, RecordedNoul(rec), out / "state", encoder=enc, config=Config(seed=a.seed, training="inline"))
    t0 = time.perf_counter()
    first_local = None
    for i in range(0, len(stream), a.batch):
        res = js.classify_batch([t for t, _ in stream[i:i + a.batch]])
        if first_local is None and any(r.source != "teacher" for r in res):
            first_local = i
        if (i // a.batch) % 40 == 39:
            st = js.status()
            print(f"[{i + a.batch:>6,}] prod={st.production or '-':>11}  student {st.student_share:.1%}", flush=True)
    st = js.status()
    ev_texts = [t for t, _ in ev]
    ev_p = np.array([rec[t]["noul"] for t in ev_texts])
    truth = np.array([y for _, y in ev])
    e = js.evaluate(ev_texts, ev_p)
    result = {"task": a.task, "cutoffs": list(cutoffs), "target": a.target, "n_stream": len(stream), "n_eval": len(ev),
              "production": st.production, "student_share_stream": st.student_share, "first_local_at": first_local,
              "teacher_calls": st.teacher_calls, "audit_n": st.audit_n, "audit_agreement": st.audit_agreement,
              "seconds": round(time.perf_counter() - t0, 1), "date": time.strftime("%Y-%m-%d")}
    if e.get("version"):
        jev_out = noul_labels(ev_p, cutoffs)
        yes = np.where(e["accepted"], e["student_label"], jev_out) == "true"
        result.update(coverage=e["coverage"], system_agreement=e["system_agreement"],
                      selective_disagreement=e["selective_disagreement"],
                      jev_accuracy=float(((ev_p >= 0.5) == truth).mean()) if len(cutoffs) == 1 else None,
                      system_accuracy=float((yes == truth).mean()) if len(cutoffs) == 1 else None)
    (out / "results.json").write_text(json.dumps(result, indent=1, default=float))
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in result.items()}, indent=1))
    print(st.report(events=False))
    js.close()
    shutil.rmtree(out / "state", ignore_errors=True)


if __name__ == "__main__":
    main()
