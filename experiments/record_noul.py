"""Record Jev's `noul` answer for every message of the yes/no benchmark tasks (experiments/noul_tasks.py).

    python3 experiments/record_noul.py [--task offensive] [--threads 16]     # TYPESAFE_API_KEY from env or .env

Writes experiments/cache/noul_<task>.jsonl, one line per message: text, noul, input tokens, latency, model.
Resumable: messages already recorded are skipped. About 67,000 calls and $0.50 for all four tasks.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jev_profile import load_env  # noqa: E402
from noul_tasks import NOUL_TASKS, rows_for  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def record(name: str, threads: int) -> int:
    import typesafe_sdk as ts
    t = NOUL_TASKS[name]
    path = ROOT / "experiments" / "cache" / f"noul_{name}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if path.exists():
        for line in open(path):
            done.add(json.loads(line)["text"])
    todo = [text for text, _ in rows_for(name) if text not in done]
    print(f"== {name}: {len(done):,} recorded, {len(todo):,} to go", flush=True)
    q = ts.Noul(instructions=t["instructions"], criteria=t["criteria"])
    local, lock, errors, n = threading.local(), threading.Lock(), [0], [0]
    f = open(path, "a")

    def one(text: str) -> None:
        c = getattr(local, "c", None)
        if c is None:
            c = local.c = ts.TypeSafeClient(retry=ts.RetryPolicy(max_retries=8))
        t0 = time.perf_counter()
        try:
            r = c.system_one(state=text, questions={"q": q})
            row = {"text": text, "noul": float(r.nouls["q"].noul), "input_tokens": r.usage.input_tokens,
                   "latency_ms": round((time.perf_counter() - t0) * 1000, 1), "model": r.model}
        except Exception as e:  # recorded on the next run
            with lock:
                errors[0] += 1
                if errors[0] <= 3:
                    print("  error:", repr(e)[:160], flush=True)
            return
        with lock:
            f.write(json.dumps(row) + "\n")
            n[0] += 1
            if n[0] % 2000 == 0:
                f.flush()
                print(f"  {n[0]:,}/{len(todo):,}  errors {errors[0]}", flush=True)

    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(one, todo))
    f.close()
    print(f"  {name}: +{n[0]:,} recorded, {errors[0]} errors", flush=True)
    return errors[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="all", choices=["all", *NOUL_TASKS])
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    load_env()
    errors = sum(record(n, a.threads) for n in (NOUL_TASKS if a.task == "all" else [a.task]))
    print("ALL-DONE" if not errors else f"DONE-WITH-ERRORS {errors} (run again to fill them)", flush=True)


if __name__ == "__main__":
    main()
