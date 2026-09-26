"""Record Jev's answer for every message of a dataset into experiments/cache/<dataset>.jsonl.

A replay only asks Jev about the rows it routes to the teacher, so a cache built from replays alone has holes,
and a later replay that routes slightly differently finds messages Jev never answered. This fills the cache
completely, so `experiments/reproduce.sh` (`--teacher cached`) can run without an API key. Banking77 is
~13k messages: about 6 minutes at Jev's rate limit and $0.35.

    python3 experiments/record_answers.py --dataset banking77      # TYPESAFE_API_KEY from the environment or .env
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from datasets import LOADERS  # noqa: E402
from jev_profile import load_env  # noqa: E402

from jevstiller import Task  # noqa: E402
from jevstiller.teachers import CachedTeacher  # noqa: E402
from jevstiller.teachers.jev import JevTeacher  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="banking77", choices=[d for d in LOADERS if d != "synthetic"])
    ap.add_argument("--target", type=float, default=0.98, help="must match run.py's --target (part of the task version)")
    ap.add_argument("--batch", type=int, default=200)
    a = ap.parse_args()
    load_env()
    ds = LOADERS[a.dataset]()
    task = Task(name=a.dataset, instructions=ds["instructions"], classes=ds["classes"], target_agreement=a.target)
    root = Path(__file__).resolve().parents[1]
    teacher = CachedTeacher(JevTeacher(), root / "experiments" / "cache" / f"{a.dataset}.jsonl")
    texts = [t for t, _ in ds["rows"]]
    errors = 0
    for i in range(0, len(texts), a.batch):
        out = teacher.classify(texts[i:i + a.batch], task)
        errors += sum(isinstance(o, Exception) for o in out)
        print(f"{min(i + a.batch, len(texts)):>6,}/{len(texts):,}  cache hits {teacher.hits:,}  Jev calls {teacher.misses:,}  errors {errors}",
              flush=True)
    if errors:
        print(f"{errors} answers failed; run again to fetch them.")
        sys.exit(1)
    print(f"complete: {len(teacher.cache):,} answers in {teacher.path}")


if __name__ == "__main__":
    main()
