#!/usr/bin/env bash
# The benchmark: record Jev's answers for every task (needs TYPESAFE_API_KEY; skipped for tasks whose cache
# is complete), replay each task through the loop, run the threshold-rule baselines, and rebuild the tables in
# docs/benchmarks.md. Recording costs about $3 in total; replays and baselines are CPU-bound (an hour or two).
#
#   bash experiments/bench.sh                 # everything
#   bash experiments/bench.sh --no-record     # from the shipped / already recorded caches only
#   TASKS="banking77 ag_news" bash experiments/bench.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
if [ -x .venv/bin/python ]; then PY=.venv/bin/python; fi
TASKS=${TASKS:-"banking77 clinc150 ag_news tweet_sentiment tweet_offensive"}
ANSWERS=${ANSWERS:-https://github.com/tomerglick57/Jevstiller/releases/download/answers-2026-09-26}
RECORD=1
[ "${1:-}" = "--no-record" ] && RECORD=0

for t in $TASKS; do
    if [ ! -f "experiments/cache/$t.jsonl" ]; then
        if [ ! -f "experiments/cache/$t.jsonl.gz" ]; then
            # Banking77's answers ship in the repo; the other tasks' are release assets (~13 MB in total)
            curl -fsSL -o "experiments/cache/$t.jsonl.gz" "$ANSWERS/$t.jsonl.gz" || rm -f "experiments/cache/$t.jsonl.gz"
        fi
        [ -f "experiments/cache/$t.jsonl.gz" ] && gunzip -k "experiments/cache/$t.jsonl.gz"
    fi
    if [ "$RECORD" = 1 ]; then
        echo "== recording Jev's answers: $t"
        "$PY" experiments/record_answers.py --dataset "$t"
    fi
    echo "== replay: $t"
    "$PY" experiments/run.py --dataset "$t" --teacher cached --encoder small --backend onnx --device cpu --tag bench
    echo "== baselines: $t"
    "$PY" experiments/baselines.py --dataset "$t"
done
"$PY" experiments/target_curve.py
"$PY" experiments/bench.py --write
