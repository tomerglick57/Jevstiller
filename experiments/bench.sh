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
RECORD=1
[ "${1:-}" = "--no-record" ] && RECORD=0

for t in $TASKS; do
    if [ -f "experiments/cache/$t.jsonl.gz" ] && [ ! -f "experiments/cache/$t.jsonl" ]; then
        gunzip -k "experiments/cache/$t.jsonl.gz"
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
"$PY" experiments/bench.py --write
