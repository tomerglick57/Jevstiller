#!/usr/bin/env bash
# Reproduce the headline Banking77 result from the README, with no API key and no GPU.
#
# Jev's answers for every Banking77 message were recorded on 2026-09-24/26 and ship with the repo
# (experiments/cache/banking77.jsonl.gz), so the replay never calls Jev. The dataset itself (~1 MB of CSV)
# is downloaded from the PolyAI GitHub repository on first use. bge-small runs on CPU through ONNX Runtime;
# the first run downloads it from Hugging Face (~130 MB). Expect 10-15 minutes on a laptop.
#
#   pip install jevstiller            # or: uv sync
#   bash experiments/reproduce.sh     # extra run.py flags are passed through
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
if [ -x .venv/bin/python ]; then PY=.venv/bin/python; fi

if [ ! -f experiments/cache/banking77.jsonl ]; then
    gunzip -k experiments/cache/banking77.jsonl.gz
fi

"$PY" experiments/run.py --dataset banking77 --teacher cached --encoder small --backend onnx --device cpu --tag reproduce "$@"

"$PY" - <<'PYEOF'
import json
r = json.load(open("experiments/results/banking77-cached-small-probs-t98-reproduce/results.json"))
e = [c for c in r["checkpoints"] if c.get("eval")][-1]["eval"]
cache = r["teacher_cache"]
print()
print("                              this run   shipped replay   live run (README)")
print(f"  held-out coverage           {e['coverage']:6.1%}       71.9%            70.7%")
print(f"  system agreement with Jev   {e['system_agreement']:6.2%}      99.50%           99.45%   (target 98%)")
print(f"  accuracy vs true labels     {e['system_accuracy_vs_truth']:6.1%}       78.5%            78.5%   (Jev alone: {e['teacher_accuracy_vs_truth']:.1%})")
print(f"  Jev answers from the cache  {cache['hits']:,} hits, {cache['misses']} misses")
print()
print("The replay is deterministic, so 'this run' should match 'shipped replay' exactly on the same kind of CPU;")
print("another CPU's floating point can move the numbers by a few tenths of a point. The live run of 2026-09-25")
print("saw Jev's answers for ~650 messages that were re-recorded since, and used a different train/calibration split.")
PYEOF
