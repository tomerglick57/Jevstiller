# Contributing

Thanks for looking. Jevstiller is small on purpose; the fastest way to help is to run it on a real task and report what happened.

## Set up

```bash
git clone https://github.com/tomerglick57/Jevstiller
cd Jevstiller
python -m pip install -e ".[dev]"      # or: uv sync --extra dev
pytest -q                               # ~15 s, CPU only, no network
ruff check .
```

Real encoders and the Jev adapter are optional extras: `pip install -e ".[onnx]"`, `".[torch]"`, `".[jev]"`. The test suite never needs them.

## Where things live

- `DESIGN.md` — why the loop is shaped the way it is. Read §5 (the guarantee) and §7.8 (the audit channel) before changing routing or calibration.
- `jevstiller/core.py` — the loop. `store.py` — the sample store. `calibrate.py` — the bound.
- `experiments/` — the replay runner; results land in `experiments/results/` (gitignored).

## Rules that keep the guarantee honest

1. Rows from the `deferred` channel never enter the calibration split.
2. Thresholds are chosen against a confidence bound, never a point estimate.
3. Anything shown to a user as "agreement" is agreement with the teacher — never call it accuracy.

A change that touches any of these needs a test.

## Pull requests

- One change per PR; keep the diff readable.
- Add or update a test in `tests/`. The synthetic teacher (`SyntheticTeacher`) is there so tests never need an API key.
- If behaviour changes, update `DESIGN.md` and `CHANGELOG.md` in the same PR.

## Reporting a result

The most valuable issue is "I ran it on task X and here is `status().report()` at 1k / 5k / 20k requests". Include the encoder tier and `Config` values you used.
