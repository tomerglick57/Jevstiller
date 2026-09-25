# Contributing

Thanks for looking. Jevstiller is small on purpose; the fastest way to help is to run it on a real task and report what happened.

## Set up

```bash
git clone https://github.com/tomerglick57/Jevstiller
cd Jevstiller
python -m pip install -e ".[dev]"      # or: uv sync --extra dev   (includes the jev + server extras)
pytest -q                               # ~1 min, CPU only, no network
ruff check .
```

Real encoders are optional extras (`".[onnx]"`, `".[torch]"`); the test suite never needs them.

Live tests call the real Jev API (a fraction of a cent) and are skipped unless asked for:

```bash
JEVSTILLER_LIVE=1 pytest -m live       # TYPESAFE_API_KEY from the environment or .env
python experiments/jev_profile.py      # re-record tests/fixtures/jev/ and measure Jev latency
```

Performance claims come from `benchmarks/` (see `benchmarks/README.md`); every number in the docs is listed with its command in `docs/benchmarks.md`.

## Where things live

- `DESIGN.md` — why the loop is shaped the way it is. Read §5 (the guarantee) and §7.8 (the audit channel) before changing routing or calibration.
- `DEPLOYMENT_PLAN.md` — what is done and what is next, with the measurements behind each decision.
- `docs/` — the proxy (`proxy.md`), every setting (`configuration.md`), every measurement (`benchmarks.md`).
- `jevstiller/_core.py` — one task's loop (`route` / `complete`). `_store.py` — the sample store. `_calibrate.py` — the bound. `_training.py` — the training job. `_registry.py` — versions.
- `jevstiller/_manager.py` — many tasks in one process. `_scheduler.py` — shared training. `_server.py` / `_cli.py` — the proxy.
- **The public API** is `jevstiller.__all__` plus `jevstiller.server`, `jevstiller.encoders`, `jevstiller.teachers` and `jevstiller.teachers.jev`. Modules starting with `_` are internal. `tests/test_public_api.py` pins the public API against `tests/public_api.txt`: a change there is a deliberate one. Regenerate it with `python tests/test_public_api.py --update` and say what changed in `CHANGELOG.md`.
- `experiments/` — replay runners; results land in `experiments/results/` (gitignored). `benchmarks/` — performance scripts.

## Rules that keep the guarantee honest

1. Rows from the `deferred` channel never enter the calibration split.
2. Thresholds are chosen against a confidence bound, never a point estimate.
3. Anything shown to a user as "agreement" is agreement with the teacher — never call it accuracy.
4. Training, calibration and audit statistics use one teacher lineage (`teacher_model`) only.
5. The proxy answers locally only for an API key Jev has accepted, and never stores or logs a key.

A change that touches any of these needs a test.

## Pull requests

- One change per PR; keep the diff readable.
- Add or update a test in `tests/`. The synthetic teacher (`SyntheticTeacher`) is there so tests never need an API key.
- If behaviour changes, update `DESIGN.md`, the relevant `docs/` page and `CHANGELOG.md` in the same PR. If a performance number changes, update `docs/benchmarks.md`.

## Reporting a result

The most valuable issue is "I ran it on task X and here is `status().report()` at 1k / 5k / 20k requests". Include the encoder tier and `Config` values you used.
