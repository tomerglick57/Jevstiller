---
title: Contributing
description: Set up a dev environment, the rules that keep the guarantee honest, and how to send a PR.
---

Jevstiller is small on purpose. The fastest way to help is to run it on a real task and report what happened.

## Set up

```bash
git clone https://github.com/tomerglick57/Jevstiller
cd Jevstiller
python -m pip install -e ".[dev]"      # or: uv sync --extra dev
pytest -q                               # CPU only, no network
ruff check .
```

The test suite never needs an API key or a real encoder; `SyntheticTeacher` drives the whole loop.

## Rules that keep the guarantee honest

1. Rows that went to Jev because the student was unsure never enter the calibration split.
2. Thresholds are chosen against a confidence bound, never a point estimate.
3. Anything shown as "agreement" is agreement with the teacher. Never call it accuracy.

A change that touches any of these needs a test.

## Pull requests

- One change per PR.
- Add or update a test in `tests/`.
- If behaviour changes, update `DESIGN.md` and `CHANGELOG.md` in the same PR.

## This website

The site lives in `website/` and is built with [Starlight](https://starlight.astro.build). Pages are Markdown
files in `website/src/content/docs/`.

```bash
cd website
npm install
npm run dev
```

Full guide: [CONTRIBUTING.md](https://github.com/tomerglick57/Jevstiller/blob/main/CONTRIBUTING.md).
Security issues: [SECURITY.md](https://github.com/tomerglick57/Jevstiller/blob/main/SECURITY.md).
