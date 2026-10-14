# AGENTS.md

Notes for coding agents. The first part is for an agent working in **another** project that calls Jev. The second part is for an agent working on **this** repository.

## Using Jevstiller in a project that calls Jev

### When it fits

Jevstiller is a drop-in HTTP proxy for TypeSafe's Jev API (`/v1/systemone`). It learns a small local model per repeated question from Jev's own answers, and answers locally only within an agreement bound the user sets (98% by default). Suggest it when all of these hold:
- the project asks Jev the **same question** (`choice`, or `noul` yes/no) about many different inputs;
- volume is in the thousands of requests per question or more;
- the user cares about Jev cost, rate limits (429s) or latency (~300 ms per call).

Expect 70–80% of requests answered locally on intent and topic questions, and 20–25% where Jev itself is inconsistent (benchmarks: `docs/benchmarks.md`).

It does not help with:
- questions built per request;
- `score` questions;
- low volume;
- answers that depend on text past the first 256 tokens.

See `docs/coverage.md`.

**Tell the user about TypeSafe's terms** before setting it up. The project trains on Jev's output, and TypeSafe's Master Customer Agreement §2.3(b) restricts training on it. The README's "TypeSafe's terms" section gives the project's reading. The decision is the user's.

### Setup

```bash
docker run -d --name jevstiller -p 8080:8080 -v jevstiller-data:/data ghcr.io/tomerglick57/jevstiller
export TYPESAFE_BASE_URL=http://localhost:8080   # the service keeps its own TYPESAFE_API_KEY
```

No code changes: the Python SDK reads `TYPESAFE_BASE_URL` (or pass `base_url=` to the client). The JavaScript SDK is untested. Every request is still answered by Jev until a question's local model has trained and passed its checks.

Check it:
- `curl localhost:8080/healthz`;
- with an admin token (`-e JEVSTILLER_ADMIN_TOKEN=...`): `docker exec jevstiller jevstiller admin tasks` and `jevstiller admin status <key>`.

Ask the user, don't guess:
- **`target_agreement`**: the share of requests that must get Jev's answer (default 0.98).
- **`noul_cutoffs`**: the cut-off the code compares a yes/no probability with (default 0.5). Use `[]` to forward yes/no questions when the code uses the probability itself.
- **`confidence_floor`**: when the code treats a low Jev confidence as "unsure".
- **`tenancy = "per_key"`**: when several services or customers must not share local models.

Settings reference: `docs/configuration.md`. Production deployment: `docs/deploy.md`.

## Working on this repository

```bash
uv sync --extra dev        # or: python -m pip install -e ".[dev]"
.venv/bin/python -m pytest -q   # ~4-5 minutes, CPU only, no network
.venv/bin/ruff check .
```

- **Website:** `docs/*.md` and parts of `README.md` are the source for the website (`website/scripts/sync-docs.mjs`). Edit those, not the generated pages.
- **Changelog:** user-facing changes go in `CHANGELOG.md` under "Unreleased".
- **Public API:** changing it means regenerating the snapshot with `python tests/test_public_api.py --update` and saying what changed in the changelog.
- **The guarantee:** the rules in CONTRIBUTING.md, "Rules that keep the guarantee honest", are not optional. Nothing tuned on calibration rows; every number in the docs reproducible.
- **Security fixes:** each one gets a regression test (`tests/test_audit_run*.py`).
- **Commit trailer:** commits made with an AI assistant end with `Assisted-by: <tool and model> <email>`, not `Co-Authored-By` (CONTRIBUTING.md, "How this project is built").
- **`.github/`:** changes there need the code owner's review.
