"""Profile live Jev: latency under concurrency (P0.4) and real wire samples for contract tests (P0.3).

    python experiments/jev_profile.py                   # needs TYPESAFE_API_KEY (.env is read)
    python experiments/jev_profile.py --no-fixtures
    python experiments/jev_profile.py --model jev-1.14.0 --fixtures-only   # 2 requests: that model's answers

Writes `tests/fixtures/jev/*.json` (status, the headers that matter, body; no keys: responses never carry one)
and `experiments/results/jev-profile.json`. About 420 requests: cents.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "jev"
KEEP_HEADERS = {"content-type", "x-typesafe-request-id", "retry-after", "retry-after-ms"}
QUESTION = {"type": "choice", "instructions": "Which team should handle this customer message?",
            "criteria": {"billing": "Charges, invoices, refunds, payment methods",
                         "technical": "Bugs, errors, integrations, things not working",
                         "cancellation": "Wants to cancel, downgrade, or close the account",
                         "sales": "Pre-sales questions, plan comparison, quotes",
                         "other": "Anything that does not fit the categories above"}}
MESSAGES = ["I was charged twice this month", "The app crashes when I upload a file", "Please close my account",
            "Do you offer an enterprise plan?", "What's the weather like?", "My invoice shows the wrong VAT number",
            "Webhooks stopped firing yesterday", "I want to downgrade to the free tier"]


def load_env() -> None:
    p = ROOT / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def record(client: httpx.Client, name: str, method: str, path: str, **kw) -> httpx.Response:
    r = client.request(method, path, **kw)
    FIXTURES.mkdir(parents=True, exist_ok=True)
    try:
        body = r.json()
    except ValueError:
        body = r.text
    (FIXTURES / f"{name}.json").write_text(json.dumps({
        "request": {"method": method, "path": path, "json": kw.get("json")},
        "status": r.status_code,
        "headers": {k: v for k, v in r.headers.items() if k.lower() in KEEP_HEADERS},
        "body": body}, indent=2) + "\n")
    print(f"  fixture {name}: {r.status_code}")
    return r


def fixtures(base: str, key: str, model: str = "jev-latest") -> None:
    """The answers of `model` (fixture names get a `_<model>` suffix unless it is jev-latest) and, for jev-latest,
    the error responses and the model list too, which don't depend on the model."""
    h = {"Authorization": f"Bearer {key}"}
    sfx = "" if model == "jev-latest" else f"_{model}"
    with httpx.Client(base_url=base, timeout=30) as c:
        record(c, f"systemone_choice_200{sfx}", "POST", "/v1/systemone", headers=h,
               json={"state": MESSAGES[0], "model": model, "questions": {"label": QUESTION}})
        record(c, f"systemone_mixed_200{sfx}", "POST", "/v1/systemone", headers=h,
               json={"state": {"subject": "Refund", "body": MESSAGES[5]}, "model": model,
                     "questions": {"label": QUESTION,
                                   "urgent": {"type": "noul", "instructions": "Does the customer need an answer today?"},
                                   "tone": {"type": "score", "instructions": "How upset is the customer?",
                                            "criteria": ["calm", "annoyed", "angry"]}}})
        if sfx:
            return
        record(c, "systemone_401", "POST", "/v1/systemone", headers={"Authorization": "Bearer tsk_invalid_key"},
               json={"state": "x", "model": "jev-latest", "questions": {"label": QUESTION}})
        record(c, "systemone_422", "POST", "/v1/systemone", headers=h,
               json={"state": "x", "model": "jev-latest", "questions": {}})
        record(c, "models_200", "GET", "/v1/models", headers=h)


def latency(base: str, key: str, levels: list[int], per_level: int) -> dict:
    h = {"Authorization": f"Bearer {key}"}
    out = {}
    with httpx.Client(base_url=base, timeout=30, limits=httpx.Limits(max_connections=64)) as c:
        def one(i: int) -> tuple[int, float, str | None]:
            t0 = time.perf_counter()
            r = c.post("/v1/systemone", headers=h, json={"state": MESSAGES[i % len(MESSAGES)] + f" (#{i})",
                                                         "model": "jev-latest", "questions": {"label": QUESTION}})
            return r.status_code, (time.perf_counter() - t0) * 1000, r.headers.get("retry-after")
        for conc in levels:
            n = per_level * max(1, conc // 2)
            t0 = time.perf_counter()
            with ThreadPoolExecutor(conc) as ex:
                res = list(ex.map(one, range(n)))
            wall = time.perf_counter() - t0
            ok = sorted(ms for s, ms, _ in res if s == 200)
            statuses: dict[str, int] = {}
            for s, _, _ in res:
                statuses[str(s)] = statuses.get(str(s), 0) + 1
            def q(p: float, ok: list[float] = ok) -> float:
                return ok[min(len(ok) - 1, int(p * len(ok)))] if ok else float("nan")
            out[conc] = {"n": n, "statuses": statuses, "p50_ms": q(.5), "p90_ms": q(.9), "p99_ms": q(.99),
                         "mean_ms": statistics.fmean(ok) if ok else None, "req_per_s": n / wall,
                         "retry_after": sorted({ra for _, _, ra in res if ra})}
            print(f"  concurrency {conc:>2}: n={n}  {statuses}  p50 {q(.5):.0f} ms  p90 {q(.9):.0f}  "
                  f"p99 {q(.99):.0f}  {n / wall:.1f} req/s", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fixtures", action="store_true")
    ap.add_argument("--fixtures-only", action="store_true", help="no latency profile")
    ap.add_argument("--model", default="jev-latest", help="record this model's answers as extra fixtures")
    ap.add_argument("--levels", default="1,8,16")
    ap.add_argument("--per-level", type=int, default=30)
    a = ap.parse_args()
    load_env()
    key = os.environ["TYPESAFE_API_KEY"]
    base = os.environ.get("TYPESAFE_BASE_URL", "https://api.typesafe.ai")
    if not a.no_fixtures:
        print("recording fixtures ...")
        fixtures(base, key, a.model)
    if a.fixtures_only:
        raise SystemExit(0)
    print("latency ...")
    lat = latency(base, key, [int(x) for x in a.levels.split(",")], a.per_level)
    out = ROOT / "experiments" / "results" / "jev-profile.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"date": time.strftime("%Y-%m-%d"), "base_url": base, "latency": lat}, indent=2))
    print(f"wrote {out}")
