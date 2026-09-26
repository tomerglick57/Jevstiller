"""Build the benchmark tables from the result files.

    python3 experiments/bench.py [--tag bench] [--write]

Reads experiments/results/<dataset>-cached-<encoder>-probs-t98-<tag>/results.json (a replay per dataset, see
bench.sh) and experiments/results/baselines-<dataset>.json (baselines.py), prints two Markdown tables, and with
`--write` replaces the block between `<!-- bench:start -->` and `<!-- bench:end -->` in docs/benchmarks.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from datasets import LOADERS  # noqa: E402

DATASETS = [d for d in LOADERS if d != "synthetic"]
NAMES = {"banking77": "Banking77 (77 intents)", "clinc150": "CLINC150 (150 intents + other, 12k sample)",
         "ag_news": "AG News (4 sections, 20k sample)", "tweet_sentiment": "TweetEval sentiment (3)",
         "tweet_offensive": "TweetEval offensive (2)"}


def replay_row(d: str, tag: str, encoder: str) -> str | None:
    p = ROOT / "experiments" / "results" / f"{d}-cached-{encoder}-probs-t98-{tag}" / "results.json"
    if not p.exists():
        return None
    r = json.loads(p.read_text())
    fs = r["final_status"]
    evals = [c for c in r["checkpoints"] if c.get("eval")]
    if not evals:
        return (f"| {NAMES.get(d, d)} | {r['n_stream']:,} | no student | – | {r['teacher_profile']['teacher_accuracy_vs_truth']:.1%} / – "
                f"| 0% | – | ${fs.get('teacher_cost_usd', 0.0) + r['teacher_profile']['cost_usd']:.2f} |")
    e = evals[-1]["eval"]
    d = d + " †" if r["args"].get("rare_classes") == "defer" else d
    cost = fs.get("teacher_cost_usd", 0.0) + r["teacher_profile"]["cost_usd"]
    rps = r["burst"].get("student_rows_per_s")
    name = NAMES.get(d.rstrip(" †"), d.rstrip(" †")) + (" †" if d.endswith("†") else "")
    return (f"| {name} | {r['n_stream']:,} | **{e['coverage']:.1%}** | **{e['system_agreement']:.2%}** "
            f"| {e['teacher_accuracy_vs_truth']:.1%} / {e['system_accuracy_vs_truth']:.1%} "
            f"| {fs['student_share']:.0%} | {rps:,.0f}/s | ${cost:.2f} |" if rps else
            f"| {name} | {r['n_stream']:,} | {e['coverage']:.1%} | {e['system_agreement']:.2%} "
            f"| {e['teacher_accuracy_vs_truth']:.1%} / {e['system_accuracy_vs_truth']:.1%} | {fs['student_share']:.0%} | – | ${cost:.2f} |")


def baseline_rows(d: str) -> list[str]:
    p = ROOT / "experiments" / "results" / f"baselines-{d}.json"
    if not p.exists():
        return []
    r = json.loads(p.read_text())
    rows = []
    for rule in ("soft+bound", "hard+bound", "soft+point", "hard+point"):
        runs = r["runs"][rule]
        cov = sum(x["coverage"] for x in runs) / len(runs)
        dis = sum(x["disagreement"] for x in runs) / len(runs)
        worst = max(x["disagreement"] for x in runs)
        viol = sum(x["violated"] for x in runs)
        acc = sum(x["system_accuracy"] for x in runs) / len(runs)
        label = {"soft+bound": "**Jevstiller** (soft labels, bound, OOD gate)", "hard+bound": "hard labels, bound, OOD gate",
                 "soft+point": "soft labels, point estimate", "hard+point": "hard labels, point estimate (stuntd's recipe)"}[rule]
        rows.append(f"| {NAMES.get(d, d)} | {label} | {cov:.1%} | {dis:.2%} | {worst:.2%} | {viol}/{len(runs)} | {acc:.1%} |")
    return rows


def build(tag: str, encoder: str) -> str:
    L = ["### Quality: the loop against live Jev's answers", "",
         "One replay per task (`experiments/bench.sh`): the dataset's labels are hidden, 2,000 rows are held out, "
         "the rest stream through the loop with Jev's recorded answers as the teacher (bge-small on CPU, target "
         "agreement 98%, audit rate 2%). Coverage and agreement are measured on the held-out rows against Jev; "
         "accuracy is against the dataset's own labels, for Jev alone and for the system (student where it "
         "answers, Jev elsewhere). \"Local share\" is over the whole stream, cold start included.", "",
         "| Task | Stream | Held-out coverage | Agreement with Jev | Accuracy: Jev / system | Local share | Student path | Jev cost |",
         "|---|---:|---:|---:|---:|---:|---:|---:|"]
    got = [replay_row(d, tag, encoder) for d in DATASETS]
    L += [g for g in got if g] or ["| (no replays yet) | | | | | | | |"]
    if any(g and "†" in g for g in got):
        L += ["", "† run with `rare_classes = \"defer\"`: Jev never used one of the task's labels, and the default "
              "(`wait`) would have kept the first student from training for the whole stream."]
    L += ["", "### Threshold rules on the same data", "",
          "`experiments/baselines.py`: 20 random train / calibration / test splits per task, one head per split, "
          "then the confidence threshold is picked on the calibration split by each rule and judged on the test "
          "split. Disagreement is the share of *all* test requests the student answered differently from Jev, "
          "which is what the 2% budget limits; a violation is a split where it exceeded the budget.", "",
          "| Task | Rule | Coverage | Disagreement (mean) | Disagreement (worst) | Violations | System accuracy |",
          "|---|---|---:|---:|---:|---:|---:|"]
    rows = [r for d in DATASETS for r in baseline_rows(d)]
    L += rows or ["| (not run yet) | | | | | | |"]
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="bench")
    ap.add_argument("--encoder", default="small")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    table = build(a.tag, a.encoder)
    print(table)
    if a.write:
        p = ROOT / "docs" / "benchmarks.md"
        s = p.read_text()
        start, end = "<!-- bench:start -->", "<!-- bench:end -->"
        i, j = s.index(start) + len(start), s.index(end)
        p.write_text(s[:i] + "\n" + table + s[j:])
        print(f"updated {p}")


if __name__ == "__main__":
    main()
