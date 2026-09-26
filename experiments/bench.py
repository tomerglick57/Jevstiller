"""Build the benchmark tables from the result files.

    python3 experiments/bench.py [--tag bench] [--write]

Reads experiments/results/<dataset>-cached-<encoder>-probs-t98-<tag>/results.json (a replay per dataset, see
bench.sh) and experiments/results/baselines-<dataset>.json (baselines.py), prints the Markdown tables, and with
`--write` replaces the block between `<!-- bench:start -->` and `<!-- bench:end -->` in docs/benchmarks.md, and the
short version of it (one table, one sentence) between the same markers in README.md.
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
    L += curve_table()
    return "\n".join(L) + "\n"


SHORT_NAMES = {"banking77": "Banking77, 77 intents", "clinc150": "CLINC150, 150 intents", "ag_news": "AG News, 4 sections",
               "tweet_sentiment": "TweetEval sentiment, 3 classes", "tweet_offensive": "TweetEval offensive, 2 classes"}


def build_readme(tag: str, encoder: str) -> str:
    """The README's version: held-out coverage, agreement and accuracy per task, and how often each threshold rule
    broke the budget, so the README can't drift from docs/benchmarks.md."""
    L = ["Five public tasks, replayed through the loop with Jev's recorded answers as the teacher (bge-small on CPU, "
         "target agreement 98%). Answered locally and agreement are measured on 2,000 held-out rows against Jev; "
         "accuracy is against the dataset's own labels, for Jev alone and for the system (student where it answers, "
         "Jev elsewhere).", "",
         "| Task | Answered locally | Agreement with Jev | Accuracy, Jev / system |", "|---|---:|---:|---:|"]
    defer = False
    for d in DATASETS:
        p = ROOT / "experiments" / "results" / f"{d}-cached-{encoder}-probs-t98-{tag}" / "results.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        evals = [c for c in r["checkpoints"] if c.get("eval")]
        if not evals:
            continue
        e = evals[-1]["eval"]
        mark = " †" if r["args"].get("rare_classes") == "defer" else ""
        defer = defer or bool(mark)
        L.append(f"| {SHORT_NAMES.get(d, d)}{mark} | **{e['coverage']:.1%}** | **{e['system_agreement']:.2%}** "
                 f"| {e['teacher_accuracy_vs_truth']:.1%} / {e['system_accuracy_vs_truth']:.1%} |")
    if defer:
        L += ["", "† with `rare_classes = \"defer\"`: Jev never used one of the task's labels."]
    bound_viol = bound_n = 0
    point_viol: list[int] = []
    for d in DATASETS:
        p = ROOT / "experiments" / "results" / f"baselines-{d}.json"
        if not p.exists():
            continue
        runs = json.loads(p.read_text())["runs"]
        bound_viol += sum(x["violated"] for x in runs["soft+bound"])
        bound_n += len(runs["soft+bound"])
        point_viol += [sum(x["violated"] for x in runs[rule]) for rule in ("soft+point", "hard+point")]
    if bound_n:
        per = bound_n // max(1, len(point_viol) // 2)
        L += ["", f"Over {bound_n} random splits across the {len(point_viol) // 2} tasks, the calibrated threshold exceeded the 2% "
              f"budget {'once' if bound_viol == 1 else f'{bound_viol} times'}; the usual point-estimate rule exceeded it on "
              f"{min(point_viol)}–{max(point_viol)} of {per} splits per task."]
    return "\n".join(L) + "\n"


def curve_table() -> list[str]:
    p = ROOT / "experiments" / "results" / "target-curve.json"
    if not p.exists():
        return []
    c = json.loads(p.read_text())
    if not c:
        return []
    targets = [pt["target"] for pt in next(iter(c.values()))["points"]]
    L = ["", "### What the agreement target buys", "",
         "`experiments/target_curve.py`: each replay's production student, its calibration rows, and the loop's own "
         "rule (the bound, with headroom, on the fixed grid) refitted at other targets, then measured on the 2,000 "
         "held-out rows. Each cell is held-out coverage at the agreement with Jev actually reached; the last "
         "column is the system's accuracy against the dataset's labels across all targets, next to Jev's. "
         "`jevstiller admin target <task> 0.95` moves a running task along this curve. The refit uses every calibration "
         "row of the version's lineage, where the loop caps them at `max_calib_samples`, so a long stream (TweetEval "
         "sentiment) can differ by a few points from the replay table.", "",
         "| Task | " + " | ".join(f"target {t:.0%}" for t in targets) + " | Accuracy (Jev / system) |",
         "|---|" + "---:|" * len(targets) + "---|"]
    for d, r in c.items():
        cells = []
        for pt in r["points"]:
            cells.append("–" if pt["threshold"] is None else f"{pt['coverage']:.0%} @ {pt['agreement']:.1%}")
        accs = [pt["accuracy"] for pt in r["points"] if pt["threshold"] is not None]
        acc = f"{r['jev_accuracy']:.1%} / {min(accs):.1%}–{max(accs):.1%}" if accs else f"{r['jev_accuracy']:.1%} / –"
        L.append(f"| {NAMES.get(d, d)} | " + " | ".join(cells) + f" | {acc} |")
    return L


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="bench")
    ap.add_argument("--encoder", default="small")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    table = build(a.tag, a.encoder)
    print(table)
    if a.write:
        start, end = "<!-- bench:start -->", "<!-- bench:end -->"
        for p, block in ((ROOT / "docs" / "benchmarks.md", table), (ROOT / "README.md", build_readme(a.tag, a.encoder))):
            s = p.read_text()
            i, j = s.index(start) + len(start), s.index(end)
            p.write_text(s[:i] + "\n" + block + s[j:])
            print(f"updated {p}")


if __name__ == "__main__":
    main()
