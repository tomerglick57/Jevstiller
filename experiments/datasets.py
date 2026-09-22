"""Dataset loaders for the validation experiment. Labels are hidden from Jevstiller; used only for research metrics."""
from __future__ import annotations

import csv
import json
import urllib.request
from pathlib import Path

DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)


def _fetch(url: str, name: str) -> Path:
    p = DATA / name
    if not p.exists():
        with urllib.request.urlopen(url) as r:
            p.write_bytes(r.read())
    return p


def banking77() -> dict:
    base = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/"
    rows = []
    for split in ("train", "test"):
        with open(_fetch(base + f"{split}.csv", f"banking77_{split}.csv"), newline="") as f:
            rows += [(r["text"], r["category"]) for r in csv.DictReader(f)]
    labels = sorted({c for _, c in rows})
    return {"name": "banking77", "rows": rows,
            "instructions": "What is the customer asking about? Pick the banking intent that best matches the message.",
            "classes": {c: c.replace("_", " ") for c in labels}}


def clinc150() -> dict:
    p = _fetch("https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json", "clinc150_full.json")
    d = json.loads(p.read_text())
    rows = []
    for _k, v in d.items():
        for text, label in v:
            rows.append((text, "other" if label == "oos" else label))
    labels = sorted({c for _, c in rows if c != "other"})
    classes = {c: c.replace("_", " ") for c in labels}
    classes["other"] = "None of the above: the request does not match any listed intent"
    return {"name": "clinc150", "rows": rows,
            "instructions": "What does the user want? Pick the intent that best matches the utterance.",
            "classes": classes}


def ag_news() -> dict:
    from datasets import load_dataset
    ds = load_dataset("fancyzhx/ag_news")
    names = ["world", "sports", "business", "scitech"]
    desc = {"world": "World news: international affairs, politics, conflicts, disasters",
            "sports": "Sports: games, athletes, teams, results",
            "business": "Business: companies, markets, economy, finance",
            "scitech": "Science and technology: research, software, hardware, internet, space"}
    rows = [(r["text"], names[r["label"]]) for split in ("train", "test") for r in ds[split]]
    return {"name": "ag_news", "rows": rows,
            "instructions": "Which section of a news site does this article belong to?", "classes": desc}


def synthetic(n: int = 20000, seed: int = 0) -> dict:
    from jevstiller import SyntheticWorld
    labels = ["billing", "technical", "cancellation", "sales", "other"]
    world = SyntheticWorld(labels, seed=seed)
    return {"name": "synthetic", "rows": world.sample(n), "instructions": "Which team handles this?",
            "classes": {c: "" for c in labels}, "world": world}


LOADERS = {"banking77": banking77, "clinc150": clinc150, "ag_news": ag_news, "synthetic": synthetic}
