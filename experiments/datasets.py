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


def clinc150(n: int = 12_000, seed: int = 0) -> dict:
    """A fixed 12,000-row sample of CLINC150 (23,700 rows; every Jev call carries 151 class descriptions)."""
    import random
    p = _fetch("https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json", "clinc150_full.json")
    d = json.loads(p.read_text())
    rows = []
    for _k, v in d.items():
        for text, label in v:
            rows.append((text, "other" if label == "oos" else label))
    random.Random(seed).shuffle(rows)
    rows = rows[:n]
    labels = sorted({c for _, c in rows if c != "other"})
    classes = {c: c.replace("_", " ") for c in labels}
    classes["other"] = "None of the above: the request does not match any listed intent"
    return {"name": "clinc150", "rows": rows,
            "instructions": "What does the user want? Pick the intent that best matches the utterance.",
            "classes": classes}


def ag_news(n: int = 20_000, seed: int = 0) -> dict:
    """A fixed 20,000-row sample of AG News (127,600 rows would cost ~$2 of Jev answers per recording)."""
    import random
    base = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/"
    names = ["world", "sports", "business", "scitech"]
    desc = {"world": "World news: international affairs, politics, conflicts, disasters",
            "sports": "Sports: games, athletes, teams, results",
            "business": "Business: companies, markets, economy, finance",
            "scitech": "Science and technology: research, software, hardware, internet, space"}
    rows = []
    for split in ("train", "test"):
        with open(_fetch(base + f"{split}.csv", f"ag_news_{split}.csv"), newline="", encoding="utf-8") as f:
            for cls, title, body in csv.reader(f):
                rows.append((f"{title}. {body}".replace("\\", " "), names[int(cls) - 1]))
    random.Random(seed).shuffle(rows)
    return {"name": "ag_news", "rows": rows[:n],
            "instructions": "Which section of a news site does this article belong to?", "classes": desc}


def _tweeteval(task: str) -> list[tuple[str, str]]:
    base = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/"
    names = _fetch(base + "mapping.txt", f"tweeteval_{task}_mapping.txt").read_text().splitlines()
    mapping = {line.split("\t")[0]: line.split("\t")[1].strip() for line in names if line.strip()}
    rows = []
    for split in ("train", "val", "test"):
        texts = _fetch(base + f"{split}_text.txt", f"tweeteval_{task}_{split}_text.txt").read_text().splitlines()
        labels = _fetch(base + f"{split}_labels.txt", f"tweeteval_{task}_{split}_labels.txt").read_text().splitlines()
        rows += [(t.strip(), mapping[l.strip()]) for t, l in zip(texts, labels, strict=True) if t.strip()]
    return rows


def tweet_sentiment() -> dict:
    """TweetEval sentiment: 59,899 tweets, negative / neutral / positive."""
    return {"name": "tweet_sentiment", "rows": _tweeteval("sentiment"),
            "instructions": "What is the overall sentiment of this tweet?",
            "classes": {"negative": "Negative: complaint, anger, sadness, disapproval",
                        "neutral": "Neutral: factual, mixed or no clear sentiment",
                        "positive": "Positive: praise, joy, approval, excitement"}}


def tweet_offensive() -> dict:
    """TweetEval offensive: 14,100 tweets, offensive or not (moderation-style traffic)."""
    return {"name": "tweet_offensive", "rows": _tweeteval("offensive"),
            "instructions": "Is this tweet offensive? Offensive means insults, threats, profanity aimed at "
                            "someone, or hateful content.",
            "classes": {"offensive": "Offensive", "not-offensive": "Not offensive"}}


def synthetic(n: int = 20000, seed: int = 0) -> dict:
    from jevstiller import SyntheticWorld
    labels = ["billing", "technical", "cancellation", "sales", "other"]
    world = SyntheticWorld(labels, seed=seed)
    return {"name": "synthetic", "rows": world.sample(n), "instructions": "Which team handles this?",
            "classes": {c: "" for c in labels}, "world": world}


LOADERS = {"banking77": banking77, "clinc150": clinc150, "ag_news": ag_news, "tweet_sentiment": tweet_sentiment,
           "tweet_offensive": tweet_offensive, "synthetic": synthetic}
