"""Yes/no (`noul`) questions over the benchmark datasets, for measuring how well a local head reproduces Jev's
probability. Each task: the dataset, the question as sent to Jev, and which dataset labels count as "yes"
(only for accuracy against the dataset; Jev never sees them)."""
from __future__ import annotations

NOUL_TASKS = {
    "offensive": {
        "dataset": "tweet_offensive", "limit": 0,
        "instructions": "Is this tweet offensive?",
        "criteria": {"true": "Insults, threats, profanity aimed at someone, or hateful content",
                     "false": "Not offensive"},
        "yes": {"offensive"},
    },
    "negative": {
        "dataset": "tweet_sentiment", "limit": 20_000,
        "instructions": "Is the overall sentiment of this tweet negative?",
        "criteria": None,
        "yes": {"negative"},
    },
    "sports": {
        "dataset": "ag_news", "limit": 0,
        "instructions": "Is this news article about sports?",
        "criteria": None,
        "yes": {"sports"},
    },
    "card_lost": {
        "dataset": "banking77", "limit": 0,
        "instructions": "Is the customer reporting a lost or stolen card?",
        "criteria": {"true": "The card is lost, stolen or missing and the customer wants help with that",
                     "false": "Anything else, including cards that are late, declined, blocked or swallowed by an ATM"},
        "yes": {"lost_or_stolen_card"},
    },
}


def rows_for(name: str) -> list[tuple[str, bool]]:
    """(text, dataset says yes) for a task, in a fixed order."""
    import random

    from datasets import LOADERS
    t = NOUL_TASKS[name]
    rows = list(LOADERS[t["dataset"]]()["rows"])
    random.Random(0).shuffle(rows)
    if t["limit"]:
        rows = rows[:t["limit"]]
    seen, out = set(), []
    for text, label in rows:
        if text not in seen:
            seen.add(text)
            out.append((text, label in t["yes"]))
    return out
