"""Replay your own texts through Jevstiller with Jev as the teacher.

    TYPESAFE_API_KEY=... python examples/replay_csv.py messages.csv --column text

The CSV needs one text column. Answers are cached in ./jev-cache.jsonl so re-runs are free.
"""
import argparse
import csv

from jevstiller import Config, Jevstiller, Task, load_encoder
from jevstiller.teachers import CachedTeacher
from jevstiller.teachers.jev import JevTeacher

ap = argparse.ArgumentParser()
ap.add_argument("csv")
ap.add_argument("--column", default="text")
ap.add_argument("--encoder", default="small", help="small | base | large  (base needs a GPU to be fast)")
ap.add_argument("--target", type=float, default=0.98)
a = ap.parse_args()

task = Task(
    name="my_task",
    instructions="What is this message about?",
    classes={
        "billing": "Charges, invoices, refunds, payment methods",
        "technical": "Bugs, errors, things not working",
        "cancellation": "Wants to cancel, downgrade, or close the account",
        "sales": "Pre-sales questions, plans, quotes, upgrades",
        "other": "Anything that does not fit the categories above",
    },
    target_agreement=a.target,
)

with open(a.csv, newline="") as f:
    texts = [row[a.column] for row in csv.DictReader(f)]

teacher = CachedTeacher(JevTeacher(), "./jev-cache.jsonl")
js = Jevstiller(task, teacher, data_dir="./jevstiller-data", encoder=load_encoder(a.encoder), config=Config())

for i in range(0, len(texts), 100):
    js.classify_batch(texts[i:i + 100])
    if (i // 100) % 10 == 9:
        st = js.status()
        print(f"{i + 100:>7,}  local {st.student_share:5.1%}  production={st.production}")

print()
print(js.status().report())
