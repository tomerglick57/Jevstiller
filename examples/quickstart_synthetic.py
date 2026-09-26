"""The whole loop in under a minute, on CPU, with no API key.

A synthetic "teacher" stands in for Jev. Watch the student take over traffic while the audit channel
keeps checking agreement against the contract.
"""
import tempfile

from jevstiller import Config, Jevstiller, SyntheticTeacher, SyntheticWorld, Task

labels = ["billing", "technical", "cancellation", "sales", "other"]
world = SyntheticWorld(labels, seed=1)
teacher = SyntheticTeacher(world)

task = Task(
    name="support_router",
    instructions="Which team should handle this message?",
    classes={c: "" for c in labels},
    target_agreement=0.95,
)
# training="inline": this replay pushes weeks of traffic through in seconds, faster than a background
# trainer could keep up with. A live service keeps the default ("background").
cfg = Config(audit_rate=0.05, min_train_samples=400, min_samples_per_class=20, min_calib_samples=150,
             min_new_samples=1500, shadow_min_samples=200, training="inline")

# A fresh directory each run, so the student always starts from nothing. A real service keeps a fixed data_dir.
js = Jevstiller(task, teacher, data_dir=tempfile.mkdtemp(prefix="jevstiller-quickstart-"), config=cfg)

for step in range(40):
    batch = [text for text, _ in world.sample(200)]
    results = js.classify_batch(batch)
    local = sum(r.source != "teacher" for r in results) / len(results)
    if step % 5 == 4:
        prod = js.status().production
        print(f"after {(step + 1) * 200:>6,} requests: {local:5.1%} served locally   production={prod}")

print()
print(js.status().report(events=False))   # events=True (the default) also lists the last loop events
