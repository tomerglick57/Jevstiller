# Experiments

Replays a labelled dataset through Jevstiller with the labels hidden (DESIGN.md §15). Outputs go to
`experiments/results/<run>/` (`results.json`, `report.md`, `curves.png`); Jev answers are cached in
`experiments/cache/<dataset>.jsonl` so a re-run is free.

```bash
bash experiments/reproduce.sh                                                     # the README's Banking77 result from Jev's recorded answers (shipped, gzipped); no key
bash experiments/bench.sh --no-record                                             # the five-task benchmark from recorded answers (release assets); no key, an hour or two
python3 experiments/run.py --dataset banking77 --teacher jev --encoder base        # the real thing (needs TYPESAFE_API_KEY in .env)
python3 experiments/run.py --dataset banking77 --teacher oracle --encoder base     # dry run: hidden labels as a perfect teacher
python3 experiments/run.py --dataset synthetic --teacher synthetic --encoder hash  # no downloads, no GPU, ~1 min
python3 experiments/jev_profile.py                                                 # live Jev latency + wire fixtures
python3 experiments/jev_profile.py --model jev-1.14.0 --fixtures-only               # another model's answers as extra fixtures (2 requests)
python3 experiments/run.py --dataset banking77 --teacher jev --jev-model jev-1.14.0 --encoder small --backend onnx --max-jev-calls 9000   # another Jev model, cached on its own
python3 experiments/record_answers.py --dataset banking77                          # record Jev's answer for every message into the cache (~$0.35)
python3 experiments/confidence_floor.py --dataset banking77                        # a caller's confidence check, with and without Task.confidence_floor (no key)
```

`bench.sh` runs, per task, `run.py --teacher cached` (the replay), `baselines.py` (four threshold rules on identical
splits: soft or hard labels × the bound or a point estimate), then `target_curve.py` (coverage at other targets) and
`bench.py --write`, which rebuilds the tables in docs/benchmarks.md and the README.

More options: `--backend onnx --device cpu` (CPU-only), `--ood-max-ref N` (size of the OOD reference, default 5,000),
`--limit`, `--target`, `--audit`, `--min-*`. Replays train `inline` so results don't depend on thread timing.
All results with their dates are collected in [docs/benchmarks.md](../docs/benchmarks.md).

Arms: `--teacher jev|cached|oracle|synthetic`, `--encoder small|base|large|hash`, `--label-target probs|hard`,
`--target 0.98`, `--rare-classes defer|wait`, `--backend torch|onnx`, `--device cpu|cuda`.

## Images (Jev 1.14): step 1, the labels as the teacher

Can a frozen image encoder plus the same loop answer image questions locally? Three public datasets, each one
repeated question, embedded with three encoders (`image_data.py`, `image_embed.py`; data and embeddings go to
`~/.cache/jevstiller/images/`, outside the repository), then replayed with the dataset's own labels as the teacher.
No Jev call; about 15 minutes on one GPU, most of it embedding.

```bash
python3 experiments/image_data.py fetch food101                                     # also rvl_cdip (4.6 GB), crack
python3 experiments/image_embed.py --dataset food101 --encoder clip-b32              # needs PyTorch; GPU recommended
python3 experiments/run.py --dataset img_food101 --teacher oracle --encoder clip-b32 --tag img
python3 experiments/image_bench.py                                                  # the table below
```

Target agreement 98%, 2,000 rows held out, `run.py`'s defaults otherwise (2026-10-17):

| Dataset | Encoder | Classes | Stream | Labels before the first student | Coverage (held out) | Agreement (held out) | Student alone | Labels used in all | CPU ms/image |
|---|---|---|---|---|---|---|---|---|---|
| Food-101 (dish in a photo) | clip-b32 | 101 | 23,250 | 1,835 | 60.5% | 99.1% | 85.8% | 12,426 | 59.0 |
| | siglip-b16 | 101 | 23,250 | 1,819 | 64.0% | 99.7% | 93.5% | 11,909 | 173.5 |
| | dinov2-s | 101 | 23,250 | 1,826 | 58.4% | 99.5% | 87.4% | 12,456 | 67.6 |
| RVL-CDIP (document type) | clip-b32 | 16 | 14,000 | 1,936 | 30.8% | 99.3% | 73.5% | 10,683 | 64.5 |
| | siglip-b16 | 16 | 14,000 | 1,963 | 46.9% | 98.4% | 79.8% | 10,219 | 206.0 |
| | dinov2-s | 16 | 14,000 | 1,948 | 18.5% | 99.2% | 68.5% | 11,888 | 70.1 |
| Concrete crack (yes/no) | clip-b32 | 2 | 22,000 | 1,330 | 99.9% | 99.9% | 99.9% | 2,100 | 75.8 |
| | siglip-b16 | 2 | 22,000 | 1,325 | 98.8% | 99.9% | 99.9% | 1,943 | 148.6 |
| | dinov2-s | 2 | 22,000 | 1,328 | 100.0% | 99.9% | 99.9% | 2,047 | 65.4 |

How to read it:
- This is not an upper bound on agreement with Jev. On Banking77 the small text encoder covered 56% against the
  labels and 72% against Jev's answers, which are easier to imitate. It says whether the encoder separates the classes.
- Photos of things (Food-101) land where text intents do: 58–64% local, against 56–71% for Banking77's labels.
  Pages whose type is in their text (RVL-CDIP) need the stronger encoder, and still reach only 31–47%. The crack
  question is easy.
- CPU ms/image is PyTorch at batch 1 with 4 threads, decoding and preprocessing included (transformers' Python
  preprocessing is most of DINOv2's and SigLIP's). Text costs ~10 ms. Batched with 8 threads: CLIP 26–37 ms.
- Coverage levels off well before the end of the stream: Food-101 with SigLIP reached 62% after 6,700 labels,
  RVL-CDIP 47% after 8,000, the crack question 98% after 1,300. That sizes step 2.

## Images: step 2, Jev's answers

The same replays against Jev (`jev-1.14.0`, 2026-10-17), with each question's best encoder from step 1 and 1,000
rows held out. `--teacher jev` asks Jev only about the held-out rows and the rows the loop sends to the teacher, not
the whole dataset; every answer is cached, and `--max-jev-calls` is required. Images are sent at Jev's minimum size
(shorter side 280 px), document pages at theirs (`image_jev.py`).

```bash
python3 experiments/image_jev.py --dataset crack --probe 3                          # 3 images: tokens per image, projected cost
python3 experiments/run.py --dataset img_crack --teacher jev --encoder clip-b32 --eval-size 1000 --limit 3000 --max-jev-calls 4000 --tag img
python3 experiments/run.py --dataset img_food101 --teacher jev --encoder siglip-b16 --eval-size 1000 --limit 10000 --max-jev-calls 11000 --tag img
python3 experiments/run.py --dataset img_rvl_cdip --teacher jev --encoder siglip-b16 --eval-size 1000 --limit 10000 --max-jev-calls 11000 --tag img
```

Without a key: the answers are a release asset. With the datasets and embeddings built as in step 1, the same
commands with `--teacher cached` (and no cap) reproduce the table exactly:

```bash
ANSWERS=https://github.com/tomerglick57/Jevstiller/releases/download/answers-images-2026-10-17
for f in img_crack-jev-1.14.0-s280 img_food101-jev-1.14.0-s280 img_rvl_cdip-jev-1.14.0-snative; do
    curl -fsSL "$ANSWERS/$f.jsonl.gz" | gunzip > "experiments/cache/$f.jsonl"; done
```

| Dataset | Encoder | Jev vs labels | Jev below 0.6 | Held out | Stream | Coverage (held out) | Agreement with Jev (held out) | Jev calls | Tokens | Cost |
|---|---|---|---|---|---|---|---|---|---|---|
| Concrete crack | clip-b32 | 96.4% | 9.5% | 1,000 | 3,000 | 91.4% | 99.4% | 2,491 | 1.3M | $0.06 |
| Food-101 | siglip-b16 | 88.6% | 10.0% | 1,000 | 10,000 | 62.5% | 99.0% | 6,443 | 8.8M | $0.37 |
| RVL-CDIP | siglip-b16 | 68.8% | 17.1% | 1,000 | 10,000 | 40.1% | 98.1% | 8,557 | 20.7M | $0.87 |

- The guarantee held on all three: at least 98% agreement with Jev on held-out rows (1,000 rows: about ±0.9 points).
- Jev's answers were not easier to imitate than the labels here, unlike Banking77's. At the same point of the
  stream, Food-101 came out the same (62%), the crack question and RVL-CDIP lower (91% against 98%, 40% against 47%).
  Where Jev is unsure (17% of RVL-CDIP pages below 0.6), less is answered locally, as on the noisy text tasks.
- Jev took 270–320 ms on average per image request (`teacher profile` in each run's report). Tokens per request:
  about 527 for a 280×280 photo with 2 options, 1,336 with Food-101's 101 options, 2,394 for a 755×1,000 page:
  roughly 2,100–2,500 tokens per megapixel of image.
- One run per question: no seeds, no other encoders against Jev (a replay with another encoder asks Jev about the
  rows that one sends to the teacher and the cache doesn't have).

## Dry-run results (oracle teacher, Banking77, 77 classes, target agreement 98%)

Upper bound on distillation with a perfect teacher; 11,083 streamed rows, 2,000 held out.

| encoder | held-out coverage | system agreement | student path rows/s (GPU) |
|---|---:|---:|---:|
| small | 56.2% | 99.50% | 1,868 |
| **base** | **70.9%** | 99.55% | 1,950 |
| large | 59.8% | 99.45% | 1,038 |

Jev's published limit is 20 rows/s (1,200/min); on 2026-09-26 our key sustained 190/s without errors (docs/benchmarks.md), so treat the limit as a policy, not a ceiling.

Things these runs found and fixed in the loop (all now in DESIGN.md): shadow candidates starved by the 2% audit
trickle; policies fitted exactly at the budget failing the re-test by construction; training drifting toward the
deferred (hard) slice; a fixed L2 that hid the encoder differences; exact importance weights collapsing the
effective sample size.

The live run against Jev (2026-09-24) is in [docs/benchmarks.md](../docs/benchmarks.md#banking77-against-live-jev) and DESIGN.md §15.5.
