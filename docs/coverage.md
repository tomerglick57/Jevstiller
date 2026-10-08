# Coverage of Jev's API

What Jevstiller does with each thing Jev offers: what it answers itself, what it passes to Jev, and what this design cannot answer. As of 0.5, checked against TypeSafe's documentation on 2026-10-05.

Two questions have very different answers.

## Passing requests through: complete

Everything Jev accepts, the proxy accepts. A request it does not answer itself goes to Jev unchanged, with the caller's key, and Jev's response comes back unchanged: other question types, unknown fields, other paths, every status code. The proxy never fails a request that Jev would have answered ([compatibility](compatibility.md)).

- **Tested:** the Python SDK (`typesafe-sdk` 0.7.1), sync and async, in CI over real HTTP.
- **Not tested:** the JavaScript SDK. It speaks the same wire format, so it should work, but nothing checks it.

## Answering locally: choice and yes/no questions, fixed questions, short text

| Jev offers | Answered locally | Notes |
|---|---|---|
| `choice`, the same question asked repeatedly, short English text | **Yes**, under the agreement guarantee | The case the project is built for. Up to 255 options, as in Jev. |
| `noul` (yes/no, returns a probability) | **Yes** (since 0.5), for the outcome your code makes of the number | You state the cut-off your code uses (`noul_cutoffs`, default 0.5), or two for an unsure band; the guarantee covers landing on Jev's side of it. The probability value itself is not covered; a tolerance on it is planned ([roadmap](https://jevstiller.pages.dev/project/roadmap/)). |
| `score` (2–10 ordered levels) | No, forwarded | Planned. |
| Several questions in one request (fan-out) | All or nothing | If one question can't be answered locally, the whole request goes to Jev and the other local answers are discarded. Forwarding only the unanswered questions is planned. A request with a `score` question is therefore always forwarded today. |
| Long state: conversations, documents (Jev reads up to 32K tokens) | Only the first 256 tokens reach the student | The default encoders truncate there (about 1–2k characters of prose; the state itself is kept up to 32,768 characters). When the answer depends on later text, the student disagrees with Jev, calibration sees it, and little is answered locally. Safe, but low coverage. |
| Structured state: JSON objects and arrays | Yes, as text | The state is serialised to canonical JSON and embedded like prose. A sentence encoder reads JSON less well than sentences; not benchmarked. |
| Text in other languages | With another encoder | The default encoders (bge) are English. Any Hugging Face embedding model can be selected (`--encoder onnx:<repo>` or `torch:<model>`); no multilingual default and no benchmark yet. |
| `probabilities` and `confidence` in the answer | The student's own numbers | The guarantee covers the label, and with `confidence_floor` one cut-off on the confidence ([configuration](configuration.md)). It does not cover the probability values themselves. |
| Structured `instructions` and `criteria` (objects, arrays) | Yes | A question is identified by its exact content, whatever its shape. |
| Model aliases (`jev-latest`, `jev-preview`) | Yes | The requested model is part of the task; a change in what an alias resolves to starts a new teacher lineage. |
| Images, audio, video | No | Jev does not accept them either. |

## What this design cannot answer

The local model is one small head per fixed question. That is a boundary, not a missing feature:

- **Questions or options that change with every request.** In re-ranking, semantic search and entity matching the candidates are the options, so each call asks a new question. There is nothing repeated to learn from; such questions never become tasks (a question must be asked `admit_after` times, 50 by default) and are forwarded.
- **One-off and exploratory questions.** The same.
- **A changing list of classes.** Adding or removing a class makes a new question, which starts from nothing.

Answering those locally would take a model that reads the question as well as the state: a small general decision model, not a distillation of one question's traffic. That is out of scope.

## Also not yet

Listed in [the proxy docs](proxy.md#not-yet): rate limiting before Jev's own 429, and queueing audit calls while Jev is unreachable.
