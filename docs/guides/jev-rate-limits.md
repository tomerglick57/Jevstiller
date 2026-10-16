# Jev rate limits and 429s

TypeSafe publishes Jev's rate limits on its models page: 80 requests and 100,000 tokens a second as of October 2026. At launch they were 1,200 requests a minute. TypeSafe changes them at its own discretion. Over a limit, Jev answers `429` with a `retry-after` header, and the TypeSafe SDK retries 429s with back-off.

## What helps

- **Honour `retry-after`** and spread bursts. The SDK does the first part. Many callers retrying at once is what turns a short limit into an outage.
- **Ask several questions per request.** One state with several questions is one request.
- **Send fewer requests.** Requests answered locally never reach Jev and never count against its limits. With [Jevstiller](https://github.com/tomerglick57/Jevstiller) in front, that's typically 70–80% of a repeated intent or topic question once its local model has trained ([benchmarks](../benchmarks.md)). The remaining requests and a 2% audit still go to Jev.

## How Jevstiller handles a 429

When Jev answers a key with `429`, the proxy:
- answers that key's forwarded requests itself with `429` and the same `retry-after`, until it passes (at most an hour), without calling Jev;
- keeps answering that key's requests locally where it can;
- leaves other keys unaffected.

In the chaos test, 100 requests sent 8 at a time into a 429 on one key reached Jev at most 8 times, and another key was unaffected ([benchmarks](../benchmarks.md)).

Not yet: limiting a key before Jev's own 429 ([the proxy docs](../proxy.md#not-yet)).

## Before you deploy

Jevstiller trains on Jev's output, and TypeSafe's terms restrict that. Read [TypeSafe's terms](../../README.md#typesafes-terms) and decide for your own use.
