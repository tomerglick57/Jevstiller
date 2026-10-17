# Jev rate limits and 429s
<!-- title: Jev rate limits and 429 errors: what the limits are, how the SDK retries, and how to send fewer requests -->
<!-- description: Jev's rate limits are set per organization in requests and tokens per second and shown in TypeSafe's developer console. What a 429 and a 529 mean, the TypeSafe SDK's retry defaults, what helps, and how a local proxy keeps 70–80% of a repeated question's requests from reaching Jev at all. -->

## What the limits are

From TypeSafe's [models page](https://docs.typesafe.ai/models) and [API reference](https://docs.typesafe.ai/api), checked on 2026-10-17:

- **Limits are per organization**, measured in requests per second and tokens per second, and shown in the [developer console](https://console.typesafe.ai/rate-limits). TypeSafe no longer publishes figures: at launch in September 2026 the published limit was 1,200 requests a minute, and in early October the models page said 80 requests and 100,000 tokens a second. Higher limits are on custom and enterprise plans.
- **Over either limit, Jev answers `429 Too Many Requests`**: "You have exceeded your rate limit. Back off and retry after a short delay." `529 Overloaded` means TypeSafe itself is busy; the advice is the same.
- **The TypeSafe Python SDK retries by default**: 408, 429 and 5xx responses, 2 retries after the first attempt, waiting 0.5 s and doubling up to 5 s with 25% jitter, honouring `Retry-After` and `retry-after-ms`, within a 30 s budget per call. All of it is configurable with `RetryPolicy` ([SDK retries](https://docs.typesafe.ai/sdk/python/api/retries)).
- **Other limits** are not rate limits: a request body of at most 32 MiB (`413`), 64k tokens per request and 32k per question (`400 max_tokens_exceeded`), up to 255 options per `choice` and 10 levels per `score`.

## What helps

- **Honour `retry-after` and spread bursts.** The SDK does the first part. Many callers retrying at once is what turns a short limit into an outage; cap concurrency per key.
- **Ask several questions per request.** One state with several questions is one request against the request limit, and the state's tokens count once.
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

## FAQ

### What does a Jev 429 mean?

Your organization sent more requests per second or more tokens per second than its limit allows. TypeSafe's advice is to back off and retry after a short delay. The official SDKs do this by default and follow the `Retry-After` header.

### What are Jev's rate limits?

They are set per organization and shown in TypeSafe's developer console, in requests per second and tokens per second. As of October 2026 TypeSafe does not publish the figures; at launch the published limit was 1,200 requests a minute. Custom and enterprise plans get higher limits.

### Does the TypeSafe SDK retry 429s automatically?

Yes. By default the Python SDK retries 408, 429 and 5xx responses twice, waiting 0.5 s and doubling up to 5 s with jitter, and honours `Retry-After`. The whole call has a 30 s budget. Pass a `RetryPolicy` to change any of it.

### What is a 529 from Jev?

`529 Overloaded`: TypeSafe is temporarily overloaded. Retry with backoff, as for a 429.

### How does a local proxy reduce 429s?

A request answered locally is never sent to Jev, so it cannot count against the limit. Jevstiller answers 70–80% of a repeated intent or topic question locally once trained, within an agreement bound you set; the rest and a 2% audit still go to Jev.

### What happens to requests Jevstiller forwards while Jev is rate limiting?

The proxy returns Jev's `429` and `retry-after` to the caller for that key, without calling Jev again until the wait passes, and keeps answering what it can locally. Other keys are not affected.
