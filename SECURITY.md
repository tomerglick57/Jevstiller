# Security

The threat model, what Jevstiller stores, how API keys are handled, how to restrict access to the proxy, and the results of the security audits (2026-09-24, and two on 2026-09-25) are in [docs/security.md](docs/security.md).

In short:
- Jevstiller stores your traffic (request text, embeddings, Jev's answers) under `data_dir`. Treat that directory like a database of your traffic. Use `store_text = false` or `text_retention_days` if text must not be kept.
- API keys are never stored or logged; callers keep their own keys, and the proxy holds only salted hashes in memory.
- By default anyone who can reach the proxy can use it with a Jev key that Jev accepts. Restrict it with `allow_networks`, an `access_token`, TLS, or a private network.

## Reporting a vulnerability

Email the maintainer (address on the GitHub profile) rather than opening a public issue. You'll get a reply within a week.
