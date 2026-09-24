# Security

## What Jevstiller stores

Every request it sees is stored in a local SQLite file per task, under `data_dir` (`<data_dir>/tasks/<key>/samples.sqlite` for the proxy). A row holds the text (or canonical JSON) of the state, its embedding, the teacher's answer and the student's. Treat that directory like a database of your traffic: restrict access and back it up deliberately. `Config(store_text=False)` keeps only a hash and the embedding. Retention controls and deletion are in progress (DEPLOYMENT_PLAN P4.6).

## API keys

- **Library:** the Jev key is read from `TYPESAFE_API_KEY` (or a `.env` you keep out of version control). It is never written to the store or to logs.
- **Proxy:** callers use their own keys. The proxy forwards each caller's `Authorization` header to Jev and keeps only a salted SHA-256 of the key, in memory. The salt is a per-deployment secret at `<data_dir>/key-salt`, created with mode 0600. Keys never reach disk or logs.
- The proxy answers locally only for a key Jev has accepted within the last `key_ttl_s` (1 h). A 401/403 from Jev revokes that at once. So the proxy can't be used to get answers with a key Jev would reject.
- With `tenancy="shared"` (the default), every key's traffic trains shared tasks, which is intended when all callers belong to one organisation. Use `per_key` when callers must not share data or models.

## Network

The proxy has no authentication of its own yet (DEPLOYMENT_PLAN P4.4): anyone who can reach it can use it with a key Jev accepts. Run it on a private network, or behind a reverse proxy that enforces access and TLS (P4.7).

## Reporting a vulnerability

Email the maintainer (address on the GitHub profile) rather than opening a public issue. You'll get a reply within a week.
