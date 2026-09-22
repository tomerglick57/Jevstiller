# Security

Jevstiller stores every request it sees — text, embedding, and teacher answer — in a local SQLite file under `data_dir`. Treat that directory like a database of your traffic: restrict access, back it up deliberately, and set `Config(store_text=False)` if raw text must not be kept.

The Jev API key is read from `TYPESAFE_API_KEY` (or a `.env` you keep out of version control). It is never written to the store or to logs.

To report a vulnerability, email the maintainer (address on the GitHub profile) rather than opening a public issue. You'll get a reply within a week.
