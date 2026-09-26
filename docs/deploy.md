# Deploying Jevstiller

One process, one data directory, anywhere that can reach Jev. The calling services change one environment variable:

```bash
export TYPESAFE_BASE_URL=http://jevstiller:8080
```

## Options

### Docker

```bash
docker run -d --name jevstiller -p 8080:8080 -v jevstiller-data:/data ghcr.io/tomerglick57/jevstiller:0.3.3
curl -s localhost:8080/readyz
```

About the image:
- Published for every release for linux/amd64 and linux/arm64: `ghcr.io/tomerglick57/jevstiller:<version>` (also `:<major>.<minor>` and `:latest`), with build provenance and an SBOM attached.
- CPU, ONNX Runtime, bge-small baked in. It runs as uid 10001, with `/data` as the volume, and needs no network except to Jev: `HF_HUB_OFFLINE` is set when the encoder is baked in.
- It works with a read-only root filesystem when `/tmp` is writable, and with all capabilities dropped. `deploy/smoke_test.py` checks this, and more, against any image: `python deploy/smoke_test.py ghcr.io/tomerglick57/jevstiller:0.3.3`. CI runs it on every image build.

To build it yourself: `docker build -t jevstiller .`. Every package in the build is hash-checked: the dependencies from `uv.lock`, and the build tools (the build backend, `build`, `uv`) from `build-requirements.txt`. The project is built without build isolation, so nothing unpinned is fetched. Build arguments: `PRELOAD_ENCODER=base` (or empty, which downloads on first start), `EXTRAS=server,onnx`, `PYTHON=3.12`.

**GPU:**
1. Build on a CUDA base image with `--build-arg EXTRAS=gpu` (adds PyTorch), e.g. start from `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04` and install Python 3.12.
2. Run with `--gpus all`. With CUDA available, the encoder runs on PyTorch automatically.

The encoder is the only GPU user. On CPU, bge-small encodes about 1.6–7 ms per text. Measured student-path throughput on CPU is about 130 messages/s, 6.5× Jev's rate limit.

### docker compose

`docker-compose.yml` runs the image with `deploy/jevstiller.toml` and an admin token from a file:

```bash
openssl rand -hex 32 > deploy/admin-token.txt     # gitignored
docker compose up -d
```

### Kubernetes

`deploy/kubernetes/jevstiller.yaml` contains a ConfigMap (the settings), a PVC (data), a Deployment and a Service:
- The Deployment has **1 replica**, the **Recreate** strategy, non-root, a read-only root filesystem, and readiness on `/readyz` and liveness on `/healthz`.
- The admin token comes from the `jevstiller` Secret:

```bash
kubectl create secret generic jevstiller --from-literal=admin-token=$(openssl rand -hex 32)
kubectl apply -f deploy/kubernetes/jevstiller.yaml
```

More than one replica is not supported. Each task's state is a local SQLite file, and two processes must never share a data directory (DEPLOYMENT_PLAN P2.3). Scale up with CPU/GPU on one pod, or run separate deployments for separate groups of services.

### pip

```bash
pip install jevstiller              # on a GPU machine: pip install "jevstiller[gpu]"
jevstiller serve --config jevstiller.toml
```

## Configure

Start from `deploy/jevstiller.toml`. Every key is in [configuration.md](configuration.md). The decisions worth making on day one:

| Decision | Setting | Default |
|---|---|---|
| Do all callers share trained tasks? | `tenancy = "shared" \| "per_key"`, or a tenants map | shared |
| Who may use the proxy? | `allow_networks`, `access_token_file` | anyone who can reach it |
| Keep request text? | `store_text`, `text_retention_days` | kept |
| Admin API and metrics | `admin_token_file` | off |
| How close to Jev? | `target_agreement` (per task via the admin API) | 0.98 |
| How much CPU for training? | `train_workers` (× 2 threads each) | 2 |

Secrets (`admin_token`, `access_token`) belong in files or environment variables, not flags (flags show in the process list). `jevstiller config` prints the effective settings with secrets redacted.

### TLS and reverse proxies

- **Built in:** `ssl_certfile` + `ssl_keyfile`. Callers then use `https://`, and need to trust the certificate.
- **Behind a reverse proxy or load balancer:** terminate TLS there. If you use `allow_networks`, list the proxy's address in `trust_forwarded_for` and make sure it **overwrites or appends** `X-Forwarded-For` rather than passing the client's value through. Without `trust_forwarded_for`, the TCP peer is used and `X-Forwarded-For` is ignored.
- **Timeouts:** keep the proxy's upstream timeout (9 s) under the SDK's 10 s client timeout. A load balancer in front should allow at least 10 s.

### Sizing

Measured on a 16-vCPU VM (docs/benchmarks.md):

| Resource | Guide |
|---|---|
| CPU | Serving is about 2–5 ms of CPU per locally answered request with a real encoder, plus training bursts capped at `train_workers × 2` cores (low priority). 2–4 cores cover hundreds of requests per second. |
| Memory | ~70 MB base, the encoder (~150 MB for bge-small ONNX), and ~2–8 MB per loaded task (`max_loaded`, default 64). Plan 2 GB, and cap it with `max_memory_mb`. Set `max_loaded` above the number of tasks that are active at once: below it, tasks reload from disk many times a second. That is safe (memory stays flat: a 20-minute soak with 20 active tasks and `max_loaded = 12` held ~160 MB) but costs CPU and disk reads, and slows each task's learning. |
| Disk | Per task: ~3–4 KB per stored request (text, embedding, answers and indexes; 3.9 KB measured with a 512-dim encoder) and ~8 MB per kept model version (production, shadow and at most `keep_versions`, 3, of each finished state). A task seeing 10k requests/day grows ~35 MB/day. Stored requests are not deleted yet; `text_retention_days` blanks old text, and `idle_ttl_days` deletes tasks nobody uses. |
| Network | Only to the upstream (`api.typesafe.ai:443`) and from the callers. |

## Upgrade

1. `jevstiller backup --data-dir /data --out /backups/$(date +%F)`. This is safe while serving.
2. Deploy the new version: stop the old one, then start the new one (never both on one data directory).
3. Stores migrate themselves on open (schema version). Tasks re-key themselves if the key scheme changed. Both are logged.
4. Check `/readyz`, `jevstiller admin stats`, and the `jevstiller_requests_total` metric.

To roll back, restore the backup into an empty data directory with `jevstiller restore`, and run the old version.
