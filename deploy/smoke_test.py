"""Check a Jevstiller container image end to end: python deploy/smoke_test.py [IMAGE]  (default jevstiller:latest)

Runs the image on a private Docker network next to a mock Jev (this script, run in the same image), and checks:
- /healthz and /readyz, as a non-root user, with a read-only root filesystem and every capability dropped;
- a /v1/systemone request is forwarded with the caller's key and Jev's answer comes back unchanged; a key Jev
  rejects gets Jev's 401;
- the task is admitted and recorded (admin API), and /metrics needs the admin token;
- with no network at all, the baked-in encoder still loads (the image needs no network except to Jev);
- `docker stop` shuts it down cleanly, and the log has no traceback.

Standard library only; needs the docker CLI. CI runs it on every image build (.github/workflows/docker.yml).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

KEY = "smoke-test-key"
ADMIN = "smoke-test-admin-token-0123456789"
LABELS = ["billing", "technical", "other"]
REQUEST = {"state": "My invoice is wrong", "model": "jev-latest",
           "questions": {"team": {"type": "choice", "instructions": "Which team handles this?",
                                  "criteria": {c: "" for c in LABELS}}}}


class MockJev(BaseHTTPRequestHandler):
    def do_POST(self):                                        # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))) or b"{}")
        if self.headers.get("authorization") != f"Bearer {KEY}":
            return self._send(401, {"detail": "invalid API key"})
        answers = {name: {"type": "choice", "choice": "billing", "confidence": 0.9,
                          "probabilities": {"billing": 0.9, "technical": 0.05, "other": 0.05}}
                   for name in body.get("questions", {})}
        self._send(200, {"model": "jev-smoke-1", "answers": answers, "usage": {"input_tokens": 12, "output_tokens": 1}})

    def _send(self, status: int, data: dict) -> None:
        raw = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("x-typesafe-request-id", "req_smoke")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args):
        pass


def docker(*args: str, check: bool = True) -> str:
    r = subprocess.run(["docker", *args], capture_output=True, text=True)
    if check and r.returncode:
        raise SystemExit(f"docker {' '.join(args)} failed:\n{r.stdout}{r.stderr}")
    return r.stdout.strip()


def http(method: str, url: str, body: dict | None = None, headers: dict | None = None):
    req = urllib.request.Request(url, method=method, headers={"content-type": "application/json", **(headers or {})},
                                 data=json.dumps(body).encode() if body is not None else None)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def wait_ready(url: str, container: str, seconds: float = 180) -> dict:
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            status, _, body = http("GET", url)
            if status == 200:
                return json.loads(body)
        except OSError:
            pass
        if docker("inspect", "-f", "{{.State.Running}}", container) != "true":
            raise SystemExit(f"container exited:\n{docker('logs', container, check=False)}")
        time.sleep(1)
    raise SystemExit(f"not ready after {seconds:.0f} s:\n{docker('logs', container, check=False)}")


def check(ok: bool, what: str) -> None:
    print(("ok    " if ok else "FAIL  ") + what)
    if not ok:
        raise SystemExit(1)


HARDENED = ["--read-only", "--tmpfs", "/tmp", "--cap-drop", "ALL", "--security-opt", "no-new-privileges"]


def main(image: str) -> None:
    name = f"jevstiller-smoke-{os.getpid()}"
    net, mock, offline = f"{name}-net", f"{name}-jev", f"{name}-offline"
    docker("network", "create", net)
    try:
        docker("run", "-d", "--name", mock, "--network", net, "--entrypoint", "python",
               "-v", f"{os.path.abspath(__file__)}:/smoke_test.py:ro", image, "/smoke_test.py", "--mock-jev", "8000")
        docker("run", "-d", "--name", name, "--network", net, "-p", "127.0.0.1::8080", *HARDENED,
               "-e", f"JEVSTILLER_UPSTREAM=http://{mock}:8000", "-e", f"JEVSTILLER_ADMIN_TOKEN={ADMIN}",
               "-e", "JEVSTILLER_ADMIT_AFTER=1", image)
        base = "http://" + docker("port", name, "8080/tcp").splitlines()[0].replace("0.0.0.0", "127.0.0.1")
        run_checks(image, name, base, offline)
    finally:
        for c in (name, mock, offline):
            docker("rm", "-f", c, check=False)
        docker("network", "rm", net, check=False)
    print("smoke test passed")


def run_checks(image: str, name: str, base: str, offline: str) -> None:
    ready = wait_ready(f"{base}/readyz", name)
    check(ready.get("ready") is True and all(ready["checks"].values()), f"/readyz: {ready['checks']}")
    check(http("GET", f"{base}/healthz")[0] == 200, "/healthz")
    check(docker("exec", name, "id", "-u") == "10001", "runs as uid 10001")

    status, headers, body = http("POST", f"{base}/v1/systemone", REQUEST, {"authorization": f"Bearer {KEY}"})
    answer = json.loads(body)
    check(status == 200 and headers.get("x-jevstiller-source") == "upstream"
          and answer["answers"]["team"]["choice"] == "billing" and answer["model"] == "jev-smoke-1",
          "a systemone request is forwarded and Jev's answer relayed")
    status, _, _ = http("POST", f"{base}/v1/systemone", REQUEST, {"authorization": "Bearer wrong-key"})
    check(status == 401, "a key Jev rejects gets Jev's 401")

    auth = {"authorization": f"Bearer {ADMIN}"}
    status, _, body = http("GET", f"{base}/jevstiller/v1/tasks", headers=auth)
    check(status == 200 and len(json.loads(body)) == 1, "the question became a task (admin API)")
    check(http("GET", f"{base}/metrics")[0] == 401, "/metrics needs the admin token")
    status, _, body = http("GET", f"{base}/metrics", headers=auth)
    check(status == 200 and b"jevstiller_requests_total" in body, "/metrics with the admin token")

    # no network at all: the baked-in encoder must still load
    docker("run", "-d", "--name", offline, "--network", "none", *HARDENED, image)
    probe = ("import json, time, urllib.request\n"
             "for _ in range(180):\n"
             "    try:\n"
             "        r = json.load(urllib.request.urlopen('http://127.0.0.1:8080/readyz', timeout=2))\n"
             "        print(json.dumps(r)); break\n"
             "    except Exception:\n"
             "        time.sleep(1)\n")
    out = docker("exec", offline, "python", "-c", probe, check=False)
    check('"ready": true' in out, f"ready with no network (encoder baked in): {out or 'no answer'}")

    docker("stop", "-t", "30", name)
    check(docker("inspect", "-f", "{{.State.ExitCode}}", name) == "0", "docker stop: clean exit")
    logs = subprocess.run(["docker", "logs", name], capture_output=True, text=True)
    check("Traceback" not in logs.stdout + logs.stderr, "no traceback in the log")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--mock-jev"]:                        # inside the mock's container
        ThreadingHTTPServer(("0.0.0.0", int(sys.argv[2])), MockJev).serve_forever()
    else:
        main(sys.argv[1] if len(sys.argv) > 1 else "jevstiller:latest")
