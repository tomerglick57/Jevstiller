"""Requests with images (Jev 1.14, experimental): read one, check it against Jev's rules, and say whether it can go
to an image task.

Scope (v1): exactly one image, as the whole state or as the whole value of one top-level field of a JSON-object state,
and referenced nowhere else in the request. Anything else, or anything Jev would refuse, comes back as a reason
instead, and the proxy forwards the request as it is: a mistake here can only send a request to Jev, never answer
one. Jev's rules are docs.typesafe.ai/go/images (2026-10-14).

The parser works on the body already in memory: nothing is written to disk, and an image is never decoded here
(`check_image` reads its header only).
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from email.message import Message
from typing import Any

from .encoders._image import check_image

FIELDS = {"state", "model", "questions", "assets"}
ASSET_FIELDS = {"type", "part", "data_base64", "format"}
FORMATS = {"png", "jpg", "jpeg"}
PART_TYPES = {"", "image/png", "image/jpeg", "image/jpg", "application/octet-stream"}
MAX_REF_BYTES = 1024
MAX_PARTS = 8
MAX_PART_HEADER = 8192
MAX_JSON_ITEMS = 50_000              # commas, brackets and braces in the request's JSON: parsing more cost up to
                                     # 800 MiB per 32 MiB body (security audit run 5); a real request has a few dozen
_B64 = re.compile(r"[A-Za-z0-9+/]*={0,2}")
_END = {"PNG": b"\x00\x00\x00\x00IEND\xaeB`\x82", "JPEG": b"\xff\xd9"}   # how each format's file ends

# reasons a request is forwarded instead (the `reason` of jevstiller_unrouted_total)
UNREADABLE = "image_unreadable"     # not a well-formed image request
INVALID = "image_invalid"           # breaks a rule of Jev's: Jev would refuse it too
SCOPE = "image_scope"               # valid, but more than v1 answers locally (several images, images in the question)


@dataclass(frozen=True)
class ImageRequest:
    req: dict           # state, model, questions: the image's reference replaced by its content ("<image:sha256:…>")
    image: bytes        # the image file, as the caller sent it
    digest: str         # SHA-256 of the image file
    state_text: str     # what is stored for the row: the canonical state, with the content reference


@dataclass(frozen=True)
class _Part:
    name: str
    filename: str | None
    content_type: str
    data: bytes


def _boundary(content_type: str) -> str | None:
    """The boundary of a multipart/form-data content type, or None unless there is exactly one, read as standard
    parsers read parameters. The first `boundary=` anywhere used to be taken, even inside another parameter's
    quoted value, so the proxy and Jev could frame one body differently and the proxy record Jev's answer about
    one image against another (security audit run 5)."""
    if content_type.lower().count("boundary=") != 1:
        return None
    msg = Message()
    msg["content-type"] = content_type
    if msg.get_content_type() != "multipart/form-data":
        return None
    found = [v for k, v in msg.get_params(header="content-type")[1:] if k.strip().lower() == "boundary"]
    if len(found) != 1 or not isinstance(found[0], str) or not 1 <= len(found[0]) <= 70:
        return None
    return found[0]


def _multipart(content_type: str, body: bytes) -> list[_Part] | None:
    """The parts of a multipart/form-data body, or None if it isn't one this strict reader accepts."""
    boundary = _boundary(content_type)
    if boundary is None:
        return None
    delim = b"--" + boundary.encode("latin-1")
    if not body.startswith(delim):
        return None                                    # no preamble: the SDK never sends one
    pieces = body[len(delim):].split(b"\r\n" + delim)
    if len(pieces) < 2 or not pieces[-1].startswith(b"--") or len(pieces) - 1 > MAX_PARTS:
        return None
    parts, names = [], set()
    for raw in pieces[:-1]:
        if not raw.startswith(b"\r\n"):
            return None
        head, sep, data = raw[2:].partition(b"\r\n\r\n")
        if not sep or len(head) > MAX_PART_HEADER:
            return None
        msg = Message()
        for line in head.decode("latin-1").split("\r\n"):
            k, colon, v = line.partition(":")
            if not colon:
                return None
            msg[k.strip()] = v.strip()
        if msg.get_content_disposition() != "form-data":
            return None
        name = msg.get_param("name", header="content-disposition")
        filename = msg.get_param("filename", header="content-disposition")
        if not isinstance(name, str) or not name or name in names or isinstance(filename, tuple):
            return None
        names.add(name)
        ctype = (msg.get("content-type") or "").split(";")[0].strip().lower()
        parts.append(_Part(name, filename, ctype, data))
    return parts


def _strings(value: Any) -> Iterator[str]:
    """Every string value in a JSON value (object keys aren't: Jev doesn't replace references in keys). Iterative:
    a state nested past the recursion limit made the recursive version raise, and the request fail (run 5)."""
    stack = [value]
    while stack:
        v = stack.pop()
        if isinstance(v, str):
            yield v
        elif isinstance(v, dict):
            stack.extend(v.values())
        elif isinstance(v, list):
            stack.extend(v)


def _b64(s: Any) -> bytes | None:
    """Jev's base64: standard alphabet, `=` padding, no whitespace, no `data:` prefix."""
    if not isinstance(s, str) or not s or len(s) % 4 or not _B64.fullmatch(s):
        return None
    try:
        return base64.b64decode(s, validate=True)
    except (binascii.Error, ValueError):
        return None


def read(content_type: str, body: bytes, images_header: bool) -> ImageRequest | str:
    """The request as an image task can take it, or the reason it goes to Jev as it is."""
    if not images_header:
        return INVALID                                 # Jev refuses images without its opt-in header
    multipart = content_type.lower().startswith("multipart/")
    files: dict[str, _Part] = {}
    if multipart:
        parts = _multipart(content_type, body)
        if not parts or parts[0].name != "request" or parts[0].content_type.split(";")[0] != "application/json":
            return UNREADABLE
        for p in parts[1:]:
            if p.name == "request" or p.content_type not in PART_TYPES:
                return INVALID
            files[p.name] = p
        raw = parts[0].data
    else:
        raw = body
    if raw.count(b",") + raw.count(b"[") + raw.count(b"{") > MAX_JSON_ITEMS:
        return UNREADABLE                              # before parsing it: see MAX_JSON_ITEMS
    try:
        req = json.loads(raw)
    except (ValueError, RecursionError):
        return UNREADABLE
    if not (isinstance(req, dict) and set(req) <= FIELDS and isinstance(req.get("assets"), dict)
            and isinstance(req.get("questions"), dict) and req["questions"] and isinstance(req.get("model"), str)
            and "state" in req):
        return UNREADABLE
    assets = req["assets"]
    if not assets:
        return INVALID                                 # an empty `assets` is refused
    for ref, asset in assets.items():
        if (not ref.strip() or len(ref.encode("utf-8")) > MAX_REF_BYTES or not isinstance(asset, dict)
                or not set(asset) <= ASSET_FIELDS or asset.get("type", "image") != "image"
                or any(not isinstance(asset[k], str) for k in ("part", "format") if k in asset)):
            return INVALID
        if "part" in asset and "data_base64" in asset:
            return INVALID                             # two sources: which one Jev reads is unknown (run 5)
    claimed = [a["part"] for a in assets.values() if "part" in a]
    if not multipart and claimed or sorted(claimed) != sorted(files) or len(set(claimed)) != len(claimed):
        return INVALID                                 # each file part belongs to exactly one asset, and only parts
    if len(assets) != 1:
        return SCOPE
    (ref, asset), = assets.items()
    if "part" in asset:
        data = files[asset["part"]].data
        if "format" in asset and asset["format"] not in FORMATS:
            return INVALID
    else:
        if asset.get("format") not in FORMATS:
            return INVALID
        data = _b64(asset.get("data_base64"))
        if data is None:
            return INVALID
    check = check_image(data)
    if not check.ok:
        return INVALID
    if not data.endswith(_END[check.format]):
        return SCOPE                                   # bytes after the image: another reader may see more (run 5)
    state, questions = req["state"], req["questions"]
    if not any(ref in s for s in _strings(state)) and not any(ref in s for s in _strings(questions)):
        return INVALID                                 # every asset must be referenced somewhere
    if state == ref:
        slot = None                                    # the image is the whole state
    elif isinstance(state, dict) and sum(v == ref for v in state.values()) == 1:
        slot = next(k for k, v in state.items() if v == ref)
    else:
        return SCOPE
    rest = {k: v for k, v in state.items() if k != slot} if slot is not None else None
    if any(ref in s for s in _strings(questions)) or any(ref in s for s in _strings(rest)):
        return SCOPE                                   # elsewhere too: another copy of the image Jev would see
    digest = hashlib.sha256(data).hexdigest()
    content = f"<image:sha256:{digest[:32]}>"
    canonical = content if slot is None else {**state, slot: content}
    from ._task import state_text
    try:
        text = state_text(canonical)
    except (ValueError, RecursionError):                # e.g. a state nested past what json.dumps takes
        return UNREADABLE
    out = {"state": canonical, "model": req["model"], "questions": questions}
    return ImageRequest(out, data, digest, text)
