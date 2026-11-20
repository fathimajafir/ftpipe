from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
import uuid


class ProviderError(RuntimeError):
    pass


def api_key(env_name, provider):
    value = os.environ.get(env_name, "").strip()
    if not value:
        raise ProviderError(
            "%s needs %s. Put it in .env (see .env.example) or export it in your shell." % (provider, env_name)
        )
    return value


def base_url(env_name, default):
    return os.environ.get(env_name, "").strip() or default


def _read(req, timeout):
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:1200]
        raise ProviderError("HTTP %s from %s: %s" % (exc.code, req.full_url, detail)) from exc
    except urllib.error.URLError as exc:
        raise ProviderError("cannot reach %s: %s" % (req.full_url, exc.reason)) from exc
    return json.loads(body) if body.strip() else {}


def http_json(url, payload=None, headers=None, method=None, timeout=180):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method or ("POST" if data else "GET"))
    req.add_header("Content-Type", "application/json")
    for key, value in (headers or {}).items():
        req.add_header(key, value)
    return _read(req, timeout)


def http_multipart(url, fields, files, headers=None, timeout=600):
    """Upload one file plus form fields without extra dependencies."""
    boundary = "----ftpipe" + uuid.uuid4().hex
    body = bytearray()
    for key, value in (fields or {}).items():
        body += b"--" + boundary.encode() + b"\r\n"
        body += ('Content-Disposition: form-data; name="%s"\r\n\r\n' % key).encode()
        body += str(value).encode() + b"\r\n"
    for key, (filename, content, content_type) in files.items():
        if isinstance(content, str):
            content = content.encode("utf-8")
        body += b"--" + boundary.encode() + b"\r\n"
        body += ('Content-Disposition: form-data; name="%s"; filename="%s"\r\n' % (key, filename)).encode()
        body += ("Content-Type: %s\r\n\r\n" % content_type).encode()
        body += content + b"\r\n"
    body += b"--" + boundary.encode() + b"--\r\n"

    req = urllib.request.Request(url, data=bytes(body), method="POST")
    req.add_header("Content-Type", "multipart/form-data; boundary=%s" % boundary)
    for key, value in (headers or {}).items():
        req.add_header(key, value)
    return _read(req, timeout)


def wait_for(poll_fn, is_done, is_failed, timeout_s, interval_s, label):
    """Poll an async job until it finishes. poll_fn() -> payload."""
    deadline = time.time() + float(timeout_s)
    last = None
    while True:
        payload = poll_fn()
        last = payload
        status = str(payload.get("status", "")).lower()
        if is_done(payload) or status in ("succeeded", "success", "completed", "ready", "active"):
            return payload
        if status in ("failed", "cancelled", "error", "canceled") or is_failed(payload):
            raise ProviderError("%s failed: %s" % (label, json.dumps(payload)[:800]))
        if time.time() > deadline:
            raise ProviderError(
                "%s still running after %ss. Re-run with finetune.wait=false and poll manually. Last: %s"
                % (label, timeout_s, json.dumps(last)[:400])
            )
        time.sleep(float(interval_s))


class Provider:
    """Contract every backend implements.

    export   -> write the provider's own JSONL training format
    finetune -> start/await training, return {"provider","job_id","model","status"}
    generate -> one completion for a list of chat messages
    """

    name = "base"
    env_key = ""
    base_url_env = ""
    default_base_url = ""
    supports_finetune = True
    supports_validation_file = True

    def export(self, records, out_path, with_validation=False):
        raise NotImplementedError

    def finetune(self, train_path, val_path, hyperparams, cfg):
        raise NotImplementedError

    def generate(self, model, messages, temperature=0.0, max_tokens=256):
        raise NotImplementedError

    # helpers
    def key(self):
        return api_key(self.env_key, self.name)

    def url(self, suffix=""):
        return base_url(self.base_url_env, self.default_base_url).rstrip("/") + suffix

    def auth(self):
        return {"Authorization": "Bearer " + self.key()}

    def chat_plus_messages(self, messages):
        return [{"role": m["role"], "content": m["content"]} for m in messages]
