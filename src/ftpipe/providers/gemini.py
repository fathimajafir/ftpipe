from __future__ import annotations

from .base import Provider, ProviderError, http_json, wait_for


class GeminiProvider(Provider):
    """Gemini supervised tuning.

    Gemini reads training data from Cloud Storage, not from an uploaded file.
    Put the exported JSONL in a bucket and point `finetune.train_uri`
    (and optionally `finetune.val_uri`) at gs:// paths.
    """

    name = "gemini"
    env_key = "GEMINI_API_KEY"
    base_url_env = "GEMINI_BASE_URL"
    default_base_url = "https://generativelanguage.googleapis.com/v1beta"
    supports_validation_file = True

    def export(self, records, out_path, with_validation=False):
        from ..data import write_jsonl

        def mapper(record):
            contents = []
            for message in record.messages:
                role = "model" if message["role"] == "assistant" else "user"
                contents.append({"role": role, "parts": [{"text": message["content"]}]})
            return {"contents": contents}

        return write_jsonl(records, out_path, mapper=mapper)

    def finetune(self, train_path, val_path, hyperparams, cfg):
        train_uri = cfg.get("finetune.train_uri") or train_path
        if not str(train_uri).startswith("gs://"):
            raise ProviderError(
                "gemini: training data must be a gs:// URI. Upload build/train.gemini.jsonl to a bucket and set "
                '"finetune": {"train_uri": "gs://bucket/train.jsonl"} in the config.'
            )
        body = {
            "displayName": cfg.get("run_name", "ftpipe-run").replace("/", "-")[:40],
            "baseModel": cfg.get("finetune.base_model", "gemini-2.0-flash-001"),
            "tuningDatasetUri": train_uri,
            "epochCount": str(int(hyperparams.get("epochs", 3))),
        }
        val_uri = cfg.get("finetune.val_uri")
        if val_uri:
            body["validationDatasetUri"] = val_uri
        job = http_json(self.url("/tuningJobs"), body, headers={"x-goog-api-key": self.key()})
        job_name = job.get("name") or job.get("metadata", {}).get("name")
        print("[gemini] tuning job %s submitted" % job_name)
        if not cfg.get("finetune.wait", True):
            return {"provider": self.name, "job_id": job_name, "model": None, "status": "running"}

        result = wait_for(
            lambda: http_json(self.url("/" + str(job_name)), headers={"x-goog-api-key": self.key()}, method="GET"),
            is_done=lambda p: bool(p.get("tunedModel", {}).get("model")),
            is_failed=lambda p: bool(p.get("error")),
            timeout_s=cfg.get("finetune.poll_timeout_s", 5400),
            interval_s=cfg.get("finetune.poll_interval_s", 30),
            label="gemini tuning %s" % job_name,
        )
        return {
            "provider": self.name,
            "job_id": job_name,
            "model": "gemini:" + result["tunedModel"]["model"],
            "status": result.get("state"),
        }

    def generate(self, model, messages, temperature=0.0, max_tokens=256):
        if not model:
            raise ProviderError("gemini: no model id (training may still be running)")
        contents = []
        for message in messages:
            role = "model" if message["role"] == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": message["content"]}]})
        payload = http_json(
            self.url("/" + str(model).lstrip("/") + ":generateContent"),
            {
                "contents": contents,
                "generationConfig": {
                    "temperature": float(temperature),
                    "maxOutputTokens": int(max_tokens),
                },
            },
            headers={"x-goog-api-key": self.key()},
        )
        candidates = payload.get("candidates") or []
        if not candidates:
            raise ProviderError("gemini: empty response: %s" % str(payload)[:400])
        parts = candidates[0].get("content", {}).get("parts", [])
        return "".join(part.get("text", "") for part in parts)
