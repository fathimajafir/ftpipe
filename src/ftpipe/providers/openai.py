from __future__ import annotations

import os

from .base import Provider, ProviderError, http_json, http_multipart, wait_for


class OpenAIProvider(Provider):
    name = "openai"
    env_key = "OPENAI_API_KEY"
    base_url_env = "OPENAI_BASE_URL"
    default_base_url = "https://api.openai.com/v1"

    def export(self, records, out_path, with_validation=False):
        from ..data import write_jsonl

        return write_jsonl(records, out_path, mapper=lambda r: {"messages": r.messages})

    def _upload(self, path):
        with open(path, "rb") as handle:
            content = handle.read()
        payload = http_multipart(
            self.url("/files"),
            fields={"purpose": "fine-tune"},
            files={"file": (os.path.basename(path), content, "application/jsonl")},
            headers=self.auth(),
        )
        return payload["id"]

    def finetune(self, train_path, val_path, hyperparams, cfg):
        train_id = self._upload(train_path)
        body = {
            "model": cfg.get("finetune.base_model", "gpt-4o-mini-2024-07-18"),
            "training_file": train_id,
            "hyperparameters": {
                "n_epochs": int(hyperparams.get("epochs", 3)),
            },
        }
        if val_path:
            body["validation_file"] = self._upload(val_path)

        job = http_json(self.url("/fine_tuning/jobs"), body, headers=self.auth())
        job_id = job["id"]
        print("[openai] fine-tune job %s submitted" % job_id)
        if not cfg.get("finetune.wait", True):
            return {"provider": self.name, "job_id": job_id, "model": None, "status": job.get("status")}

        result = wait_for(
            lambda: http_json(self.url("/fine_tuning/jobs/" + job_id), headers=self.auth(), method="GET"),
            is_done=lambda p: bool(p.get("fine_tuned_model")),
            is_failed=lambda p: bool(p.get("error")),
            timeout_s=cfg.get("finetune.poll_timeout_s", 5400),
            interval_s=cfg.get("finetune.poll_interval_s", 30),
            label="openai fine-tune %s" % job_id,
        )
        return {
            "provider": self.name,
            "job_id": job_id,
            "model": "openai:" + str(result.get("fine_tuned_model")),
            "status": result.get("status"),
        }

    def generate(self, model, messages, temperature=0.0, max_tokens=256):
        if not model:
            raise ProviderError("openai: no model id (training may still be running)")
        payload = http_json(
            self.url("/chat/completions"),
            {
                "model": model,
                "messages": self.chat_plus_messages(messages),
                "temperature": float(temperature),
                "max_tokens": int(max_tokens),
            },
            headers=self.auth(),
        )
        return payload["choices"][0]["message"]["content"]
