from __future__ import annotations

import os

from .base import Provider, ProviderError, http_json, http_multipart, wait_for


class OpenAICompatibleMixin:
    """Shared upload + chat-completions code for OpenAI-shaped APIs."""

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

    def generate(self, model, messages, temperature=0.0, max_tokens=256):
        if not model:
            raise ProviderError("%s: no model id (training may still be running)" % self.name)
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


class MistralProvider(OpenAICompatibleMixin, Provider):
    name = "mistral"
    env_key = "MISTRAL_API_KEY"
    base_url_env = "MISTRAL_BASE_URL"
    default_base_url = "https://api.mistral.ai/v1"

    def finetune(self, train_path, val_path, hyperparams, cfg):
        files = [{"file_id": self._upload(train_path), "weight": 1}]
        body = {
            "model": cfg.get("finetune.base_model", "open-mistral-7b"),
            "training_files": files,
            "hyperparameters": {"training_steps": int(hyperparams.get("training_steps", 100))},
        }
        if val_path:
            body["validation_files"] = [self._upload(val_path)]
        job = http_json(self.url("/fine_tuning/jobs"), body, headers=self.auth())
        job_id = job["id"]
        print("[mistral] fine-tune job %s submitted" % job_id)
        if not cfg.get("finetune.wait", True):
            return {"provider": self.name, "job_id": job_id, "model": None, "status": job.get("status")}
        result = wait_for(
            lambda: http_json(self.url("/fine_tuning/jobs/" + job_id), headers=self.auth(), method="GET"),
            is_done=lambda p: bool(p.get("fine_tuned_model")),
            is_failed=lambda p: bool(p.get("error_message")),
            timeout_s=cfg.get("finetune.poll_timeout_s", 5400),
            interval_s=cfg.get("finetune.poll_interval_s", 30),
            label="mistral fine-tune %s" % job_id,
        )
        return {
            "provider": self.name,
            "job_id": job_id,
            "model": "mistral:" + str(result["fine_tuned_model"]),
            "status": result.get("status"),
        }


class TogetherProvider(OpenAICompatibleMixin, Provider):
    name = "together"
    env_key = "TOGETHER_API_KEY"
    base_url_env = "TOGETHER_BASE_URL"
    default_base_url = "https://api.together.xyz/v1"

    def finetune(self, train_path, val_path, hyperparams, cfg):
        body = {
            "training_file": self._upload(train_path),
            "model": cfg.get("finetune.base_model", "meta-llama/Meta-Llama-3.1-8B-Instruct-Reference"),
            "n_epochs": int(hyperparams.get("epochs", 3)),
            "n_checkpoints": 1,
            "suffix": cfg.get("run_name", "ftpipe")[:18],
        }
        if val_path:
            body["validation_file"] = self._upload(val_path)
        job = http_json(self.url("/fine-tunes"), body, headers=self.auth())
        job_id = job.get("id") or job.get("job_id")
        print("[together] fine-tune job %s submitted" % job_id)
        if not cfg.get("finetune.wait", True):
            return {"provider": self.name, "job_id": job_id, "model": None, "status": job.get("status")}
        result = wait_for(
            lambda: http_json(self.url("/fine-tunes/" + str(job_id)), headers=self.auth(), method="GET"),
            is_done=lambda p: bool(p.get("output_name")),
            is_failed=lambda p: bool(p.get("error")),
            timeout_s=cfg.get("finetune.poll_timeout_s", 5400),
            interval_s=cfg.get("finetune.poll_interval_s", 30),
            label="together fine-tune %s" % job_id,
        )
        return {
            "provider": self.name,
            "job_id": job_id,
            "model": "together:" + str(result["output_name"]),
            "status": result.get("status"),
        }
