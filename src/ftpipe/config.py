from __future__ import annotations

import copy
import json
import os

DEFAULTS = {
    "run_name": "run",
    "seed": 42,
    "dry_run": False,
    "data": {
        "train": "data/raw/train.jsonl",
        "eval": "data/raw/eval.jsonl",
        "val_fraction": 0.0,
        "max_chars": 8000,
        "dedupe": True,
    },
    "finetune": {
        "targets": ["mock"],
        "hyperparameters": {"epochs": 3, "batch_size": 4, "learning_rate": 0.0001},
        "poll_timeout_s": 5400,
        "poll_interval_s": 30,
        "wait": True,
    },
    "evaluate": {
        "models": ["mock:base"],
        "metrics": ["exact_match", "token_f1", "rouge_l"],
        "limit": 0,
        "temperature": 0.0,
        "max_tokens": 256,
        "judge": {
            "enabled": False,
            "provider": "mock",
            "model": "judge",
            "criteria": ["correctness", "relevance", "clarity"],
            "sample": 0,
        },
    },
}


def _merge(base, override):
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_dotenv(path):
    """Minimal .env loader. Existing environment variables always win."""
    if not os.path.isfile(path):
        return
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


class Config:
    """Typed-ish accessor over the merged config dict.

    Relative paths resolve against the project root: the folder holding the
    config file, or its parent when the config sits in a `configs/` folder.
    So `configs/pipeline.json` maps to the repo root and keeps paths short.
    """

    def __init__(self, raw, path):
        self.raw = raw
        self.path = os.path.abspath(path)
        holder = os.path.dirname(self.path)
        if os.path.basename(holder) in ("configs", "config"):
            holder = os.path.dirname(holder)
        self.root = holder

    # -- dict helpers -----------------------------------------------------
    def get(self, dotted, default=None):
        node = self.raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def __getitem__(self, dotted):
        value = self.get(dotted, _MISSING)
        if value is _MISSING:
            raise KeyError(dotted)
        return value

    # -- paths ------------------------------------------------------------
    def path_for(self, rel):
        if not rel:
            raise ValueError("empty path in config")
        if rel.startswith("gs://") or rel.startswith("s3://") or rel.startswith("https://"):
            return rel
        if os.path.isabs(rel):
            return rel
        return os.path.normpath(os.path.join(self.root, rel))

    @property
    def build_dir(self):
        return self.path_for("build")

    @property
    def runs_dir(self):
        return self.path_for("runs")

    def run_dir(self, name=None):
        directory = os.path.join(self.runs_dir, name or self.raw["run_name"])
        os.makedirs(directory, exist_ok=True)
        return directory

    def dry_run(self):
        if os.environ.get("FTPIPE_DRY_RUN", "").lower() in ("1", "true", "yes"):
            return True
        return bool(self.raw.get("dry_run"))


_MISSING = object()


def load_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    cfg = Config(_merge(DEFAULTS, raw), path)
    load_dotenv(os.path.join(cfg.root, ".env"))
    return cfg
