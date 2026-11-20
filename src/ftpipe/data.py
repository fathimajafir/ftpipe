from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List

ROLES = ("system", "user", "assistant")


class DataError(ValueError):
    pass


@dataclass
class Record:
    """One training/eval example, normalised to chat messages."""

    messages: List[Dict[str, str]]
    meta: Dict[str, str] = field(default_factory=dict)

    def prompt(self):
        """Everything the model sees: all messages except a trailing assistant turn."""
        msgs = list(self.messages)
        if msgs and msgs[-1]["role"] == "assistant":
            msgs = msgs[:-1]
        parts = [m["content"] for m in msgs if m["role"] != "system"]
        system = [m["content"] for m in msgs if m["role"] == "system"]
        body = "\n".join(parts).strip()
        return (system[0] + "\n\n" + body).strip() if system else body

    def reference(self):
        if self.messages and self.messages[-1]["role"] == "assistant":
            return self.messages[-1]["content"]
        return ""

    def to_json(self):
        out = {"messages": self.messages}
        if self.meta:
            out["meta"] = self.meta
        return out

    def fingerprint(self):
        blob = json.dumps(self.messages, sort_keys=True, ensure_ascii=False)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()

    def size(self):
        return sum(len(m["content"]) for m in self.messages)


def normalize_row(obj, line_no=0):
    """Accept the common JSONL shapes and return a Record.

    Supported:
      {"messages": [{"role": ..., "content": ...}, ...]}
      {"prompt": "...", "completion": "..."}
      {"instruction": "...", "input": "...", "output": "..."}
    """
    if not isinstance(obj, dict):
        raise DataError("line %s: expected a JSON object" % line_no)

    if "messages" in obj:
        raw = obj["messages"]
        if not isinstance(raw, list) or not raw:
            raise DataError("line %s: 'messages' must be a non-empty list" % line_no)
        messages = []
        for item in raw:
            if not isinstance(item, dict):
                raise DataError("line %s: message must be an object" % line_no)
            role = str(item.get("role", "")).strip().lower()
            content = item.get("content")
            if role not in ROLES:
                raise DataError("line %s: bad role %r" % (line_no, item.get("role")))
            if not isinstance(content, str) or not content.strip():
                raise DataError("line %s: empty content for role %s" % (line_no, role))
            messages.append({"role": role, "content": content.strip()})
    elif "prompt" in obj and "completion" in obj:
        messages = [
            {"role": "user", "content": str(obj["prompt"]).strip()},
            {"role": "assistant", "content": str(obj["completion"]).strip()},
        ]
    elif "instruction" in obj and "output" in obj:
        prompt = str(obj["instruction"]).strip()
        extra = str(obj.get("input", "") or "").strip()
        if extra:
            prompt = prompt + "\n\n" + extra
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": str(obj["output"]).strip()},
        ]
    else:
        raise DataError(
            "line %s: unrecognised schema; need one of messages / prompt+completion / instruction+output"
            % line_no
        )

    if len(messages) < 2:
        raise DataError("line %s: need at least one non-assistant and one assistant message" % line_no)
    if messages[-1]["role"] != "assistant":
        raise DataError("line %s: last message must be from the assistant" % line_no)
    if "user" not in [m["role"] for m in messages[:-1]]:
        raise DataError("line %s: no user message before the assistant reply" % line_no)

    meta = {k: str(v) for k, v in (obj.get("meta") or {}).items()} if isinstance(obj.get("meta"), dict) else {}
    return Record(messages=messages, meta=meta)


def load_jsonl(path, max_chars=0, dedupe=True):
    if not os.path.isfile(path):
        raise DataError("dataset not found: %s" % path)
    records, seen, problems = [], set(), []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                record = normalize_row(obj, line_no)
            except (json.JSONDecodeError, DataError) as exc:
                problems.append(str(exc))
                continue
            if max_chars and record.size() > max_chars:
                problems.append("line %s: dropped, %d chars > max_chars %d" % (line_no, record.size(), max_chars))
                continue
            if dedupe:
                fp = record.fingerprint()
                if fp in seen:
                    problems.append("line %s: duplicate of an earlier row" % line_no)
                    continue
                seen.add(fp)
            records.append(record)
    if problems:
        print("[data] %d row(s) skipped:" % len(problems))
        for item in problems[:10]:
            print("  - " + item)
        if len(problems) > 10:
            print("  - ... %d more" % (len(problems) - 10))
    if not records:
        raise DataError("no usable rows in %s" % path)
    return records


def write_jsonl(records, path, mapper=None):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            row = mapper(record) if mapper else record.to_json()
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def split(records, val_fraction, seed=42):
    """Deterministic train/val split. Returns (train, val)."""
    if val_fraction <= 0 or len(records) < 2:
        return records, []
    rows = list(records)
    random.Random(seed).shuffle(rows)
    n_val = max(1, int(round(len(rows) * val_fraction)))
    n_val = min(n_val, len(rows) - 1)
    return rows[n_val:], rows[:n_val]


def stats(records):
    lengths = [r.size() for r in records]
    refs = [len(r.reference()) for r in records]
    return {
        "rows": len(records),
        "max_chars": max(lengths) if lengths else 0,
        "mean_chars": round(sum(lengths) / len(lengths), 1) if lengths else 0,
        "max_reference_chars": max(refs) if refs else 0,
        "empty_references": sum(1 for x in refs if x == 0),
        "turns": sum(len(r.messages) for r in records),
    }
