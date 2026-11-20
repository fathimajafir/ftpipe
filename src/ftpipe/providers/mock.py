from __future__ import annotations

import json
import re

from .base import Provider

# --- tiny rule-based stand-ins for the demo task (support-ticket triage) -----
_BILLING = ("charge", "billed", "bill", "invoice", "refund", "card", "vat", "subscription", "seats", "plan")
_TECHNICAL = ("crash", "error", "fail", "time out", "timed out", "logs me out", "upload", "search",
              "api", "release", "spins", "broken", "500")
_ACCOUNT = ("email", "delete my account", "admin", "two factor", "2fa", "ownership", "contact", "login")
_URGENT = ("twice", "crash", "forever", "completely", "declined", "duplicate", "500", "broke", "missing")


def _guess(text):
    lowered = text.lower()
    scores = {
        "billing": sum(1 for w in _BILLING if w in lowered),
        "technical": sum(1 for w in _TECHNICAL if w in lowered),
        "account": sum(1 for w in _ACCOUNT if w in lowered),
    }
    category = max(scores, key=scores.get) if max(scores.values()) else "account"
    urgency = "high" if any(w in lowered for w in _URGENT) else ("medium" if scores[category] > 1 else "low")
    return category, urgency


class MockProvider(Provider):
    """Offline provider so the whole pipeline runs with no keys and no cost.

    `mock:ft` behaves like a model that learned the demo triage task; `mock:base`
    behaves like an untuned model that rambles. Scores are synthetic - they prove
    the plumbing works, they say nothing about a real model.
    """

    name = "mock"
    supports_finetune = True
    supports_validation_file = True

    def export(self, records, out_path, with_validation=False):
        from ..data import write_jsonl

        return write_jsonl(records, out_path)

    def finetune(self, train_path, val_path, hyperparams, cfg):
        return {
            "provider": self.name,
            "job_id": "mock-job",
            "model": "mock:ft",
            "status": "succeeded",
            "note": "mock training - nothing was actually trained",
        }

    def generate(self, model, messages, temperature=0.0, max_tokens=256):
        prompt = " ".join(m["content"] for m in messages if m["role"] != "system").strip()

        if "You are grading one answer" in prompt:
            return self._judge(prompt)

        tag = str(model).split(":")[-1]
        category, urgency = _guess(prompt)
        if tag == "ft":
            return json.dumps({"category": category, "urgency": urgency})
        # untuned baseline: verbose, hedged, not machine-readable
        return (
            "It sounds like this is mostly a %s related request. I would treat the urgency as %s, "
            "but a human should confirm before anything is changed."
            % (category, urgency)
        )

    @staticmethod
    def _judge(prompt):
        answer = ""
        match = re.search(r"Model answer:\s*(.*?)\s*Rate the model answer", prompt, re.S)
        if match:
            answer = match.group(1).strip()
        criteria = ["correctness"]
        found = re.search(r"criterion[s]?: (.+?)\.", prompt)
        if found:
            criteria = [c.strip() for c in found.group(1).split(",") if c.strip()]
        if not answer:
            score = 1
        elif answer.lstrip().startswith("{"):
            score = 5
        elif len(answer) > 15:
            score = 3
        else:
            score = 2
        return json.dumps({"scores": {c: score for c in criteria}, "reason": "mock judge (offline)"})


def judge_reply(messages):  # kept for backwards compatibility with older configs
    return MockProvider().generate("judge", messages)
