from __future__ import annotations

import json
import re

JUDGE_PROMPT = """You are grading one answer against a reference answer.

Task instructions given to the model:
{task}

Reference answer:
{reference}

Model answer:
{answer}

Rate the model answer 1-5 on each criterion: {criteria}.
1 = wrong/unusable, 3 = partly right, 5 = fully correct and clear.
Reply with JSON only: {{"scores": {{"<criterion>": <int>}}, "reason": "<one sentence>"}}"""


def _parse(text):
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def judge(provider, model, task, reference, answer, criteria):
    """Return {"scores": {...}, "mean": float, "reason": str} or None if unparsable."""
    prompt = JUDGE_PROMPT.format(
        task=task[:2000],
        reference=reference[:2000],
        answer=answer[:2000],
        criteria=", ".join(criteria),
    )
    text = provider.generate(model, [{"role": "user", "content": prompt}], temperature=0.0, max_tokens=300)
    parsed = _parse(text)
    if not parsed:
        return None
    scores = parsed.get("scores") or {}
    clean = {}
    for name in criteria:
        try:
            clean[name] = float(scores.get(name))
        except (TypeError, ValueError):
            continue
    mean = round(sum(clean.values()) / len(clean), 3) if clean else None
    return {"scores": clean, "mean": mean, "reason": str(parsed.get("reason", ""))[:400]}
