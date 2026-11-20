from __future__ import annotations

import re
import unicodedata
from collections import Counter

_ARTICLES = {"a", "an", "the"}


def normalize(text):
    text = unicodedata.normalize("NFKD", str(text or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def tokens(text):
    return [t for t in normalize(text).split() if t not in _ARTICLES]


def exact_match(prediction, reference):
    return 1.0 if normalize(prediction) == normalize(reference) and normalize(reference) else 0.0


def token_f1(prediction, reference):
    pred_tokens, ref_tokens = tokens(prediction), tokens(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(ref_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def _lcs(a, b):
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for token in a:
        cur = [0]
        for j, other in enumerate(b, start=1):
            cur.append(prev[j - 1] + 1 if token == other else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def rouge_l(prediction, reference):
    pred_tokens, ref_tokens = tokens(prediction), tokens(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs(pred_tokens, ref_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def contains_reference(prediction, reference):
    ref = normalize(reference)
    return 1.0 if ref and ref in normalize(prediction) else 0.0


METRICS = {
    "exact_match": exact_match,
    "token_f1": token_f1,
    "rouge_l": rouge_l,
    "contains_reference": contains_reference,
}


def score(prediction, reference, names):
    unknown = [n for n in names if n not in METRICS]
    if unknown:
        raise ValueError("unknown metric(s): %s. Available: %s" % (", ".join(unknown), ", ".join(sorted(METRICS))))
    return {name: round(METRICS[name](prediction, reference), 4) for name in names}


def aggregate(rows, names):
    out = {}
    for name in names:
        values = [row[name] for row in rows if name in row]
        out[name] = round(sum(values) / len(values), 4) if values else 0.0
    return out
