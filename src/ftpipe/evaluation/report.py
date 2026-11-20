from __future__ import annotations

import csv
import json
import os


def write_results(results, path):
    """results: {model_spec: {"rows": [...], "summary": {...}}}"""
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "row", "metric", "value", "prediction_chars", "reference_chars"])
        for spec, block in results.items():
            for index, row in enumerate(block["rows"]):
                for metric, value in row["metrics"].items():
                    writer.writerow(
                        [
                            spec,
                            index,
                            metric,
                            value,
                            len(row.get("prediction", "")),
                            len(row.get("reference", "")),
                        ]
                    )
    return path


def write_markdown(results, path, meta):
    lines = []
    lines.append("# Fine-tuning evaluation: %s" % meta.get("run_name", "run"))
    lines.append("")
    if meta.get("dry_run"):
        lines.append("> **DRY RUN** - the mock provider answered every prompt. "
                     "These numbers check the pipeline, not model quality.")
        lines.append("")
    lines.append("- Data: `%s` (train rows: %s, val rows: %s)" % (meta.get("train"), meta.get("n_train"), meta.get("n_val")))
    lines.append("- Eval set: `%s` (%s rows)" % (meta.get("eval"), meta.get("n_eval")))
    lines.append("- Metrics: %s" % ", ".join(meta.get("metrics", [])))
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    header = ["model", "rows"] + list(meta.get("metrics", [])) + (["judge_mean"] if meta.get("judge") else [])
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for spec, block in results.items():
        summary = block["summary"]
        cells = [spec, str(summary.get("rows", 0))]
        for metric in meta.get("metrics", []):
            cells.append(_fmt(summary.get(metric)))
        if meta.get("judge"):
            cells.append(_fmt(summary.get("judge_mean")))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    baseline = meta.get("baseline")
    if baseline and baseline in results and len(results) > 1:
        base = results[baseline]["summary"]
        lines.append("## Change vs `%s`" % baseline)
        lines.append("")
        lines.append("| model | " + " | ".join(meta.get("metrics", [])) + " |")
        lines.append("|" + "|".join(["---"] * (len(meta.get("metrics", [])) + 1)) + "|")
        for spec, block in results.items():
            if spec == baseline:
                continue
            deltas = []
            for metric in meta.get("metrics", []):
                delta = (block["summary"].get(metric) or 0) - (base.get(metric) or 0)
                deltas.append("%+.4f" % delta)
            lines.append("| " + spec + " | " + " | ".join(deltas) + " |")
        lines.append("")

    failed = meta.get("errors") or []
    if failed:
        lines.append("## Problems")
        lines.append("")
        for item in failed:
            lines.append("- " + item)
        lines.append("")

    lines.append("## Sample outputs")
    lines.append("")
    for spec, block in list(results.items())[:3]:
        lines.append("### %s" % spec)
        lines.append("")
        for row in block["rows"][:3]:
            lines.append("- **prompt:** %s" % _clip(row.get("prompt", ""), 220))
            lines.append("  - **prediction:** %s" % _clip(row.get("prediction", ""), 220))
            lines.append("  - **reference:** %s" % _clip(row.get("reference", ""), 220))
        lines.append("")

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return path


def write_json(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, ensure_ascii=False)
    return path


def _fmt(value):
    if value is None:
        return "-"
    if isinstance(value, float):
        return "%.4f" % value
    return str(value)


def _clip(text, limit):
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "..."
