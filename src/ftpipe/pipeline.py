from __future__ import annotations

import os

from .providers import get_provider, resolve

STAGES = ("validate", "build", "finetune", "evaluate")


def build_data(cfg, verbose=True):
    """Export the canonical dataset into every target provider's format.

    Returns {"records": (train, val), "paths": {provider: {...}}}.
    """
    from .data import load_jsonl, split, stats, write_jsonl

    train_path = cfg.path_for(cfg["data.train"])
    eval_path = cfg.path_for(cfg["data.eval"])
    max_chars = cfg.get("data.max_chars", 0)
    dedupe = cfg.get("data.dedupe", True)

    train_records = load_jsonl(train_path, max_chars=max_chars, dedupe=dedupe)
    eval_records = load_jsonl(eval_path, max_chars=max_chars, dedupe=dedupe)
    train_records, val_records = split(train_records, cfg.get("data.val_fraction", 0.0), cfg.get("seed", 42))

    # stop train/eval leakage
    fingerprints = {r.fingerprint() for r in train_records} | {r.fingerprint() for r in val_records}
    before = len(eval_records)
    eval_records = [r for r in eval_records if r.fingerprint() not in fingerprints]
    if len(eval_records) != before:
        print("[data] removed %d eval row(s) that also appear in train" % (before - len(eval_records)))

    out_dir = cfg.build_dir
    os.makedirs(out_dir, exist_ok=True)
    canonical_train = write_jsonl(train_records, os.path.join(out_dir, "train.canonical.jsonl"))
    canonical_eval = write_jsonl(eval_records, os.path.join(out_dir, "eval.canonical.jsonl"))

    targets = cfg.get("finetune.targets", ["mock"])
    paths = {}
    for target in targets:
        provider = get_provider(target, cfg.dry_run())
        train_out = os.path.join(out_dir, "train.%s.jsonl" % target)
        val_out = os.path.join(out_dir, "val.%s.jsonl" % target) if val_records else None
        provider.export(train_records, train_out)
        if val_records and provider.supports_validation_file:
            provider.export(val_records, val_out)
        else:
            val_out = None
        paths[target] = {"provider": provider, "train": train_out, "val": val_out}

    if verbose:
        print("[data] train %s | val %s | eval %s" % (stats(train_records), stats(val_records), stats(eval_records)))
        for target, item in paths.items():
            print("[data] %-9s -> %s" % (target, os.path.relpath(item["train"], cfg.root)))

    return {
        "train": train_records,
        "val": val_records,
        "eval": eval_records,
        "paths": paths,
        "canonical_train": canonical_train,
        "canonical_eval": canonical_eval,
    }


def finetune(cfg, data_bundle=None, only=None):
    """Run training for each configured target. Returns list of job results."""
    from .evaluation.report import write_json

    bundle = data_bundle or build_data(cfg, verbose=False)
    hyper = cfg.get("finetune.hyperparameters", {})
    targets = only or cfg.get("finetune.targets", [])
    run_dir = cfg.run_dir()
    results = []

    for target in targets:
        item = bundle["paths"].get(target)
        if item is None:
            print("[finetune] %s: no exported data (is it in finetune.targets?)" % target)
            continue
        provider = item["provider"]
        print("[finetune] %s ..." % target)
        try:
            result = provider.finetune(item["train"], item["val"], hyper, cfg)
        except Exception as exc:  # keep going: one provider failing should not stop the run
            print("[finetune] %s FAILED: %s" % (target, exc))
            results.append({"provider": target, "status": "failed", "error": str(exc)})
            continue
        results.append(result)
        print("[finetune] %s -> %s (%s)" % (target, result.get("model"), result.get("status")))

    write_json(results, os.path.join(run_dir, "finetune.json"))
    trained = [r["model"] for r in results if r.get("model")]
    if trained:
        print("[finetune] add these to evaluate.models: %s" % ", ".join(trained))
    return results


def evaluate(cfg, models=None, data_bundle=None):
    """Score every model on the eval set. Writes report.md / results.csv / metrics.json."""
    from .evaluation import judge as judge_mod
    from .evaluation.metrics import aggregate, score
    from .evaluation.report import write_json, write_markdown, write_results
    from .providers.base import ProviderError

    bundle = data_bundle or build_data(cfg, verbose=False)
    eval_records = bundle["eval"]
    limit = int(cfg.get("evaluate.limit", 0) or 0)
    if limit:
        eval_records = eval_records[:limit]

    specs = models or cfg.get("evaluate.models", [])
    metric_names = cfg.get("evaluate.metrics", ["token_f1"])
    temperature = cfg.get("evaluate.temperature", 0.0)
    max_tokens = cfg.get("evaluate.max_tokens", 256)
    dry = cfg.dry_run()
    judge_cfg = cfg.get("evaluate.judge", {}) or {}

    results, errors = {}, []
    for spec in specs:
        try:
            provider, model = resolve(spec, dry)
        except ProviderError as exc:
            errors.append(str(exc))
            print("[evaluate] skip %s: %s" % (spec, exc))
            continue

        rows = []
        print("[evaluate] %s on %d examples ..." % (spec, len(eval_records)))
        for record in eval_records:
            prompt, reference = record.prompt(), record.reference()
            try:
                prediction = provider.generate(model, record.messages, temperature=temperature, max_tokens=max_tokens)
            except Exception as exc:
                prediction = ""
                errors.append("%s: %s" % (spec, exc))
            rows.append(
                {
                    "prompt": prompt,
                    "prediction": prediction,
                    "reference": reference,
                    "metrics": score(prediction, reference, metric_names),
                }
            )

        summary = aggregate([row["metrics"] for row in rows], metric_names)
        summary["rows"] = len(rows)

        if judge_cfg.get("enabled"):
            judge_provider = get_provider(judge_cfg.get("provider", "mock"), dry)
            judge_model = judge_cfg.get("model", "judge")
            criteria = judge_cfg.get("criteria", ["correctness"])
            sample = int(judge_cfg.get("sample", 0) or 0)
            judged = rows[:sample] if sample else rows
            means = []
            for row in judged:
                verdict = judge_mod.judge(judge_provider, judge_model, row["prompt"], row["reference"],
                                          row["prediction"], criteria)
                row["judge"] = verdict
                if verdict and verdict.get("mean") is not None:
                    means.append(verdict["mean"])
            summary["judge_mean"] = round(sum(means) / len(means), 3) if means else None

        results[spec] = {"rows": rows, "summary": summary}
        print("[evaluate] %s -> %s" % (spec, {k: v for k, v in summary.items() if k != "rows"}))

    if not results:
        raise SystemExit("nothing to evaluate: %s" % ("; ".join(errors) or "no models configured"))

    run_dir = cfg.run_dir()
    meta = {
        "run_name": cfg.get("run_name"),
        "dry_run": dry,
        "train": cfg.get("data.train"),
        "eval": cfg.get("data.eval"),
        "n_train": len(bundle["train"]),
        "n_val": len(bundle["val"]),
        "n_eval": len(eval_records),
        "metrics": metric_names,
        "judge": bool(judge_cfg.get("enabled")),
        "baseline": specs[0] if specs else None,
        "errors": errors[:20],
    }
    md = write_markdown(results, os.path.join(run_dir, "report.md"), meta)
    csv_path = write_results(results, os.path.join(run_dir, "results.csv"))
    write_json({"meta": meta, "summary": {k: v["summary"] for k, v in results.items()}}, os.path.join(run_dir, "metrics.json"))
    print("[evaluate] report  -> %s" % md)
    print("[evaluate] results -> %s" % csv_path)
    return results


def run_all(cfg):
    bundle = build_data(cfg)
    jobs = finetune(cfg, bundle)
    models = list(cfg.get("evaluate.models", []))
    for job in jobs:
        model = job.get("model")
        if model and model not in models:
            models.append(model)
    return evaluate(cfg, models=list(dict.fromkeys(models)), data_bundle=bundle)
