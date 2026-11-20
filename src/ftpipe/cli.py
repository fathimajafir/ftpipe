from __future__ import annotations

import argparse
import json
import sys

from .config import load_config
from .pipeline import build_data, evaluate, finetune, run_all

DESCRIPTION = """ftpipe - small, provider-agnostic text fine-tuning + evaluation pipeline.

Examples
  python -m ftpipe validate  -c configs/pipeline.json
  python -m ftpipe build     -c configs/pipeline.json
  python -m ftpipe finetune  -c configs/pipeline.json --target openai
  python -m ftpipe evaluate  -c configs/pipeline.json --models mock:base,mock:ft
  python -m ftpipe run       -c configs/pipeline.json --dry-run
"""


def main(argv=None):
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-c", "--config", default=argparse.SUPPRESS, help="path to pipeline config")
    common.add_argument("--dry-run", action="store_true", default=argparse.SUPPRESS,
                        help="force the offline mock provider everywhere")

    parser = argparse.ArgumentParser(prog="ftpipe", description=DESCRIPTION,
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     parents=[common])
    parser.set_defaults(config="configs/pipeline.json", dry_run=False)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("validate", parents=[common], help="check the datasets without calling any provider")

    p_build = sub.add_parser("build", parents=[common], help="normalise data and export per-provider JSONL")
    p_build.add_argument("--sample", action="store_true", help="print a couple of exported rows")

    p_train = sub.add_parser("finetune", parents=[common], help="submit/await fine-tuning jobs")
    p_train.add_argument("--target", action="append", help="restrict to one or more providers")

    p_eval = sub.add_parser("evaluate", parents=[common], help="score models on the eval set")
    p_eval.add_argument("--models", help="comma-separated model specs, e.g. mock:base,openai:ft:gpt-4o-mini:org::id")
    p_eval.add_argument("--limit", type=int, help="cap the number of eval rows")

    sub.add_parser("run", parents=[common], help="build -> finetune -> evaluate")

    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.dry_run:
        cfg.raw["dry_run"] = True

    try:
        if args.command == "validate":
            return _validate(cfg)
        if args.command == "build":
            bundle = build_data(cfg)
            if args.sample:
                _sample(bundle)
            return 0
        if args.command == "finetune":
            finetune(cfg, only=args.target)
            return 0
        if args.command == "evaluate":
            models = [m.strip() for m in args.models.split(",")] if args.models else None
            if args.limit is not None:
                cfg.raw["evaluate"]["limit"] = args.limit
            evaluate(cfg, models=models)
            return 0
        if args.command == "run":
            run_all(cfg)
            return 0
    except KeyboardInterrupt:
        print("\naborted", file=sys.stderr)
        return 130
    except Exception as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 1
    return 0


def _validate(cfg):
    from .data import load_jsonl, stats

    print("config: %s" % cfg.path)
    print("root:   %s" % cfg.root)
    total = 0
    for split_name in ("train", "eval"):
        path = cfg.path_for(cfg["data.%s" % split_name])
        records = load_jsonl(path, max_chars=cfg.get("data.max_chars", 0), dedupe=cfg.get("data.dedupe", True))
        print("%-5s %-42s %s" % (split_name, path, stats(records)))
        total += len(records)
    print("providers: %s" % ", ".join(cfg.get("finetune.targets", [])))
    print("models:    %s" % ", ".join(cfg.get("evaluate.models", [])))
    print("OK - %d rows validated" % total)
    return 0


def _sample(bundle):
    for target, item in bundle["paths"].items():
        with open(item["train"], "r", encoding="utf-8") as handle:
            first = json.loads(handle.readline())
        print("\n[%s] %s" % (target, item["train"]))
        print(json.dumps(first, ensure_ascii=False)[:400])


if __name__ == "__main__":
    raise SystemExit(main())
