# ftpipe — text fine-tuning + evaluation pipeline

One small pipeline that takes a JSONL text dataset, exports it into the format each
LLM vendor wants, starts the fine-tune, then scores the tuned model against a
baseline on a held-out set — with the same metrics and the same report for every
vendor.

No runtime dependencies: it runs on a stock Python 3.9+.

```
data/raw/*.jsonl
      │  validate + normalise + dedupe + train/eval leak check
      ▼
build/train.<provider>.jsonl      (openai / gemini / mistral / together / local)
      │  upload + create job + poll
      ▼
runs/<run_name>/finetune.json     (job ids, model ids)
      │  generate on the eval set, score, optional LLM judge
      ▼
runs/<run_name>/report.md  results.csv  metrics.json
```

## Quickstart (offline, no keys, ~2 s)

```bash
cd fine-tuning
export PYTHONPATH=src          # or: make install

python3 -m ftpipe validate -c configs/pipeline.json   # check your data
make demo                                             # full build → finetune → eval, mock models
open runs/demo-offline/report.md
```

The demo uses a `mock` provider that fakes a partially-trained model. Its numbers
are synthetic: they prove the wiring works, not that a model is good.

## Real run

1. `cp .env.example .env` and fill in the keys for the vendors you use.
2. Put your data in `data/raw/train.jsonl` and `data/raw/eval.jsonl`.
3. Trim `finetune.targets` in `configs/pipeline.json` to the vendors you want.
4. Run:

```bash
python3 -m ftpipe build    -c configs/pipeline.json --sample   # inspect exports first
python3 -m ftpipe finetune -c configs/pipeline.json --target openai
python3 -m ftpipe evaluate -c configs/pipeline.json \
    --models mock:base,openai:ft:gpt-4o-mini-2024-07-18:acme::abc123
```

`--dry-run` on any command swaps every real provider for the mock one, so you can
rehearse a run (and check your data/config) without spending money.

## Data format

One JSON object per line. Any of these shapes is accepted and normalised:

```jsonc
{"messages": [{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
{"prompt": "...", "completion": "..."}
{"instruction": "...", "input": "...", "output": "..."}
```

Rules enforced by `validate` / `build`:
- last message must be `assistant` (that is the target),
- a `user` turn must exist,
- empty rows are rejected, duplicates dropped, over-long rows skipped (`data.max_chars`),
- rows that appear in **both** train and eval are removed from eval (leak guard).

`build` also splits off a validation slice (`data.val_fraction`) when a provider
accepts one.

## Providers

| target     | what it does | needs | notes |
|------------|--------------|-------|-------|
| `openai`   | upload → `/fine_tuning/jobs` → poll | `OPENAI_API_KEY` | chat-format JSONL, `n_epochs` |
| `gemini`   | creates a `tuningJob` | `GEMINI_API_KEY` + data in GCS | set `finetune.train_uri` to `gs://...`; the API cannot read local files |
| `mistral`  | upload → `/fine_tuning/jobs` → poll | `MISTRAL_API_KEY` | `training_steps` instead of epochs |
| `together` | upload → `/fine-tunes` → poll | `TOGETHER_API_KEY` | open-weight base models |
| `local`    | writes `runs/<run>/local/train_lora.py` | `pip install "ftpipe[local]"` + a GPU | LoRA for Qwen/Llama/etc.; run the script yourself, then evaluate via your vLLM/Ollama endpoint |
| `mock`     | offline fake | — | for tests and demos |

Azure OpenAI / self-hosted gateways: set `OPENAI_BASE_URL` (and `*_BASE_URL` for the
others) in `.env`.

### Adding a provider

```python
# src/ftpipe/providers/acme.py
from .base import Provider, http_json

class AcmeProvider(Provider):
    name = "acme"
    env_key = "ACME_API_KEY"
    default_base_url = "https://api.acme.dev/v1"

    def export(self, records, out_path, with_validation=False):
        from ..data import write_jsonl
        return write_jsonl(records, out_path, mapper=lambda r: {"messages": r.messages})

    def finetune(self, train_path, val_path, hyperparams, cfg):
        ...                       # upload, create job, poll → {"provider","job_id","model","status"}

    def generate(self, model, messages, temperature=0.0, max_tokens=256):
        ...                       # one completion, returns text
```

Register it in `src/ftpipe/providers/__init__.py` `REGISTRY` and it is usable as
`"targets": ["acme"]` and `--models acme:<model-id>`.

## Evaluation

Metrics run per example and are averaged (`runs/<run>/metrics.json`):

| metric | meaning | good for |
|--------|---------|----------|
| `exact_match` | normalised string equality | classification, extraction, closed-form answers |
| `token_f1` | SQuAD-style token overlap F1 | short free-form answers |
| `rouge_l` | longest-common-subsequence F1 | summaries / paraphrases |
| `contains_reference` | reference appears in the prediction | when extra preamble is fine |

Optional LLM judge (`evaluate.judge.enabled`): a second model scores each answer
1–5 on `correctness / relevance / clarity` and the mean lands in `judge_mean`.
Use a judge from a different family than the model under test.

The report shows a summary table, the delta of every model against the first
(baseline) one, and a few sample predictions side by side.

Set `evaluate.models` to `mock:base` plus each tuned model id (the `finetune`
step prints exactly which ids to paste in).

## Config reference (`configs/pipeline.json`)

| key | default | notes |
|-----|---------|-------|
| `run_name` | `run` | output folder under `runs/` |
| `seed` | `42` | val split |
| `data.train` / `data.eval` | — | paths relative to the repo root |
| `data.val_fraction` | `0` | 0 = no validation split |
| `data.max_chars` | `8000` | rows bigger than this are skipped |
| `finetune.targets` | `["mock"]` | run in order |
| `finetune.base_model` | per vendor | base model id |
| `finetune.hyperparameters` | `{epochs, batch_size, learning_rate, lora_*}` | mapped per vendor |
| `finetune.wait` / `poll_timeout_s` | `true` / `5400` | `wait:false` submits and exits |
| `evaluate.models` | `["mock:base"]` | `provider:model-id` specs |
| `evaluate.limit` | `0` | 0 = all eval rows; small numbers for cheap smoke runs |
| `evaluate.metrics` | the three above | |

## Layout

```
configs/           pipeline.json (real), demo.json (offline)
data/raw/          train.jsonl, eval.jsonl
src/ftpipe/        config, data, pipeline, cli
  providers/       base (+http helpers), openai, gemini, compat (mistral+together), local, mock
  evaluation/      metrics, judge, report
tests/             unittest suite (data, metrics, providers)
build/             generated per-provider files   (gitignored)
runs/<run>/        report.md, results.csv, metrics.json, finetune.json (gitignored)
```

## Practical notes

- **Cost:** every real run spends money twice — training, then one generation per
  eval row per model. Start with `evaluate.limit: 20` and a small base model.
- **Minimum data:** most vendors want 10–100 examples to show any effect; below
  ~50 the eval will be noise. Prefer a narrow task with a machine-checkable output.
- **Baseline first:** always keep one untuned model in `evaluate.models`, otherwise
  a score means nothing.
- **Leakage:** `build` removes train/eval duplicates, but near-duplicates (same
  ticket reworded) are on you — dedupe on a normalised prompt if it matters.
- **Secrets:** keys live in `.env` only (gitignored). Job payloads in `runs/` may
  contain sample data — check before sharing a run folder.
