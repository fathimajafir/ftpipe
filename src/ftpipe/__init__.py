"""ftpipe: a small, provider-agnostic fine-tuning and evaluation pipeline."""

__version__ = "0.1.0"

from .config import load_config  # noqa: F401
from .pipeline import build_data, evaluate, finetune, run_all  # noqa: F401
