from __future__ import annotations

from .base import Provider, ProviderError
from .compat import MistralProvider, TogetherProvider
from .gemini import GeminiProvider
from .local import LocalProvider
from .mock import MockProvider
from .openai import OpenAIProvider

REGISTRY = {
    "mock": MockProvider,
    "openai": OpenAIProvider,
    "gemini": GeminiProvider,
    "mistral": MistralProvider,
    "together": TogetherProvider,
    "local": LocalProvider,
}


def get_provider(name, dry_run=False):
    """Return a provider instance. dry_run maps every real provider to mock."""
    key = (name or "mock").strip().lower()
    if dry_run and key != "mock":
        return MockProvider()
    if key not in REGISTRY:
        raise ProviderError(
            "unknown provider %r. Known: %s. Add one in src/ftpipe/providers/ and register it in "
            "providers/__init__.py." % (name, ", ".join(sorted(REGISTRY)))
        )
    return REGISTRY[key]()


def resolve(spec, dry_run=False):
    """'openai:ft:gpt-4o-mini:org::abc' -> (provider, 'openai:ft:...')."""
    spec = str(spec).strip()
    if ":" not in spec:
        return get_provider(spec, dry_run), spec
    head, _, rest = spec.partition(":")
    return get_provider(head, dry_run), rest
