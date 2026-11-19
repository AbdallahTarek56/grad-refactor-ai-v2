"""Provider registry: name resolution and live availability catalog (Phase 6).

Single source of truth for:
    - resolving a requested provider name to a concrete provider instance
    - reporting availability for ``GET /api/v1/providers`` with a short cache
      so UI polling never hammers the Ollama daemon.
"""

import os
import threading
import time
from pathlib import Path

from models.base import BaseModelProvider
from models.local_provider import LocalModelProvider, adapter_status, hf_stack_status
from models.mock_provider import MockModelProvider
from models.ollama_provider import (
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_URL,
    OllamaModelProvider,
)
from models.smart_ast_provider import SmartASTProvider

_PROBE_TTL_SECONDS = 15.0
_ollama_probe_cache: tuple[float, tuple[bool, list[str], str]] | None = None
_DEFAULT_ADAPTER_PATH = Path("runs/lora-fp16/adapter")
_local_provider_cache: dict[tuple[str, str, str, bool], LocalModelProvider] = {}
_local_provider_cache_lock = threading.Lock()


def _local_adapter_path() -> str:
    """Return the configured adapter directory (never silently empty)."""
    return os.getenv("REFACTOR_AI_ADAPTER_PATH", "").strip() or str(_DEFAULT_ADAPTER_PATH)


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes", "on")


def _resolve_local_lora() -> LocalModelProvider:
    """Return one cached local LoRA provider for the configured adapter."""
    adapter_path = _local_adapter_path()
    base_model = os.getenv("REFACTOR_AI_BASE_MODEL", "").strip()
    load_in_4bit = _env_flag("REFACTOR_AI_LOAD_IN_4BIT")
    key = ("local_lora", adapter_path, base_model, load_in_4bit)
    with _local_provider_cache_lock:
        provider = _local_provider_cache.get(key)
        if provider is None:
            provider = LocalModelProvider(
                adapter_path=adapter_path,
                base_model_name=base_model or None,
                load_in_4bit=load_in_4bit,
                prefer_ollama=False,
                require_hf=True,
            )
            _local_provider_cache[key] = provider
        return provider


def _probe_ollama_cached(base_url: str = DEFAULT_OLLAMA_URL) -> tuple[bool, list[str], str]:
    global _ollama_probe_cache
    now = time.monotonic()
    if _ollama_probe_cache is not None:
        cached_at, result = _ollama_probe_cache
        if now - cached_at < _PROBE_TTL_SECONDS:
            return result
    result = OllamaModelProvider.probe(base_url, timeout=1.0)
    _ollama_probe_cache = (now, result)
    return result


def create_provider(
    name: str | None,
    default: BaseModelProvider | None = None,
) -> BaseModelProvider:
    """Resolve a provider name to an instance.

    ``None`` and ``"mock"`` fall back to the injected ``default`` provider so
    tests can inject configured mocks; every other name creates a fresh
    provider instance. ``local_lora`` always targets the configured PEFT
    adapter and refuses to fall back to the base model or SmartAST (NFR06).
    """
    normalized = (name or "mock").strip().lower()
    if normalized == "mock":
        return default or MockModelProvider()
    if normalized == "smart_ast":
        return SmartASTProvider()
    if normalized == "ollama":
        return OllamaModelProvider()
    if normalized == "local_lora":
        return _resolve_local_lora()
    if normalized in ("local", "auto"):
        return LocalModelProvider(prefer_ollama=True)
    raise ValueError(f"Unsupported provider: {name!r}")


def provider_catalog() -> list[dict[str, object]]:
    """Availability catalog consumed by ``GET /api/v1/providers``."""
    available, models, detail = _probe_ollama_cached()
    ollama_detail = detail
    if available and models:
        ollama_detail = f"{detail} Local models: {', '.join(models[:6])}."

    hf_available, hf_detail = hf_stack_status()
    adapter_ready, adapter_detail = adapter_status(_local_adapter_path())
    local_lora_available = hf_available and adapter_ready
    if not hf_available:
        local_lora_detail = hf_detail
    elif not adapter_ready:
        local_lora_detail = adapter_detail
    else:
        local_lora_detail = f"{hf_detail} {adapter_detail}"

    return [
        {
            "id": "mock",
            "label": "Deterministic Mock (testing)",
            "available": True,
            "engine": "mock-refactor-v2",
            "detail": (
                "Always available. Real deterministic AST transformations with "
                "failure-injection knobs for the bounded retry tests."
            ),
        },
        {
            "id": "smart_ast",
            "label": "Smart AST Transformer",
            "available": True,
            "engine": "ast-transformer-v2",
            "detail": (
                "Always available on CPU. Provably safe AST-to-AST refactoring "
                "for mutable defaults, nesting, and long functions."
            ),
        },
        {
            "id": "ollama",
            "label": f"Local Ollama ({DEFAULT_OLLAMA_MODEL})",
            "available": available,
            "engine": "ollama",
            "detail": ollama_detail,
        },
        {
            "id": "local_lora",
            "label": "Local LoRA / HuggingFace",
            "available": local_lora_available,
            "engine": "transformers+peft",
            "detail": local_lora_detail,
        },
    ]
