"""Tests for provider registry resolution and adapter availability."""

import json
from pathlib import Path

from models import registry
from models.local_provider import LocalModelProvider, adapter_status


def _write_adapter(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "Qwen/Qwen2.5-Coder-3B-Instruct"}),
        encoding="utf-8",
    )
    (directory / "adapter_model.safetensors").write_bytes(b"stub")
    return directory


def test_local_lora_resolves_and_caches_the_configured_adapter(monkeypatch, tmp_path):
    adapter = _write_adapter(tmp_path / "adapter")
    monkeypatch.setenv("REFACTOR_AI_ADAPTER_PATH", str(adapter))
    monkeypatch.delenv("REFACTOR_AI_BASE_MODEL", raising=False)
    monkeypatch.delenv("REFACTOR_AI_LOAD_IN_4BIT", raising=False)

    provider = registry.create_provider("local_lora")
    assert isinstance(provider, LocalModelProvider)
    assert provider.adapter_path == str(adapter)
    assert provider.require_hf is True
    assert provider.prefer_ollama is False
    assert provider.load_in_4bit is False
    assert registry.create_provider("local_lora") is provider


def test_local_lora_cache_keys_on_the_configured_adapter(monkeypatch, tmp_path):
    first = _write_adapter(tmp_path / "first")
    second = _write_adapter(tmp_path / "second")
    monkeypatch.setenv("REFACTOR_AI_ADAPTER_PATH", str(first))
    first_provider = registry.create_provider("local_lora")
    monkeypatch.setenv("REFACTOR_AI_ADAPTER_PATH", str(second))
    second_provider = registry.create_provider("local_lora")

    assert first_provider is not second_provider
    assert second_provider.adapter_path == str(second)


def test_local_and_auto_keep_the_ollama_fallback():
    assert registry.create_provider("local").prefer_ollama is True
    assert registry.create_provider("auto").prefer_ollama is True


def test_catalog_reports_a_missing_adapter_as_unavailable(monkeypatch, tmp_path):
    missing = tmp_path / "missing"
    monkeypatch.setenv("REFACTOR_AI_ADAPTER_PATH", str(missing))
    monkeypatch.setattr(registry, "hf_stack_status", lambda: (True, "stack available"))
    monkeypatch.setattr(
        registry,
        "_probe_ollama_cached",
        lambda base_url=registry.DEFAULT_OLLAMA_URL: (False, [], "Ollama not reachable."),
    )

    catalog = {entry["id"]: entry for entry in registry.provider_catalog()}
    assert catalog["local_lora"]["available"] is False
    assert "not found" in str(catalog["local_lora"]["detail"])


def test_catalog_reports_a_ready_adapter(monkeypatch, tmp_path):
    adapter = _write_adapter(tmp_path / "adapter")
    monkeypatch.setenv("REFACTOR_AI_ADAPTER_PATH", str(adapter))
    monkeypatch.setattr(registry, "hf_stack_status", lambda: (True, "stack available"))
    monkeypatch.setattr(
        "models.local_provider._module_available",
        lambda name: True,
    )
    monkeypatch.setattr(
        registry,
        "_probe_ollama_cached",
        lambda base_url=registry.DEFAULT_OLLAMA_URL: (False, [], "Ollama not reachable."),
    )

    catalog = {entry["id"]: entry for entry in registry.provider_catalog()}
    assert catalog["local_lora"]["available"] is True
    assert "Qwen/Qwen2.5-Coder-3B-Instruct" in str(catalog["local_lora"]["detail"])


def test_adapter_status_requires_config_and_weights(tmp_path):
    directory = tmp_path / "adapter"
    directory.mkdir()
    ready, detail = adapter_status(directory)
    assert ready is False
    assert "adapter_config.json" in detail

    (directory / "adapter_config.json").write_text("{}", encoding="utf-8")
    ready, detail = adapter_status(directory)
    assert ready is False
    assert "weights" in detail
