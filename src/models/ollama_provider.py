"""Ollama local model provider (Phase 2, ADR08, NFR03).

Talks to a local Ollama daemon over its zero-cost HTTP API using only the
standard library (``urllib.request``) executed in a worker thread, so the
API event loop never blocks and no API keys or cloud services are involved.

Endpoints used:
    - ``POST {base_url}/api/generate`` for refactoring completions
    - ``GET  {base_url}/api/tags``     for availability probing

Failure policy: an unreachable daemon raises ``ProviderUnavailableError``
with a transparent remediation hint (UC03) instead of silently degrading.
"""

import asyncio
import json
import urllib.error
import urllib.request
from typing import Any

from core.exceptions import (
    ExecutionTimeoutError,
    ModelInferenceError,
    ProviderUnavailableError,
)
from core.schemas import ASTFinding
from models.base import BaseModelProvider
from models.explanations import explain_refactoring
from models.prompting import (
    REFACTORING_SYSTEM_PROMPT,
    build_user_prompt,
    extract_refactored_code,
)

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "qwen2.5-coder:1.5b"


def extract_python_code(text: str) -> str:
    """Backward-compatible alias for the canonical output extractor."""
    return extract_refactored_code(text)


class OllamaModelProvider(BaseModelProvider):
    """Local inference via an Ollama daemon (HTTP, standard library transport)."""

    def __init__(
        self,
        model: str = DEFAULT_OLLAMA_MODEL,
        base_url: str = DEFAULT_OLLAMA_URL,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        num_ctx: int = 2048,
        timeout_seconds: float = 120.0,
        probe_timeout: float = 1.5,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.num_ctx = max(512, int(num_ctx))
        self.timeout_seconds = timeout_seconds
        self.probe_timeout = probe_timeout

    # ------------------------------------------------------------------ #
    # Availability
    # ------------------------------------------------------------------ #
    @staticmethod
    def probe(
        base_url: str = DEFAULT_OLLAMA_URL,
        timeout: float = 1.5,
    ) -> tuple[bool, list[str], str]:
        """Check whether the daemon is reachable; returns (available, models, detail)."""
        url = f"{base_url.rstrip('/')}/api/tags"
        try:
            request = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            models = [
                str(entry.get("name", ""))
                for entry in payload.get("models", [])
                if entry.get("name")
            ]
            return True, models, f"Ollama daemon reachable at {base_url}."
        except Exception as exc:  # noqa: BLE001 - probe must never raise
            return (
                False,
                [],
                (
                    f"Ollama daemon unreachable at {base_url} "
                    f"({type(exc).__name__}). Start it with `ollama serve`."
                ),
            )

    def is_available(self) -> bool:
        available, _, _ = self.probe(self.base_url, timeout=self.probe_timeout)
        return available

    # ------------------------------------------------------------------ #
    # Prompt construction
    # ------------------------------------------------------------------ #
    def build_prompt(
        self,
        code: str,
        findings: list[ASTFinding],
        options: dict[str, Any],
    ) -> str:
        """Assemble the canonical refactoring prompt from agent context."""
        system_prompt = str(options.get("system_prompt") or REFACTORING_SYSTEM_PROMPT)
        plan = options.get("analysis_plan")
        if not isinstance(plan, dict):
            plan = None
        critique = options.get("review_critique") or options.get("retry_feedback")
        user_prompt = build_user_prompt(
            code,
            findings,
            analysis_plan=plan,
            review_critique=str(critique) if critique else None,
        )
        return f"{system_prompt}\n\n{user_prompt}"

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def _post_generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        temperature = float(options.get("temperature", self.temperature))
        num_predict = int(options.get("max_tokens", self.max_tokens))
        payload = {
            "model": str(options.get("model", self.model)),
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
                "num_ctx": max(512, int(options.get("num_ctx", self.num_ctx))),
            },
        }
        request = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:300]
            except Exception:  # noqa: BLE001 - best-effort diagnostics
                detail = str(exc)
            raise ModelInferenceError(
                f"Ollama HTTP {exc.code} for model '{payload['model']}': {detail}"
            ) from exc

    async def generate_refactoring(
        self,
        code: str,
        findings: list[ASTFinding],
        options: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        options = options or {}
        prompt = self.build_prompt(code, findings, options)
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(self._post_generate, prompt, options),
                timeout=self.timeout_seconds,
            )
        except TimeoutError as exc:
            raise ExecutionTimeoutError(
                f"Ollama inference exceeded timeout of {self.timeout_seconds}s (FR08)"
            ) from exc
        except (urllib.error.URLError, ConnectionError, OSError) as exc:
            raise ProviderUnavailableError(
                f"Ollama daemon unreachable at {self.base_url}: {exc}. "
                "Start it with `ollama serve` or choose another provider."
            ) from exc

        generated = str(response.get("response", ""))
        if not generated:
            raise ModelInferenceError(
                f"Ollama returned an empty response for model '{self.model}'."
            )
        refactored = extract_refactored_code(generated)

        prompt_tokens = int(response.get("prompt_eval_count") or 0) or max(1, len(prompt.split()))
        completion_tokens = int(response.get("eval_count") or 0) or max(1, len(refactored.split()))
        metadata: dict[str, Any] = {
            "provider": "ollama",
            "engine": "ollama",
            "model": str(response.get("model") or self.model),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "done_reason": response.get("done_reason"),
            "is_retry": bool(options.get("retry_feedback")),
        }
        return refactored, metadata

    async def explain_changes(
        self,
        original_code: str,
        refactored_code: str,
        diff: str,
    ) -> str:
        return explain_refactoring(
            f"Local Ollama model '{self.model}'", original_code, refactored_code
        )
