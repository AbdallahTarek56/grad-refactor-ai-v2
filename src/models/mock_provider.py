"""Deterministic, reproducible Mock Model Provider (Phase 2, ADR08, ADR10).

Enables full integration testing without GPUs, network, or external API keys.
Code transformations are performed by the real deterministic AST engine
(``models.ast_transforms``), so the mock exercises the same refactoring
mechanics as production fallbacks. Controllable failure modes remain for
testing the bounded retry loop.
"""

from typing import Any

from core.exceptions import ExecutionTimeoutError
from core.schemas import ASTFinding
from models.ast_transforms import ASTRefactoringEngine
from models.base import BaseModelProvider
from models.explanations import explain_refactoring


class MockModelProvider(BaseModelProvider):
    """Deterministic model provider with reproducible AST transformations."""

    def __init__(
        self,
        fail_on_first_try: bool = False,
        always_fail: bool = False,
        simulate_timeout: bool = False,
    ) -> None:
        self.fail_on_first_try = fail_on_first_try
        self.always_fail = always_fail
        self.simulate_timeout = simulate_timeout
        self.call_count = 0
        self.engine = ASTRefactoringEngine()

    async def generate_refactoring(
        self,
        code: str,
        findings: list[ASTFinding],
        options: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        self.call_count += 1
        options = options or {}

        if self.simulate_timeout:
            raise ExecutionTimeoutError("Mock model inference timed out.")

        # Simulate syntax error on first attempt if requested, but fix on retry
        is_retry = bool(options.get("retry_feedback"))
        if self.always_fail or (self.fail_on_first_try and not is_retry):
            invalid_code = "def syntax_error_simulation(\n    invalid = "
            return invalid_code, {
                "prompt_tokens": len(code.split()),
                "completion_tokens": 10,
                "total_tokens": len(code.split()) + 10,
                "model": "mock-refactor-v2",
                "provider": "mock",
                "engine": "mock",
                "is_retry": is_retry,
            }

        refactored, steps = self._apply_deterministic_transformations(code, findings)

        # Estimate token usage
        prompt_tokens = max(1, len(code.split()) + sum(len(f.message.split()) for f in findings))
        completion_tokens = max(1, len(refactored.split()))

        metadata: dict[str, Any] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "model": "mock-refactor-v2",
            "provider": "mock",
            "engine": "mock",
            "transformations": steps,
            "is_retry": is_retry,
        }
        return refactored, metadata

    def _apply_deterministic_transformations(
        self, code: str, findings: list[ASTFinding]
    ) -> tuple[str, list[dict[str, Any]]]:
        """Apply real deterministic AST-to-AST refactorings for the reported rules."""
        return self.engine.transform(code, findings)

    async def explain_changes(
        self,
        original_code: str,
        refactored_code: str,
        diff: str,
    ) -> str:
        return explain_refactoring(
            "The deterministic mock provider", original_code, refactored_code
        )
