"""SmartASTProvider — deterministic CPU refactoring engine (Phase 2, ADR08).

The final provider tier: applies provably safe AST-to-AST transformations
(``models.ast_transforms``) to the rules reported by the Analysis Agent.
Always available, zero dependencies, fully deterministic, and never
executes submitted code (NFR01).
"""

from typing import Any

from core.schemas import ASTFinding
from models.ast_transforms import ASTRefactoringEngine
from models.base import BaseModelProvider
from models.explanations import explain_refactoring


class SmartASTProvider(BaseModelProvider):
    """Rule-guided AST transformer used as the final CPU-safe fallback tier."""

    def __init__(self, max_function_lines: int = 30) -> None:
        self.engine = ASTRefactoringEngine(max_function_lines=max_function_lines)

    async def generate_refactoring(
        self,
        code: str,
        findings: list[ASTFinding],
        options: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        refactored, steps = self.engine.transform(code, findings)
        prompt_tokens = max(1, len(code.split()) + sum(len(f.message.split()) for f in findings))
        completion_tokens = max(1, len(refactored.split()))
        metadata: dict[str, Any] = {
            "provider": "smart_ast",
            "engine": "smart_ast",
            "model": "ast-transformer-v2",
            "transformations": steps,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "is_retry": bool((options or {}).get("retry_feedback")),
        }
        return refactored, metadata

    async def explain_changes(
        self,
        original_code: str,
        refactored_code: str,
        diff: str,
    ) -> str:
        return explain_refactoring("The Smart AST Transformer", original_code, refactored_code)
