"""Abstract Base Model Provider interface (ADR08, ADR10)."""

from abc import ABC, abstractmethod
from typing import Any

from core.schemas import ASTFinding


class BaseModelProvider(ABC):
    """Abstract interface decoupling model inference from agent workflows."""

    @abstractmethod
    async def generate_refactoring(
        self,
        code: str,
        findings: list[ASTFinding],
        options: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Generate candidate refactored code.

        Args:
            code: Original Python source code.
            findings: AST findings / code smells to address.
            options: Optional inference parameters (e.g. max_tokens, temperature, retry_feedback).

        Returns:
            Tuple of (refactored_code_str, metadata_dict).
        """

    @abstractmethod
    async def explain_changes(
        self,
        original_code: str,
        refactored_code: str,
        diff: str,
    ) -> str:
        """Generate human-readable explanation of the refactoring rationale and transformations."""
