"""Canonical prompt and output contract shared by training and inference.

Keeping one renderer is the cheapest accuracy fix in the project: the LoRA
adapter must see exactly the same system prompt, findings format, code fence,
chat template and end-of-turn token during training that the provider uses
during evaluation (and that the web application uses at runtime).

The functions here are deliberately tokenizer-agnostic except for
:func:`render_prompt`, which uses ``apply_chat_template`` when the tokenizer
provides one and falls back to a plain instruction format otherwise.
"""

from __future__ import annotations

from typing import Any

from core.schemas import ASTFinding

REFACTORING_SYSTEM_PROMPT = (
    "You are the Refactoring Agent inside a three-role multi-agent system "
    "(Analysis -> Refactoring -> Review).\n"
    "Your ONLY job is to rewrite the provided Python module so that the "
    "reported AST findings are resolved while the observable behaviour of "
    "every function and class is preserved exactly.\n"
    "Hard rules:\n"
    "1. Keep every existing module-level function and class name; callers "
    "must keep working unchanged.\n"
    "2. Never change what any function returns, raises, or yields for the "
    "same inputs.\n"
    "3. Prefer small cohesive helpers, guard clauses, and fresh-container "
    "defaults over clever rewrites.\n"
    "4. Do not add new dependencies, network calls, or file access.\n"
    "5. Output the COMPLETE refactored module, never a fragment, never an "
    "explanation before the code."
)

TASK_INSTRUCTION = (
    "Refactor the following Python module so the detected issues are resolved "
    "while its observable behaviour is preserved."
)

OUTPUT_INSTRUCTION = (
    "Return only the complete refactored Python code. "
    "Do not include Markdown fences or explanations."
)


def format_findings(findings: list[ASTFinding] | None) -> str:
    """Render findings exactly as the provider and the training data do."""
    return "\n".join(f"- {finding.rule}: {finding.message}" for finding in findings or [])


def build_user_prompt(
    code: str,
    findings: list[ASTFinding] | None = None,
    analysis_plan: dict[str, Any] | None = None,
    review_critique: str | None = None,
) -> str:
    """Build the user turn used by both training and inference.

    ``analysis_plan`` and ``review_critique`` are only added for interactive
    retries; when both are ``None`` the output is byte-identical to the
    training prompt (the canonical distribution).
    """
    parts = [TASK_INSTRUCTION]
    rendered_findings = format_findings(findings)
    if rendered_findings:
        parts.append(f"Detected issues:\n{rendered_findings}")
    if isinstance(analysis_plan, dict) and analysis_plan.get("step_by_step_plan"):
        plan_parts = ["Analysis plan:"]
        strategy = analysis_plan.get("refactoring_strategy")
        if strategy:
            plan_parts.append(f"Strategy: {strategy}")
        plan_parts.extend(f"- {step}" for step in analysis_plan["step_by_step_plan"])
        parts.append("\n".join(plan_parts))
    if review_critique:
        parts.append(
            "Previous attempt was rejected by the Review Agent:\n"
            f"{review_critique}\n"
            "Fix every point above. Do not repeat the same mistake."
        )
    parts.append(f"```python\n{code.rstrip()}\n```")
    parts.append(OUTPUT_INSTRUCTION)
    return "\n\n".join(parts)


def build_messages(
    code: str,
    findings: list[ASTFinding] | None = None,
    system_prompt: str | None = None,
    analysis_plan: dict[str, Any] | None = None,
    review_critique: str | None = None,
) -> list[dict[str, str]]:
    """Return the chat messages used for training and inference."""
    return [
        {"role": "system", "content": system_prompt or REFACTORING_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_user_prompt(code, findings, analysis_plan, review_critique),
        },
    ]


def _plain_prompt(messages: list[dict[str, str]]) -> str:
    """Fallback for tokenizers without a chat template."""
    system = next((message["content"] for message in messages if message["role"] == "system"), "")
    user = next((message["content"] for message in messages if message["role"] == "user"), "")
    return f"{system}\n\n{user}\n\n"


def render_prompt(
    tokenizer: Any,
    code: str,
    findings: list[ASTFinding] | None = None,
    system_prompt: str | None = None,
    analysis_plan: dict[str, Any] | None = None,
    review_critique: str | None = None,
) -> str:
    """Render the exact prefix that precedes the completion.

    ``add_generation_prompt=True`` makes the chat template end with the
    assistant header. Training tokenizes this string with
    ``add_special_tokens=False`` so the template's own control tokens are not
    duplicated.
    """
    messages = build_messages(code, findings, system_prompt, analysis_plan, review_critique)
    apply_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_template):
        try:
            return str(
                apply_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        except (AttributeError, TypeError, ValueError):
            pass
    return _plain_prompt(messages)


def assistant_end_token(tokenizer: Any) -> tuple[str, int]:
    """Return the end-of-turn token used after a completion.

    Qwen chat models stop on ``<|im_end|>``; using ``eos_token`` alone leaves
    the model generating past the end of the assistant turn. Fall back to the
    tokenizer EOS token for non-chat checkpoints.
    """
    candidates = ("<|im_end|>", "<|eot_id|>", "</s>")
    for token in candidates:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is not None and token_id != getattr(tokenizer, "unk_token_id", None):
            return token, int(token_id)
    eos_token = getattr(tokenizer, "eos_token", None) or ""
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is None:
        raise ValueError("Tokenizer must define an end-of-turn or EOS token")
    return str(eos_token), int(eos_id)


def strip_generation(text: str) -> str:
    """Cut chat/stop markers and keep only the first code block payload."""
    for marker in ("<|im_end|>", "<|eot_id|>", "<|endoftext|>"):
        if marker in text:
            text = text.split(marker, 1)[0]

    if "```python" in text:
        text = text.split("```python", 1)[1].split("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0]

    return text.strip()


def extract_refactored_code(text: str) -> str:
    """Extract the candidate module from a model response.

    The function is intentionally deterministic and never executes generated
    code (NFR01); it only removes fences, prose and chat markers.
    """
    cleaned = strip_generation(text)
    lines = cleaned.splitlines()
    while lines and lines[0].strip().lower().startswith(
        ("here is", "refactored code", "output:", "python:")
    ):
        lines.pop(0)
    return "\n".join(lines).strip() + ("\n" if cleaned.strip() else "")
