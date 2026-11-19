"""Tests for the shared prompt contract and AST-only candidate selection."""

from core.candidate_selection import score_candidate, select_best_candidate
from core.schemas import ASTFinding
from models.ollama_provider import OllamaModelProvider, extract_python_code
from models.prompting import (
    REFACTORING_SYSTEM_PROMPT,
    assistant_end_token,
    build_user_prompt,
    extract_refactored_code,
    render_prompt,
)


def finding(rule: str = "deep-nesting") -> ASTFinding:
    return ASTFinding(
        rule=rule,
        line=1,
        end_line=4,
        node_type="If",
        message="Nested conditionals should be flattened.",
        rationale="Lower nesting improves readability.",
    )


def test_user_prompt_carries_findings_and_code_without_training_markers():
    code = "def f(x):\n    return x\n"
    prompt = build_user_prompt(code, [finding()])
    assert "- deep-nesting: Nested conditionals should be flattened." in prompt
    assert "```python\ndef f(x):" in prompt
    assert "# example:" not in prompt
    assert REFACTORING_SYSTEM_PROMPT.startswith("You are the Refactoring Agent")


def test_user_prompt_optional_context_is_absent_by_default():
    code = "def f(x):\n    return x\n"
    assert build_user_prompt(code, [finding()]) == build_user_prompt(
        code, [finding()], analysis_plan=None, review_critique=None
    )
    prompt = build_user_prompt(code, [finding()])
    assert "Analysis plan" not in prompt
    assert "Previous attempt" not in prompt


def test_user_prompt_adds_plan_and_review_critique_only_on_retry():
    prompt = build_user_prompt(
        "def f(x):\n    return x\n",
        [finding()],
        analysis_plan={
            "refactoring_strategy": "Flatten with guard clauses",
            "step_by_step_plan": ["Flatten the nested ifs", "Keep the signature"],
        },
        review_critique="Tests fail on empty input.",
    )
    assert "Analysis plan:" in prompt
    assert "Strategy: Flatten with guard clauses" in prompt
    assert "- Flatten the nested ifs" in prompt
    assert "Previous attempt was rejected by the Review Agent:" in prompt
    assert "Tests fail on empty input." in prompt


def test_ollama_prompt_is_the_canonical_training_prompt():
    code = "def f(x):\n    return x\n"
    provider = OllamaModelProvider()
    prompt = provider.build_prompt(code, [finding()], {})
    assert prompt == f"{REFACTORING_SYSTEM_PROMPT}\n\n{build_user_prompt(code, [finding()])}"
    assert "## Detected issues" not in prompt
    assert "Return the complete refactored module" not in prompt
    assert "- deep-nesting: Nested conditionals should be flattened." in prompt


def test_ollama_prompt_forwards_plan_and_critique():
    prompt = OllamaModelProvider().build_prompt(
        "def f(x):\n    return x\n",
        [finding()],
        {
            "analysis_plan": {
                "refactoring_strategy": "Guard clauses",
                "step_by_step_plan": ["Flatten"],
            },
            "review_critique": "Signature changed.",
        },
    )
    assert "Strategy: Guard clauses" in prompt
    assert "Previous attempt was rejected by the Review Agent:" in prompt
    assert "Signature changed." in prompt


def test_render_prompt_forwards_optional_plan_and_critique_to_template():
    captured: dict[str, object] = {}

    class Tokenizer:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            captured["messages"] = messages
            captured["add_generation_prompt"] = add_generation_prompt
            return "PROMPT"

    rendered = render_prompt(
        Tokenizer(),
        "def f(x):\n    return x\n",
        [finding()],
        analysis_plan={"step_by_step_plan": ["Flatten"]},
        review_critique="Tests fail.",
    )
    assert rendered == "PROMPT"
    assert captured["add_generation_prompt"] is True
    user_content = captured["messages"][1]["content"]
    assert "Analysis plan:" in user_content
    assert "Tests fail." in user_content


def test_ollama_extract_alias_uses_canonical_extractor():
    raw = "Here is the refactored code:\n```python\ndef f(x):\n    return x\n```\n<|im_end|>"
    assert extract_python_code(raw) == extract_refactored_code(raw)


def test_extract_refactored_code_removes_fences_and_chat_markers():
    raw = "Here is the refactored code:\n```python\ndef f(x):\n    return x + 1\n```\n<|im_end|>"
    assert extract_refactored_code(raw) == "def f(x):\n    return x + 1\n"


def test_assistant_end_token_prefers_chat_turn_marker():
    class Tokenizer:
        unk_token_id = 0
        eos_token = "<eos>"
        eos_token_id = 2

        def convert_tokens_to_ids(self, token):
            return {"<|im_end|>": 7, "<|eot_id|>": 0, "</s>": 0}[token]

    assert assistant_end_token(Tokenizer()) == ("<|im_end|>", 7)


DEEPLY_NESTED = (
    "def process(items):\n"
    "    if items:\n"
    "        if len(items) > 1:\n"
    "            if items[0]:\n"
    "                if items[0] > 0:\n"
    "                    return items[0]\n"
    "    return 0\n"
)

FLATTENED = (
    "def process(items):\n"
    "    if items and len(items) > 1 and items[0] and items[0] > 0:\n"
    "        return items[0]\n"
    "    return 0\n"
)


def test_candidate_scoring_prefers_a_resolving_rewrite():
    scores = score_candidate(DEEPLY_NESTED, FLATTENED, [finding()])
    assert scores["syntax_valid"] is True
    assert scores["preserved_symbols"] is True
    assert scores["resolved_rules"] == ["deep-nesting"]
    assert scores["score"] > 0


def test_candidate_selection_skips_invalid_and_no_op_candidates():
    invalid = "def process(items):\n    return (\n"
    no_op = DEEPLY_NESTED
    good = FLATTENED
    best, index, scores = select_best_candidate(DEEPLY_NESTED, [invalid, no_op, good], [finding()])
    assert best == good
    assert index == 2
    assert scores[0]["syntax_valid"] is False
    assert scores[1]["resolved_rules"] == []


def test_candidate_selection_rejects_missing_symbols():
    dropped = "def other(items):\n    if items and items[0]:\n        return items[0]\n"
    result = score_candidate(DEEPLY_NESTED, dropped, [finding()])
    assert result["preserved_symbols"] is False
    assert result["missing_symbols"] == ["process"]
