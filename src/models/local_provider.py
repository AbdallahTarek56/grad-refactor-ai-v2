"""Local open-weight model provider with tiered resolution (Phase 2, ADR08).

Resolution order for ``generate_refactoring``:

    1. HuggingFace transformers pipeline, optionally with a PEFT/LoRA adapter
       (used when the local ML stack is installed).
    2. Local Ollama daemon (when ``prefer_ollama=True`` and the daemon is live).
    3. SmartASTProvider deterministic AST transformer (always available).

An explicitly requested LoRA adapter that cannot be loaded raises
``ProviderUnavailableError`` instead of silently degrading — the review
protocol must never present an untrained model as a fine-tuned one (NFR06).
"""

import asyncio
import importlib.util
import json
from pathlib import Path
from typing import Any

from core.candidate_selection import select_best_candidate
from core.exceptions import ExecutionTimeoutError, ProviderUnavailableError
from core.schemas import ASTFinding
from models.base import BaseModelProvider
from models.explanations import explain_refactoring
from models.ollama_provider import DEFAULT_OLLAMA_MODEL, DEFAULT_OLLAMA_URL, OllamaModelProvider
from models.prompting import assistant_end_token, extract_refactored_code, render_prompt
from models.smart_ast_provider import SmartASTProvider

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
LEGACY_PLACEHOLDER_MODEL = "fine-tuned-lora-adapter"
_TOKENIZER_FILES = ("tokenizer_config.json", "tokenizer.json", "vocab.json")


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def resolve_adapter_base_model(adapter_path: str | Path | None) -> str | None:
    """Read ``base_model_name_or_path`` from a PEFT adapter directory."""
    if not adapter_path:
        return None
    config_path = Path(adapter_path) / "adapter_config.json"
    if not config_path.exists():
        return None
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    base = payload.get("base_model_name_or_path")
    return str(base) if base else None


def hf_stack_status() -> tuple[bool, str]:
    """Report whether local transformers/PEFT inference can run here."""
    missing = [name for name in ("torch", "transformers") if not _module_available(name)]
    if missing:
        return (
            False,
            f"Missing local ML packages: {', '.join(missing)}. "
            "Install training/requirements.txt to enable local HF/LoRA inference.",
        )
    if not _module_available("peft"):
        return (
            True,
            "transformers stack available; PEFT missing so LoRA adapters cannot be loaded.",
        )
    return True, "transformers + PEFT available for local base-model and LoRA inference."


def adapter_status(adapter_path: str | Path | None) -> tuple[bool, str]:
    """Report whether a PEFT adapter directory can actually be served.

    Availability must reflect the adapter, not only the Python stack: a
    ``local_lora`` run without its weights would otherwise be presented as a
    fine-tuned result (NFR06).
    """
    if not adapter_path:
        return False, "No adapter path configured. Set REFACTOR_AI_ADAPTER_PATH."
    directory = Path(adapter_path)
    if not directory.is_dir():
        return (
            False,
            f"Adapter directory '{directory}' not found. Set REFACTOR_AI_ADAPTER_PATH "
            "to the downloaded PEFT adapter.",
        )
    if not (directory / "adapter_config.json").is_file():
        return (
            False,
            f"Adapter directory '{directory}' lacks adapter_config.json.",
        )
    if not any(
        (directory / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")
    ):
        return (
            False,
            f"Adapter directory '{directory}' lacks adapter weights "
            "(adapter_model.safetensors or adapter_model.bin).",
        )
    if not _module_available("peft"):
        return (
            False,
            "PEFT is not installed; install training/requirements.txt to serve the adapter.",
        )
    base_model = resolve_adapter_base_model(directory)
    suffix = f" base model '{base_model}'" if base_model else ""
    return True, f"Adapter '{directory}'{suffix} ready for local inference."


class LocalModelProvider(BaseModelProvider):
    """Local inference engine with HuggingFace -> Ollama -> SmartAST resolution."""

    def __init__(
        self,
        model_name: str = DEFAULT_BASE_MODEL,
        adapter_path: str | None = None,
        base_model_name: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        timeout_seconds: float = 30.0,
        prefer_ollama: bool = False,
        ollama_url: str = DEFAULT_OLLAMA_URL,
        ollama_model: str = DEFAULT_OLLAMA_MODEL,
        load_in_4bit: bool = False,
        require_hf: bool = False,
        num_candidates: int = 1,
        candidate_temperature: float = 0.7,
        candidate_top_p: float = 0.9,
    ) -> None:
        self.model_name = model_name
        self.adapter_path = adapter_path
        self.base_model_name = base_model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self.prefer_ollama = prefer_ollama
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model
        self.load_in_4bit = load_in_4bit
        self.require_hf = require_hf
        self.num_candidates = max(1, int(num_candidates))
        self.candidate_temperature = candidate_temperature
        self.candidate_top_p = candidate_top_p
        self._pipeline = None
        self._tokenizer = None
        self._hf_attempted = False
        self._hf_error: str | None = None
        self._resolved_model_name: str | None = None
        self._ollama_provider: OllamaModelProvider | None = None
        self._ollama_probed = False

    # ------------------------------------------------------------------ #
    # Tier 1: HuggingFace (optionally + PEFT LoRA adapter)
    # ------------------------------------------------------------------ #
    def _resolve_base_model(self) -> str:
        """Resolve the real base checkpoint behind an adapter."""
        if self._resolved_model_name:
            return self._resolved_model_name

        if self.base_model_name:
            resolved = self.base_model_name
        elif self.adapter_path:
            resolved = resolve_adapter_base_model(self.adapter_path) or ""
            if not resolved:
                raise ProviderUnavailableError(
                    f"Adapter '{self.adapter_path}' has no readable adapter_config.json "
                    "with base_model_name_or_path. Pass base_model_name explicitly."
                )
        else:
            resolved = self.model_name

        if resolved == LEGACY_PLACEHOLDER_MODEL:
            raise ProviderUnavailableError(
                "A real base model ID is required; 'fine-tuned-lora-adapter' is a "
                "placeholder, not a HuggingFace repository. Pass --base-model or "
                "resolved base_model_name."
            )
        self._resolved_model_name = resolved
        return resolved

    def _tokenizer_source(self) -> str:
        if self.adapter_path:
            adapter_dir = Path(self.adapter_path)
            if any((adapter_dir / name).exists() for name in _TOKENIZER_FILES):
                return self.adapter_path
        return self._resolve_base_model()

    def _load_hf_pipeline(self) -> Any | None:
        if self._hf_attempted:
            return self._pipeline
        self._hf_attempted = True

        if not _module_available("torch") or not _module_available("transformers"):
            self._hf_error = "torch/transformers are not installed in this environment"
            return None

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

            base_model = self._resolve_base_model()
            tokenizer = AutoTokenizer.from_pretrained(
                self._tokenizer_source(), trust_remote_code=False
            )
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id

            model_kwargs: dict[str, Any] = {"trust_remote_code": False}
            if self.load_in_4bit:
                try:
                    from transformers import BitsAndBytesConfig
                except ImportError as exc:  # pragma: no cover - depends on optional stack
                    raise ProviderUnavailableError(
                        "4-bit inference requested but transformers does not expose "
                        "BitsAndBytesConfig; install bitsandbytes."
                    ) from exc
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
                model_kwargs["device_map"] = "auto"
            else:
                # A full-precision checkpoint must be placed on the GPU up front.
                # Loading it on CPU and relying on the pipeline's `device=` move
                # doubles peak RAM and can silently leave generation on CPU.
                if torch.cuda.is_available():
                    model_kwargs["device_map"] = "auto"
                model_kwargs["torch_dtype"] = (
                    torch.float16 if torch.cuda.is_available() else torch.float32
                )

            model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
            if self.adapter_path:
                from peft import PeftModel

                model = PeftModel.from_pretrained(model, self.adapter_path)

            pipeline_device = None
            if not self.load_in_4bit and not torch.cuda.is_available():
                pipeline_device = "cpu"
            self._tokenizer = tokenizer
            self._pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                top_p=0.95,
                device=pipeline_device,
            )
        except Exception as exc:  # noqa: BLE001 - degrade to the next tier
            self._pipeline = None
            self._tokenizer = None
            self._hf_error = f"{type(exc).__name__}: {exc}"
        return self._pipeline

    async def _generate_hf(
        self, code: str, findings: list[ASTFinding], options: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        if self._pipeline is None or self._tokenizer is None:
            raise ProviderUnavailableError(
                f"Local HuggingFace model could not be loaded ({self._hf_error})."
            )

        system_prompt = options.get("system_prompt")
        plan = options.get("analysis_plan")
        if not isinstance(plan, dict):
            plan = None
        critique = options.get("review_critique") or options.get("retry_feedback")
        prompt = render_prompt(
            self._tokenizer,
            code,
            findings,
            system_prompt,
            analysis_plan=plan,
            review_critique=str(critique) if critique else None,
        )
        candidates_requested = max(1, int(options.get("num_candidates", self.num_candidates)))
        candidate_temperature = float(
            options.get("candidate_temperature", self.candidate_temperature)
        )
        candidate_top_p = float(options.get("candidate_top_p", self.candidate_top_p))
        max_tokens = int(options.get("max_tokens", self.max_tokens))
        try:
            eos_token_id = assistant_end_token(self._tokenizer)[1]
        except ValueError:
            eos_token_id = None

        candidates: list[str] = []
        for _ in range(candidates_requested):
            generation_options: dict[str, Any] = {
                "max_new_tokens": max_tokens,
                "temperature": candidate_temperature,
                "do_sample": candidate_temperature > 0,
                "top_p": candidate_top_p,
                "num_return_sequences": 1,
                "return_full_text": False,
            }
            if eos_token_id is not None:
                generation_options["eos_token_id"] = eos_token_id
            outputs = await asyncio.to_thread(
                self._pipeline,
                prompt,
                **generation_options,
            )
            generated_text = outputs[0]["generated_text"]
            # Defensive fallback for transformers versions that ignore
            # return_full_text: remove the prompt if the pipeline echoed it.
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt) :]
            candidates.append(extract_refactored_code(generated_text))

        best, selected_index, candidate_scores = select_best_candidate(code, candidates, findings)

        prompt_ids = self._tokenizer(prompt, add_special_tokens=False)["input_ids"]
        completion_ids = self._tokenizer(best, add_special_tokens=False)["input_ids"]
        metadata: dict[str, Any] = {
            "model": self._resolved_model_name or self.model_name,
            "adapter": self.adapter_path,
            "provider": "local-hf",
            "engine": "huggingface",
            "resolution": "hf-tier",
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": len(completion_ids),
            "total_tokens": len(prompt_ids) + len(completion_ids),
            "candidates_generated": len(candidates),
            "selected_candidate": selected_index,
            "candidate_scores": candidate_scores,
            "num_candidates": candidates_requested,
            "quantized_4bit": self.load_in_4bit,
        }
        return best, metadata

    # ------------------------------------------------------------------ #
    # Tier 2: Ollama daemon
    # ------------------------------------------------------------------ #
    def _ollama_tier(self) -> OllamaModelProvider | None:
        if not self.prefer_ollama:
            return None
        if not self._ollama_probed:
            self._ollama_probed = True
            available, _, _ = OllamaModelProvider.probe(self.ollama_url, timeout=1.0)
            if available:
                self._ollama_provider = OllamaModelProvider(
                    model=self.ollama_model,
                    base_url=self.ollama_url,
                    temperature=self.temperature,
                    max_tokens=max(self.max_tokens, 512),
                    timeout_seconds=max(self.timeout_seconds, 60.0),
                )
        return self._ollama_provider

    # ------------------------------------------------------------------ #
    # Tier 3: deterministic SmartAST transformer
    # ------------------------------------------------------------------ #
    async def _generate_smart_ast(
        self, code: str, findings: list[ASTFinding], options: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        fallback = SmartASTProvider()
        code_out, meta = await fallback.generate_refactoring(code, findings, options)
        meta["model"] = f"{self._resolved_model_name or self.model_name} (CPU-fallback)"
        meta["provider"] = "local-cpu-fallback"
        meta["engine"] = "smart_ast"
        meta["resolution"] = "smart-ast-tier"
        return code_out, meta

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    async def generate_refactoring(
        self,
        code: str,
        findings: list[ASTFinding],
        options: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        options = options or {}
        timeout = float(options.get("timeout", self.timeout_seconds))

        try:
            return await asyncio.wait_for(
                self._generate_internal(code, findings, options),
                timeout=timeout,
            )
        except TimeoutError as exc:
            raise ExecutionTimeoutError(
                f"Local model generation exceeded timeout of {timeout}s (FR08)"
            ) from exc

    async def _generate_internal(
        self,
        code: str,
        findings: list[ASTFinding],
        options: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        # An explicitly requested fine-tuned adapter must actually load.
        if self.adapter_path:
            pipeline = self._load_hf_pipeline()
            if pipeline is None:
                raise ProviderUnavailableError(
                    f"Fine-tuned LoRA adapter '{self.adapter_path}' was requested but "
                    f"could not be loaded ({self._hf_error}). Install training/requirements.txt "
                    "and train the adapter, or select a different provider."
                )
            return await self._generate_hf(code, findings, options)

        pipeline = self._load_hf_pipeline()
        if pipeline is not None:
            return await self._generate_hf(code, findings, options)

        if self.require_hf:
            raise ProviderUnavailableError(
                f"Base model '{self._resolve_base_model()}' could not be loaded ({self._hf_error})."
            )

        ollama = self._ollama_tier()
        if ollama is not None:
            code_out, meta = await ollama.generate_refactoring(code, findings, options)
            meta["provider"] = "ollama"
            meta["resolution"] = "ollama-tier"
            return code_out, meta

        return await self._generate_smart_ast(code, findings, options)

    async def explain_changes(
        self,
        original_code: str,
        refactored_code: str,
        diff: str,
    ) -> str:
        tier = "HuggingFace pipeline" if self._pipeline is not None else "local fallback chain"
        return explain_refactoring(
            f"Local provider ({self._resolved_model_name or self.model_name}, {tier})",
            original_code,
            refactored_code,
        )
