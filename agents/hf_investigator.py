"""
Local-transformers Investigator policy (used for GRPO training).

Same JSON schema, same system prompt, same fallback contract as
:class:`.llm_investigator.LLMInvestigator` — only the chat backend
differs: we call ``model.generate`` on a local ``transformers`` model
(usually 4-bit-quantised + LoRA-wrapped) so TRL's ``GRPOTrainer`` can
backprop through it.

Reuses **everything** from :class:`.base.LLMPolicyBase`:

  * Schema validation + Pydantic error → fallback wiring.
  * ``fallback_count`` / ``call_count`` / ``last_error`` counters.
  * ``last_prompt`` / ``last_completion`` recording slots
    (consumed by :class:`counterfeint.training.rollout.RecordingHFInvestigator`).
  * Markdown-fence stripping in ``_parse_and_validate``.

…and all of :class:`.llm_investigator.LLMInvestigator`:

  * The exact same ``_build_user_prompt`` (so the trained model sees the
    same observation→prompt mapping the production HTTP path uses).
  * The same ``INVESTIGATOR_SYSTEM_PROMPT``.

Only the ``HFInvestigator``-specific bits live here:

  * ``_call_chat`` — local ``model.generate`` instead of HTTP.
  * ``_coerce_payload`` — robust to small Qwen-2.5-1.5B JSON quirks
    (alias keys, ``investigation_token`` etc.) that the production
    HuggingFace-router LLMInvestigator doesn't need because it talks
    to bigger models.
  * ``from_pretrained`` — convenience loader (4-bit + optional LoRA).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..models import AdReviewAction
from ..scripted._base import PolicyBase
from ..scripted.investigator import ScriptedInvestigator
from .base import LLMPolicyBase
from .llm_investigator import LLMInvestigator
from .prompts import INVESTIGATOR_SYSTEM_PROMPT


# Schema-coercion table: small Qwen-1.5B models occasionally invent
# alias key names. We map them back to the canonical AdReviewAction
# field before strict Pydantic validation.
_ALLOWED_KEYS = {
    "action_type", "ad_id",
    "investigation_target",
    "verdict", "confidence", "rationale",
    "linked_ad_id", "link_reason",
}
_ALIAS_MAP = {
    "investigation_rationale": "rationale",
    "investigation_reason": "rationale",
    "investigation_confidence": "confidence",
    "linked_ads_id": "linked_ad_id",
    "link_account_reason": "link_reason",
}
_ALLOWED_TARGETS = {
    "advertiser_history", "landing_page", "payment_method",
    "targeting_overlap", "campaign_structure", "policy_classifier",
}


class HFInvestigator(LLMPolicyBase):
    """Investigator policy backed by a local ``transformers`` model.

    Inherits *all* parse / validate / fallback / counter / recording
    machinery from :class:`LLMPolicyBase` and the prompt-building from
    :class:`LLMInvestigator`. Only the chat backend (local generate)
    and a small schema-coercion shim are implemented here.
    """

    system_prompt = INVESTIGATOR_SYSTEM_PROMPT
    action_model = AdReviewAction
    _log_name = "hf_investigator"

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        fallback_policy: Optional[PolicyBase] = None,
        max_new_tokens: int = 384,
        temperature: float = 0.7,
        do_sample: bool = True,
        enable_thinking: bool = False,
    ) -> None:
        if model is None or tokenizer is None:
            raise TypeError(
                "HFInvestigator requires both `model` and `tokenizer`."
            )
        # We deliberately bypass the OpenAI-client branch in the parent
        # constructor by passing a non-None `client` sentinel; the local
        # backend never touches `self.client`. The HTTP-only fields
        # (api_base_url / api_key / model_name) are kept on the instance
        # purely for diagnostic dumps.
        super().__init__(
            fallback_policy=fallback_policy or ScriptedInvestigator(),
            client=object(),  # sentinel — never used by _call_chat below
        )
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.do_sample = bool(do_sample)
        # Qwen3 supports thinking-mode CoT via the chat template. We default
        # to OFF — we want the model to emit the JSON action directly rather
        # than spend tokens on chain-of-thought before the JSON.  Older
        # models (Qwen2.5 etc.) silently ignore this flag.
        self.enable_thinking = bool(enable_thinking)
        # Exact token IDs from the latest sampled action.  Trajectory GRPO
        # recomputes log-probabilities over these tokens after the complete
        # environment episode has received its terminal reward.
        self.last_prompt_token_ids: Optional[list[int]] = None
        self.last_completion_token_ids: Optional[list[int]] = None

    # ------------------------------------------------------------------
    # Convenience loader
    # ------------------------------------------------------------------
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        load_in_4bit: bool = True,
        torch_dtype: Optional[str] = None,
        lora_path: Optional[str] = None,
        device_map: str = "auto",
        **kwargs: Any,
    ) -> "HFInvestigator":
        """Load ``model_name_or_path`` (optionally + a LoRA adapter)."""
        import torch  # noqa: F401  (kept to fail-fast on missing torch)
        from transformers import (  # type: ignore
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs: Dict[str, Any] = {"device_map": device_map}
        if load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype="bfloat16",
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        else:
            import torch

            dtype = torch.bfloat16 if torch_dtype in (None, "bfloat16", "bf16") else torch.float16
            model_kwargs["torch_dtype"] = dtype
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, **model_kwargs
        )

        if lora_path:
            from peft import PeftModel  # type: ignore
            model = PeftModel.from_pretrained(model, lora_path)

        return cls(model=model, tokenizer=tokenizer, **kwargs)

    # ------------------------------------------------------------------
    # Reused from LLMInvestigator (same observation-to-prompt mapping
    # the production HTTP path uses).
    # ------------------------------------------------------------------
    def _build_user_prompt(self, observation: Dict[str, Any]) -> str:
        return LLMInvestigator._build_user_prompt(self, observation)

    # ------------------------------------------------------------------
    # Local-transformers chat backend (drop-in replacement for the
    # parent's OpenAI-HTTP path).
    # ------------------------------------------------------------------
    def _call_chat(self, messages: list) -> str:
        self.last_prompt_token_ids = None
        self.last_completion_token_ids = None
        template_kwargs: Dict[str, Any] = {
            "add_generation_prompt": True,
            "return_tensors": "pt",
            "return_dict": True,
        }
        # Qwen3-only: pass enable_thinking through so the chat template
        # selects the non-thinking variant. Other tokenizers will reject
        # the unknown kwarg, in which case we silently fall back.
        try:
            encoded = self.tokenizer.apply_chat_template(
                messages,
                enable_thinking=self.enable_thinking,
                **template_kwargs,
            )
        except TypeError:
            encoded = self.tokenizer.apply_chat_template(
                messages, **template_kwargs
            )
        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = getattr(self.model, "device", None)
        if device is not None:
            encoded = {k: v.to(device) for k, v in encoded.items()}

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "do_sample": self.do_sample,
            "pad_token_id": getattr(self.tokenizer, "pad_token_id", None)
            or getattr(self.tokenizer, "eos_token_id", 0),
        }

        outputs = self.model.generate(**encoded, **gen_kwargs)
        prompt_len = encoded["input_ids"].shape[-1]
        gen_tokens = outputs[0][prompt_len:]
        self.last_prompt_token_ids = (
            encoded["input_ids"][0].detach().cpu().tolist()
        )
        self.last_completion_token_ids = gen_tokens.detach().cpu().tolist()
        return self.tokenizer.decode(gen_tokens, skip_special_tokens=True)

    # ------------------------------------------------------------------
    # Robust-JSON shim. Small models occasionally invent alias keys or
    # nest the investigation target inside a non-canonical field; we
    # normalise these BEFORE Pydantic validates so the deterministic
    # fallback only fires on hard failures.
    # ------------------------------------------------------------------
    def _coerce_payload(self, data: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in data.items():
            tgt = _ALIAS_MAP.get(k, k)
            if tgt in _ALLOWED_KEYS and tgt not in out:
                out[tgt] = v

        # Fix: model puts "reject"/"approve"/"escalate" in action_type
        # instead of "verdict". Remap it.
        at = out.get("action_type")
        if at in ("reject", "approve", "escalate"):
            out["verdict"] = at
            out["action_type"] = "verdict"

        # Recover investigation_target from common look-alike fields the
        # base model invents when the schema instruction lands ambiguously.
        if "investigation_target" not in out:
            tok = data.get("investigation_token")
            if isinstance(tok, str) and tok in _ALLOWED_TARGETS:
                out["investigation_target"] = tok
        if "investigation_target" not in out:
            sigs = data.get("investigation_signals")
            if (
                isinstance(sigs, list) and sigs
                and isinstance(sigs[0], str) and sigs[0] in _ALLOWED_TARGETS
            ):
                out["investigation_target"] = sigs[0]

        # Truncate rationale to avoid max_tokens truncation mid-string.
        rat = out.get("rationale")
        if isinstance(rat, str) and len(rat) > 100:
            out["rationale"] = rat[:100].rsplit(" ", 1)[0]

        return out


__all__ = ["HFInvestigator"]
