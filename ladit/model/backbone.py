"""
Unified backbone loader for masked-diffusion MT.

Provides a single entry point `load_backbone(name, path, ...)` that returns
`(model, tokenizer, meta)` where `meta` is a dict capturing all the
backbone-specific quirks the training / decoding pipelines need to care
about. The goal is that the rest of the code stays backbone-agnostic and
just reads fields from `meta`.

Supported backbones
-------------------
- "llada"       : GSAI-ML/LLaDA-8B-Base          (default, unchanged)
- "dream"       : Dream-org/Dream-v0-Base-7B     (Qwen2.5-init, base only)
- "diffullama"  : diffusionfamily/diffullama     (Llama-2 + MDLM CPT)
- "auto"        : detect from the model config.json's `model_type`

`meta` dict schema (read by callers)
-----------------------------------
{
    "name":                 one of {"llada","dream","diffullama"},
    "mask_token_id":        int,    # token id used for masking
    "lora_target_modules":  list of module suffix strings for PEFT LoRA,
    "template_family":      "plain" or "qwen",   # which prompt bank to use
    "forward_style":        "logits_only",       # how to read logits
                                                 # currently all three
                                                 # backbones expose
                                                 # outputs.logits of shape
                                                 # (B,L,V)
    "logit_shift":          int,    # 0 = logits[i] predicts token at i
                                    # 1 = logits[i] predicts token at i+1
                                    #     (DiffuLLaMA's AR-style shift)
    "needs_attention_patch": bool,  # DiffuLLaMA needs monkey-patched
                                    # bidirectional attention
    "notes":                str,    # free-form notes for logging
}
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Public constants — keep callers from sprinkling magic ids around.
# ---------------------------------------------------------------------------

LLADA_MASK_TOKEN_ID = 126336
# Fallback only: the canonical id is read from the tokenizer's mask_token at
# runtime (HF tokenizer_config.json maps "ти" -> 811 for DiffuLLaMA).
DIFFULLAMA_MASK_TOKEN_ID_FALLBACK = 811

# LoRA target sets, per architecture family.
LORA_TARGETS_LLADA = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "ff_proj", "up_proj", "ff_out",
]
LORA_TARGETS_QWEN = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
# Llama-2 uses the same naming convention as Qwen2 for these modules.
LORA_TARGETS_LLAMA = LORA_TARGETS_QWEN


# ---------------------------------------------------------------------------
# Autodetect from a model directory's config.json.
# ---------------------------------------------------------------------------

def _read_model_type(path: str) -> str:
    """Return (model_type lowercased, first architecture name) best-effort."""
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
    mt = (getattr(cfg, "model_type", "") or "").lower()
    return mt, cfg


def detect_backbone(path: str) -> str:
    """Auto-detect backbone name from the config.json at `path`.

    Recognised signatures:
      - model_type == "Dream"  (Dream base/instruct)
      - model_type == "llada" OR architectures include "LLaDA*"
      - DiffuLLaMA is just model_type == "llama" -> we cannot disambiguate
        plain llama from diffullama by config alone. When model_type is
        "llama" we default to "diffullama" when the tokenizer has a
        mask_token (the signature diffullama adds at tokenizer level).
    """
    mt, cfg = _read_model_type(path)
    if mt == "dream":
        return "dream"
    if mt == "llada":
        return "llada"
    archs = getattr(cfg, "architectures", None) or []
    archs_str = " ".join(archs).lower()
    if "llada" in archs_str:
        return "llada"
    if mt == "llama":
        # Try to disambiguate via tokenizer: diffullama's tokenizer has
        # mask_token set; a vanilla Llama-2 tokenizer does not.
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
            if getattr(tok, "mask_token_id", None) is not None:
                return "diffullama"
        except Exception:
            pass
        # If we cannot determine, bail out loudly.
        raise RuntimeError(
            f"Backbone at {path} has model_type=llama but no mask_token "
            "in its tokenizer. Refusing to autodetect — please pass "
            "--backbone diffullama explicitly."
        )
    raise RuntimeError(f"Cannot autodetect backbone for model_type={mt!r} at {path}")


# ---------------------------------------------------------------------------
# The actual loader.
# ---------------------------------------------------------------------------

def _apply_diffullama_attention_patch() -> None:
    """Monkey-patch transformers.Llama* to accept 4D bidirectional attention masks.

    DiffuLLaMA ships a `replace_attention_mask()` helper in its repo.
    We inline a minimal equivalent here so the training/decoding code doesn't
    need to import from the DiffuLLaMA repo (which isn't a proper package).

    transformers 4.46+ changed the causal-mask plumbing: there's no
    ``_update_causal_mask`` method anymore on LlamaModel. Instead causal
    masking is enforced at the attention-layer level (``self.is_causal``).
    To get bidirectional attention we monkey-patch ``LlamaModel.forward``
    so that, when a 4D attention_mask is supplied, every LlamaAttention
    submodule has ``is_causal`` temporarily flipped to ``False`` for the
    duration of the forward call. This works across transformers >= 4.40.
    """
    try:
        import transformers  # noqa: F401
        from transformers.models.llama import modeling_llama  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "transformers LLaMA modeling module unavailable; cannot patch "
            f"attention for DiffuLLaMA: {e}"
        )

    if getattr(modeling_llama.LlamaModel, "_bidit_patched", False):
        return  # idempotent

    _orig_forward = modeling_llama.LlamaModel.forward

    def _patched_forward(self, *args, **kwargs):
        attn = kwargs.get("attention_mask", None)
        # If caller provided a 4D mask, flip is_causal=False on every
        # attention submodule so the attention computation becomes
        # bidirectional, then restore on exit.
        if attn is not None and hasattr(attn, "dim") and attn.dim() == 4:
            attn_modules = []
            for layer in getattr(self, "layers", []):
                am = getattr(layer, "self_attn", None)
                if am is not None and getattr(am, "is_causal", False):
                    attn_modules.append(am)
            for am in attn_modules:
                am.is_causal = False
            try:
                return _orig_forward(self, *args, **kwargs)
            finally:
                for am in attn_modules:
                    am.is_causal = True
        return _orig_forward(self, *args, **kwargs)

    modeling_llama.LlamaModel.forward = _patched_forward
    modeling_llama.LlamaModel._bidit_patched = True


def _build_diffullama_4d_attention_mask(
    input_ids: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build a 4D bidirectional attention mask (all positions see all)."""
    B, L = input_ids.shape
    # All-zero bias = fully attend to every position (bidirectional).
    # Shape matches what `_update_causal_mask` would normally return for
    # LlamaSdpaAttention / LlamaAttention: (B, 1, L, L).
    mask = torch.zeros(B, 1, L, L, dtype=dtype, device=input_ids.device)
    return mask


def load_backbone(
    name: str,
    path: str,
    dtype: torch.dtype = torch.bfloat16,
    device_map: Optional[str] = None,
    attn_implementation: Optional[str] = None,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """Load a masked-diffusion backbone with its tokenizer and meta.

    Args
    ----
    name : {"llada","dream","diffullama","auto"}
    path : local directory or HF repo id
    dtype : torch dtype for model weights
    device_map : optional HF device_map (e.g. "auto")
    attn_implementation : "eager" / "sdpa" / "flash_attention_2" or None

    Returns
    -------
    (model, tokenizer, meta)
    """
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

    if name == "auto":
        name = detect_backbone(path)
    name = name.lower()
    if name not in {"llada", "dream", "diffullama"}:
        raise ValueError(f"Unknown backbone {name!r}")

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    common_kwargs = dict(trust_remote_code=True, torch_dtype=dtype)
    if device_map is not None:
        common_kwargs["device_map"] = device_map
    if attn_implementation is not None:
        common_kwargs["_attn_implementation"] = attn_implementation

    if name == "dream":
        # Dream registers only AutoModel -> DreamModel in its auto_map.
        cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
        mask_id = int(getattr(cfg, "mask_token_id", 0)) or 0
        if mask_id == 0:
            raise RuntimeError(f"Dream config at {path} missing mask_token_id")
        model = AutoModel.from_pretrained(path, **common_kwargs)
        meta = {
            "name": "dream",
            "mask_token_id": mask_id,
            "lora_target_modules": LORA_TARGETS_QWEN,
            "template_family": "plain",  # base model: plain prompts still work
            "forward_style": "logits_only",
            "logit_shift": 0,
            "needs_attention_patch": False,
            "notes": "Dream-v0-Base-7B: Qwen2.5-init, mask_token_id from config.",
        }
        return model, tokenizer, meta

    if name == "diffullama":
        # DiffuLLaMA is packaged as a vanilla LlamaForCausalLM checkpoint
        # plus a monkey patch that makes the LlamaModel.forward accept a 4D
        # bidirectional attention mask.
        _apply_diffullama_attention_patch()
        mask_id = getattr(tokenizer, "mask_token_id", None)
        if mask_id is None:
            # tokenizer_config.json should set mask_token="ти"; fall back
            # to the known id so training still works even if the tokenizer
            # object did not pick it up.
            mask_id = DIFFULLAMA_MASK_TOKEN_ID_FALLBACK
        model = AutoModelForCausalLM.from_pretrained(path, **common_kwargs)
        meta = {
            "name": "diffullama",
            "mask_token_id": int(mask_id),
            "lora_target_modules": LORA_TARGETS_LLAMA,
            "template_family": "plain",
            "forward_style": "logits_only",
            # DiffuLLaMA training adapts Llama-2 AR; inference script uses
            # shift=True: argmax(logits[i-1]) -> token at position i. Keep
            # this as 1 so the decoder can slice logits accordingly.
            "logit_shift": 1,
            "needs_attention_patch": True,
            "notes": (
                "DiffuLLaMA: Llama-2 CPT w/ MDLM. Requires 4D bidirectional "
                "attn mask (patched) and shift-by-1 logit indexing for decode."
            ),
        }
        return model, tokenizer, meta

    # --- default: LLaDA (existing path, unchanged behaviour) -----------
    model = AutoModelForCausalLM.from_pretrained(path, **common_kwargs)
    meta = {
        "name": "llada",
        "mask_token_id": LLADA_MASK_TOKEN_ID,
        "lora_target_modules": LORA_TARGETS_LLADA,
        "template_family": "plain",
        "forward_style": "logits_only",
        "logit_shift": 0,
        "needs_attention_patch": False,
        "notes": "LLaDA-8B-Base: native masked-diffusion pretrain.",
    }
    return model, tokenizer, meta


# ---------------------------------------------------------------------------
# Small helper: produce an attention_mask tensor that the backbone expects.
# Callers that just pass `torch.ones_like(input_ids)` keep working for
# llada / dream; for diffullama we upgrade to a 4D bidirectional mask.
# ---------------------------------------------------------------------------

def build_attention_mask(
    input_ids: torch.Tensor,
    meta: Dict[str, Any],
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Build the attention mask shape the backbone expects.

    - ``llada``: 2D ``(B, L)`` bool mask. LLaDA's modeling internally
      converts to whatever its attention layer needs.
    - ``dream``: 4D ``(B, 1, 1, L)`` bool mask. Dream's modeling_dream.py
      passes the mask straight to SDPA without expanding 2D -> 4D, so PyTorch
      broadcast fails with a bare (B, L) tensor at batch > 1. (B, 1, 1, L)
      broadcasts cleanly to the (B, H, L, L) QK shape.
    - ``diffullama``: 4D ``(B, 1, L, L)`` all-attend mask, the monkey-patched
      Llama forward keeps it as the bidirectional attention pattern.

    Dtype: returned as bool for llada/dream (SDPA-friendly), as the model's
    compute dtype for diffullama (the patched forward expects an additive
    mask; zeros mean "fully attend").
    """
    if meta.get("name") == "diffullama":
        return _build_diffullama_4d_attention_mask(input_ids, dtype=dtype)
    if meta.get("name") == "dream":
        # (B, L) bool -> (B, 1, 1, L). All positions = True for all-mask
        # canvases, which is what callers in length_adaptive use.
        attn_2d = torch.ones_like(input_ids, dtype=torch.bool)
        return attn_2d[:, None, None, :]
    return torch.ones_like(input_ids, dtype=torch.bool)


__all__ = [
    "load_backbone",
    "detect_backbone",
    "build_attention_mask",
    "LORA_TARGETS_LLADA",
    "LORA_TARGETS_QWEN",
    "LORA_TARGETS_LLAMA",
    "LLADA_MASK_TOKEN_ID",
    "DIFFULLAMA_MASK_TOKEN_ID_FALLBACK",
]
