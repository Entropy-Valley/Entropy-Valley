"""
Main translation pipeline for masked diffusion MT.

Implements iterative unmasking with pluggable order schedules.
Supports all schedule types: Random, L2R, MED, SIG-first, Reverse-SIG, Oracle-anchor.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ladit.decoding.schedules import get_schedule, compute_unmask_count
from ladit.decoding.sig import compute_sig_scores, compute_sig_concentration, select_distractor
from ladit.data.mt_dataset import PROMPT_PREFIX, get_prompt_prefix


# Default is LLaDA; overridden in main() after loading the model config.
MASK_TOKEN_ID = 126336
# Logit shift applied when indexing into model.forward outputs. 0 = none
# (LLaDA / Dream), 1 = DiffuLLaMA-style AR shift (logits[i-1] -> token_i).
LOGIT_SHIFT = 0
# Set by main() after loading a backbone — used to decide whether to build
# a 4D bidirectional attention mask (DiffuLLaMA) instead of ones_like.
BACKBONE_META = None


def _detect_mask_token_id(model):
    """Read mask_token_id from model config (Dream) or fall back to LLaDA default."""
    cfg = getattr(model, "config", None)
    if cfg is not None and hasattr(cfg, "mask_token_id") and cfg.mask_token_id is not None:
        return int(cfg.mask_token_id)
    return 126336


def set_mask_token_id(tid: int):
    """Globally override the mask token id (call after loading model)."""
    global MASK_TOKEN_ID
    MASK_TOKEN_ID = int(tid)


def set_logit_shift(s: int):
    """Globally set the logit shift (0 for LLaDA/Dream, 1 for DiffuLLaMA)."""
    global LOGIT_SHIFT
    LOGIT_SHIFT = int(s)


def set_backbone_meta(meta):
    """Record the backbone meta dict so helpers can build the right
    attention mask and apply shift-aware indexing."""
    global BACKBONE_META
    BACKBONE_META = dict(meta) if meta is not None else None


def _build_attn_mask(input_ids, dtype):
    """Build a backbone-appropriate attention mask for one forward pass."""
    if BACKBONE_META is not None:
        from ladit.model.backbone import build_attention_mask
        return build_attention_mask(input_ids, BACKBONE_META, dtype=dtype)
    # Fallback when no backbone meta is set: return bool to be SDPA-safe.
    return torch.ones_like(input_ids, dtype=torch.bool)


def _logits_for_positions(logits: torch.Tensor, prompt_len: int) -> torch.Tensor:
    """Return the (target_len, V) tensor of logits aligned so that
    logits_out[i] is the model's prediction for target position i.

    For LLaDA / Dream the model's logits are already "at position i" so we
    simply slice [prompt_len:]. For DiffuLLaMA with shift=1 the prediction
    for target position i lives at logits[prompt_len + i - 1]; we therefore
    slice from prompt_len-1.
    """
    if LOGIT_SHIFT == 0:
        return logits[0, prompt_len:]
    s = int(LOGIT_SHIFT)
    # Guard: if prompt_len < s this slicing would go negative.
    start = max(0, prompt_len - s)
    return logits[0, start:]


@torch.no_grad()
def translate_single(
    model,
    tokenizer,
    source_text: str,
    target_length: int,
    num_steps: int = 32,
    schedule_name: str = "med",
    sig_scores: Optional[torch.Tensor] = None,
    anchor_positions: Optional[torch.Tensor] = None,
    temperature: float = 0.0,
    phase_ratio: float = 0.5,
    device: str = "cuda",
) -> dict:
    """Translate a single source sentence using masked diffusion decoding.

    Args:
        model: LLaDA model
        tokenizer: tokenizer
        source_text: English source sentence
        target_length: number of target tokens to generate
        num_steps: denoising steps
        schedule_name: order schedule to use
        sig_scores: pre-computed SIG scores (required for sig_first/reverse_sig)
        anchor_positions: oracle anchor positions (for oracle_anchor schedule)
        temperature: sampling temperature (0 = argmax)
        device: computation device

    Returns:
        dict with translation, per-step info, timing
    """
    schedule_fn = get_schedule(schedule_name)

    # Tokenize prompt
    prompt = get_prompt_prefix().format(source=source_text)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")[0]
    prompt_len = prompt_ids.size(0)

    # Initialize all-masked target canvas
    input_ids = torch.cat([
        prompt_ids.to(device),
        torch.full((target_length,), MASK_TOKEN_ID, dtype=torch.long, device=device),
    ]).unsqueeze(0)  # (1, prompt_len + target_length)

    model_dtype = next(model.parameters()).dtype
    attention_mask = _build_attn_mask(input_ids, dtype=model_dtype)
    total_target = target_length

    step_info = []
    start_time = time.time()

    for step in range(num_steps):
        # Check which positions are still masked (in target region only)
        target_region = input_ids[0, prompt_len:]
        masked_positions = (target_region == MASK_TOKEN_ID)

        if not masked_positions.any():
            break  # All unmasked

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = _logits_for_positions(outputs.logits, prompt_len)  # (target_length, V)

        # Select positions to unmask using schedule
        schedule_kwargs = {
            "total_target_len": total_target,
            "sig_scores": sig_scores,
            "anchor_positions": anchor_positions,
            "phase_ratio": phase_ratio,
        }
        positions_to_unmask = schedule_fn(
            logits, masked_positions, step, num_steps, **schedule_kwargs
        )

        if len(positions_to_unmask) == 0:
            continue

        # Sample or argmax for selected positions
        selected_logits = logits[positions_to_unmask]  # (k, V)
        if temperature > 0:
            probs = F.softmax(selected_logits / temperature, dim=-1)
            sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
        else:
            sampled = selected_logits.argmax(dim=-1)

        # Update input_ids
        for pos, tok in zip(positions_to_unmask, sampled):
            input_ids[0, prompt_len + pos] = tok

        step_info.append({
            "step": step,
            "num_unmasked": len(positions_to_unmask),
            "remaining_masked": (input_ids[0, prompt_len:] == MASK_TOKEN_ID).sum().item(),
        })

    elapsed = time.time() - start_time

    # Force-fill any remaining masks with argmax from last forward pass
    remaining_masks = (input_ids[0, prompt_len:] == MASK_TOKEN_ID)
    if remaining_masks.any():
        outputs_final = model(input_ids=input_ids, attention_mask=attention_mask)
        final_logits = _logits_for_positions(outputs_final.logits, prompt_len)
        remaining_idx = remaining_masks.nonzero(as_tuple=True)[0]
        for pos in remaining_idx:
            input_ids[0, prompt_len + pos] = final_logits[pos].argmax()

    # Decode translation
    target_ids = input_ids[0, prompt_len:].cpu().tolist()
    # Stop at first EOS (do not skip all EOS tokens)
    eos_id = tokenizer.eos_token_id
    clean_ids = []
    for tid in target_ids:
        if tid == eos_id:
            break  # Stop at first EOS
        if tid == MASK_TOKEN_ID:
            continue  # Skip any remaining masks (shouldn't happen after force-fill)
        clean_ids.append(tid)

    translation = tokenizer.decode(clean_ids, skip_special_tokens=True)

    return {
        "source": source_text,
        "translation": translation,
        "target_length": target_length,
        "num_steps": num_steps,
        "schedule": schedule_name,
        "elapsed_sec": elapsed,
        "step_info": step_info,
    }


def batch_translate(
    model,
    tokenizer,
    sources: List[str],
    target_lengths: List[int],
    num_steps: int = 32,
    schedule_name: str = "med",
    source_corpus: Optional[List[str]] = None,
    temperature: float = 0.0,
    phase_ratio: float = 0.5,
    device: str = "cuda",
    show_progress: bool = True,
) -> List[dict]:
    """Translate multiple source sentences.

    For SIG-based schedules, computes SIG scores per sentence.
    """
    results = []
    needs_sig = schedule_name in ("sig_first", "reverse_sig", "hybrid_med_sig", "ew_sig")

    iterator = enumerate(zip(sources, target_lengths))
    if show_progress:
        iterator = tqdm(list(iterator), desc=f"Translating ({schedule_name})")

    for idx, (src, tgt_len) in iterator:
        sig_scores = None

        if needs_sig and source_corpus is not None:
            # Compute SIG scores for this sentence
            distractor = select_distractor(src, source_corpus, seed=idx)
            prompt_true = get_prompt_prefix().format(source=src)
            prompt_dist = get_prompt_prefix().format(source=distractor)

            true_ids = tokenizer.encode(prompt_true, add_special_tokens=False,
                                        return_tensors="pt")[0]
            dist_ids = tokenizer.encode(prompt_dist, add_special_tokens=False,
                                        return_tensors="pt")[0]

            sig_scores = compute_sig_scores(
                model, tokenizer, true_ids, tgt_len, dist_ids,
                mask_token_id=MASK_TOKEN_ID, device=device,
            )

        result = translate_single(
            model, tokenizer, src, tgt_len,
            num_steps=num_steps,
            schedule_name=schedule_name,
            sig_scores=sig_scores,
            temperature=temperature,
            phase_ratio=phase_ratio,
            device=device,
        )

        # Add SIG-Concentration if available
        if sig_scores is not None:
            result["sig_concentration"] = compute_sig_concentration(sig_scores)

        results.append(result)

    return results


def predict_target_lengths(
    model,
    tokenizer,
    sources: List[str],
    references: Optional[List[str]] = None,
    method: str = "oracle",
    ratio: float = 1.5,
    candidate_ratios: Optional[List[float]] = None,
    device: str = "cuda",
    probe_out: Optional[List[Dict]] = None,
) -> List[int]:
    """Predict target lengths for each source sentence.

    Methods:
    - "oracle": use reference length (for controlled experiments)
    - "ratio": multiply source length by fixed ratio
    - "eos_probe": EOS-probing length selection (LaDiT)

    If `probe_out` is a list, per-sentence probe diagnostics from adaptive
    length methods (entropy_valley, eos_probe, partial_unmask) are appended
    to it. Length matches `sources` for those methods, otherwise empty.
    """
    lengths = []

    if method == "oracle":
        assert references is not None, "References required for oracle length"
        for ref in references:
            ref_ids = tokenizer.encode(ref, add_special_tokens=False)
            # +1 for EOS token (matching training format)
            lengths.append(len(ref_ids) + 1)
    elif method == "ratio":
        for src in sources:
            src_ids = tokenizer.encode(src, add_special_tokens=False)
            lengths.append(max(1, int(len(src_ids) * ratio)))
    elif method == "eos_probe":
        from ladit.decoding.length_adaptive import eos_probe_batch
        lengths, probe_data = eos_probe_batch(
            model, tokenizer, sources,
            candidate_ratios=candidate_ratios,
            device=device,
        )
        if probe_out is not None:
            probe_out.extend(probe_data)
    elif method == "entropy_valley":
        from ladit.decoding.length_adaptive import entropy_valley_batch
        lengths, probe_data = entropy_valley_batch(
            model, tokenizer, sources,
            candidate_ratios=candidate_ratios,
            device=device,
        )
        if probe_out is not None:
            probe_out.extend(probe_data)
    elif method == "partial_unmask":
        from ladit.decoding.length_adaptive import partial_unmask_batch
        lengths, probe_data = partial_unmask_batch(
            model, tokenizer, sources,
            max_ratio=ratio,
            warmup_steps=4,
            method="first_eos_argmax",
            device=device,
        )
        if probe_out is not None:
            probe_out.extend(probe_data)
    else:
        raise ValueError(f"Unknown length method: {method}")

    return lengths


def main():
    parser = argparse.ArgumentParser(description="LaDiT Translation")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--backbone", type=str, default="auto",
                        choices=["auto", "llada", "dream", "diffullama"],
                        help="Backbone to load (auto-detected by default).")
    parser.add_argument("--lora_path", type=str, default=None,
                        help="Path to LoRA adapter (if used)")
    parser.add_argument("--input_file", type=str, required=True,
                        help="JSONL with 'en' and optionally 'zh' fields")
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--schedule", type=str, default="med",
                        choices=["random", "l2r", "med", "sig_first", "reverse_sig",
                                 "hybrid_med_sig", "ew_sig", "oracle_anchor"])
    parser.add_argument("--num_steps", type=int, default=32)
    parser.add_argument("--length_method", type=str, default="oracle",
                        choices=["oracle", "ratio", "eos_probe",
                                 "entropy_valley", "partial_unmask"])
    parser.add_argument("--length_ratio", type=float, default=1.5)
    parser.add_argument("--candidate_ratios", type=str,
                        default="0.6,0.7,0.75,0.8,0.85,0.9,1.0,1.1,1.2",
                        help="Comma-separated ratios for EOS probing candidates")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--phase_ratio", type=float, default=0.5,
                        help="Phase boundary for hybrid_med_sig (0.5 = 50%% MED + 50%% SIG)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_examples", type=int, default=-1)
    parser.add_argument("--src_key", type=str, default=None,
                        help="JSONL field for source text (overrides auto-detect). "
                             "Use 'en' for en->zh / en->de, 'zh' for zh->en.")
    parser.add_argument("--tgt_key", type=str, default=None,
                        help="JSONL field for target/reference text.")
    args = parser.parse_args()

    # Load model + tokenizer via unified backbone loader.
    from ladit.model.backbone import load_backbone
    print(f"Loading backbone={args.backbone!r} from {args.model_path}...")
    model, tokenizer, backbone_meta = load_backbone(
        name=args.backbone,
        path=args.model_path,
        dtype=torch.bfloat16,
    )
    model = model.to(args.device)

    # Propagate backbone-specific knobs to the module-level state.
    set_mask_token_id(backbone_meta["mask_token_id"])
    set_logit_shift(int(backbone_meta.get("logit_shift", 0)))
    set_backbone_meta(backbone_meta)
    # Also propagate mask_token_id to the length-adaptive probe module.
    from ladit.decoding.length_adaptive import set_mask_token_id as _la_set_mask_id
    from ladit.decoding.length_adaptive import set_logit_shift as _la_set_shift
    from ladit.decoding.length_adaptive import set_backbone_meta as _la_set_meta
    _la_set_mask_id(backbone_meta["mask_token_id"])
    _la_set_shift(int(backbone_meta.get("logit_shift", 0)))
    _la_set_meta(backbone_meta)

    # Propagate prompt-template family (plain vs qwen chat) to the dataset
    # module so `get_prompt_prefix()` returns the right template.
    from ladit.data.mt_dataset import set_template_family
    set_template_family(backbone_meta.get("template_family", "plain"))
    print(f"Backbone resolved: {backbone_meta['name']} "
          f"(mask_token_id={backbone_meta['mask_token_id']}, "
          f"template={backbone_meta['template_family']}, "
          f"logit_shift={backbone_meta.get('logit_shift', 0)})")

    # Load LoRA if specified
    if args.lora_path:
        print(f"Loading LoRA from {args.lora_path}...")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora_path)
        model = model.merge_and_unload()

    model.eval()

    # Load data
    data = []
    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    if args.max_examples > 0:
        data = data[:args.max_examples]

    # Determine src/tgt keys: explicit CLI flag wins; otherwise auto-detect.
    sample = data[0] if data else {}
    if args.src_key and args.tgt_key:
        src_key, tgt_key = args.src_key, args.tgt_key
    elif "en" in sample and "zh" in sample:
        src_key, tgt_key = "en", "zh"
    elif "en" in sample and "de" in sample:
        src_key, tgt_key = "en", "de"
    else:
        src_key, tgt_key = "en", "zh"  # legacy fallback

    sources = [d[src_key] for d in data]
    references = [d.get(tgt_key) for d in data] if all(tgt_key in d for d in data) else None

    # Predict target lengths
    candidate_ratios = [float(r) for r in args.candidate_ratios.split(",")]
    probe_data: List[Dict] = []
    target_lengths = predict_target_lengths(
        model, tokenizer, sources, references,
        method=args.length_method, ratio=args.length_ratio,
        candidate_ratios=candidate_ratios, device=args.device,
        probe_out=probe_data,
    )

    # Source corpus for distractor selection (SIG schedules)
    source_corpus = sources if args.schedule in ("sig_first", "reverse_sig", "hybrid_med_sig", "ew_sig") else None

    # Translate
    results = batch_translate(
        model, tokenizer, sources, target_lengths,
        num_steps=args.num_steps,
        schedule_name=args.schedule,
        source_corpus=source_corpus,
        temperature=args.temperature,
        phase_ratio=args.phase_ratio,
        device=args.device,
    )

    # Add references and metadata
    for i, r in enumerate(results):
        if references:
            r["reference"] = references[i]
        r["length_method"] = args.length_method
        # Surface adaptive-length diagnostics (selected_ratio etc.) when
        # available so downstream analysis can recover the per-sentence
        # ratio choice without rerunning the probe.
        if probe_data and i < len(probe_data):
            pd = probe_data[i]
            if "selected_ratio" in pd:
                r["selected_ratio"] = pd["selected_ratio"]
            if "src_len_tokens" in pd:
                r["src_len_tokens"] = pd["src_len_tokens"]
            if "candidate_ratios" in pd:
                r["candidate_ratios"] = pd["candidate_ratios"]
            if "ratio_to_length" in pd:
                # JSON-friendly: keys must be strings.
                r["ratio_to_length"] = {
                    f"{k:.4f}": v for k, v in pd["ratio_to_length"].items()
                }

    # Save results
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Translated {len(results)} sentences → {output_path}")

    # Quick stats
    if results:
        avg_time = np.mean([r["elapsed_sec"] for r in results])
        print(f"Average time per sentence: {avg_time:.2f}s")


if __name__ == "__main__":
    main()
