"""
Length-Adaptive Diffusion Translation (LaDiT).

Training-free length prediction for masked diffusion MT.

Methods:
1. EOS probing: sweep candidate lengths, check P(EOS) at last position
2. Partial-Unmask EOS Scan: decode a few steps with generous canvas, then scan P(EOS)
3. Entropy-Valley: sweep candidate lengths, find minimum mean entropy
4. Multi-candidate decode: fully decode at K lengths, score by token log-prob
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ladit.data.mt_dataset import PROMPT_PREFIX, get_prompt_prefix


# Default LLaDA; can be overridden per-run via set_mask_token_id().
MASK_TOKEN_ID = 126336
# Logit shift: 0 for LLaDA / Dream, 1 for DiffuLLaMA-style AR shift.
LOGIT_SHIFT = 0
# Backbone meta (set after load_backbone) used for 4D attention mask.
BACKBONE_META = None


def set_mask_token_id(tid: int):
    """Globally override the mask token id (call after loading model)."""
    global MASK_TOKEN_ID
    MASK_TOKEN_ID = int(tid)


def set_logit_shift(s: int):
    """Set the logit shift used when slicing target-region logits."""
    global LOGIT_SHIFT
    LOGIT_SHIFT = int(s)


def set_backbone_meta(meta):
    """Record backbone meta dict (e.g. for 4D attn mask in DiffuLLaMA)."""
    global BACKBONE_META
    BACKBONE_META = dict(meta) if meta is not None else None


def _build_attn_mask(input_ids, dtype):
    """Build a backbone-appropriate attention mask."""
    if BACKBONE_META is not None:
        from ladit.model.backbone import build_attention_mask
        return build_attention_mask(input_ids, BACKBONE_META, dtype=dtype)
    # Fallback: bool to be SDPA-safe (Dream rejects long-typed masks).
    return torch.ones_like(input_ids, dtype=torch.bool)


def _logits_for_positions(logits, prompt_len: int):
    """Return target-region logits aligned so [i] predicts target token i."""
    if LOGIT_SHIFT == 0:
        return logits[0, prompt_len:]
    s = int(LOGIT_SHIFT)
    return logits[0, max(0, prompt_len - s):]


@torch.no_grad()
def eos_probe_single(
    model,
    tokenizer,
    source_text: str,
    candidate_lengths: List[int],
    device: str = "cuda",
) -> Dict:
    """Probe EOS probability at the last canvas position for multiple candidate lengths.

    For each candidate length L, creates an all-masked canvas [prompt + MASK×L],
    runs one forward pass, and extracts P(EOS) at position prompt_len + L - 1.

    The model was trained with EOS at ref_len+1, so when L matches the oracle
    length, P(EOS) at the last position should be maximized.

    Args:
        model: LLaDA model
        tokenizer: tokenizer
        source_text: English source sentence
        candidate_lengths: list of candidate target lengths to probe
        device: computation device

    Returns:
        dict with per-length EOS probs, best length, all probe data
    """
    prompt = get_prompt_prefix().format(source=source_text)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")[0]
    prompt_len = prompt_ids.size(0)
    eos_id = tokenizer.eos_token_id

    probe_results = {}

    for L in candidate_lengths:
        if L < 1:
            continue

        # Create all-masked canvas
        input_ids = torch.cat([
            prompt_ids.to(device),
            torch.full((L,), MASK_TOKEN_ID, dtype=torch.long, device=device),
        ]).unsqueeze(0)

        attention_mask = _build_attn_mask(input_ids, dtype=next(model.parameters()).dtype)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = _logits_for_positions(outputs.logits, prompt_len)  # (L, V)

        # P(EOS) at the last position
        last_logits = logits[L - 1]
        probs = F.softmax(last_logits, dim=-1)
        eos_prob = probs[eos_id].item()

        # Also compute entropy at last position (useful for analysis)
        entropy = -(probs * (probs + 1e-10).log()).sum().item()

        probe_results[L] = {
            "eos_prob": eos_prob,
            "entropy": entropy,
        }

    if not probe_results:
        return {"best_length": 1, "probe_results": {}, "eos_probs": {}}

    # Select length with highest EOS probability
    best_length = max(probe_results.keys(), key=lambda l: probe_results[l]["eos_prob"])
    eos_probs = {l: r["eos_prob"] for l, r in probe_results.items()}

    return {
        "best_length": best_length,
        "probe_results": probe_results,
        "eos_probs": eos_probs,
    }


@torch.no_grad()
def eos_probe_batch(
    model,
    tokenizer,
    sources: List[str],
    candidate_ratios: List[float] = None,
    device: str = "cuda",
    show_progress: bool = True,
) -> Tuple[List[int], List[Dict]]:
    """Predict target lengths for a batch of sources using EOS probing.

    Args:
        model: LLaDA model
        tokenizer: tokenizer
        sources: list of source sentences
        candidate_ratios: ratios to multiply source token count by
        device: computation device
        show_progress: show tqdm progress bar

    Returns:
        (lengths, probe_data): best lengths and full probe data per sentence
    """
    if candidate_ratios is None:
        candidate_ratios = [0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0, 1.1, 1.2]

    lengths = []
    probe_data = []

    iterator = enumerate(sources)
    if show_progress:
        iterator = tqdm(list(iterator), desc="EOS-probing lengths")

    for idx, src in iterator:
        src_ids = tokenizer.encode(src, add_special_tokens=False)
        src_len = len(src_ids)

        # Generate candidate lengths from ratios, deduplicate
        candidates = sorted(set(
            max(1, int(src_len * r)) + 1  # +1 for EOS slot
            for r in candidate_ratios
        ))

        result = eos_probe_single(model, tokenizer, src, candidates, device)
        lengths.append(result["best_length"])
        probe_data.append(result)

    return lengths, probe_data


@torch.no_grad()
def partial_unmask_length_scan(
    model,
    tokenizer,
    source_text: str,
    max_length: int,
    warmup_steps: int = 4,
    device: str = "cuda",
) -> Dict:
    """Predict target length via Partial-Unmask EOS Scan.

    1. Create a generous canvas [prompt + MASK × max_length]
    2. Run `warmup_steps` MED steps to unmask easy positions (build context)
    3. Do one final forward pass; scan P(EOS) across all positions
    4. The position where P(EOS) peaks (excluding last) indicates content end

    This works because with partial context, the model has a much sharper
    EOS signal than at full noise (all-masked).

    Args:
        model: LLaDA model
        tokenizer: tokenizer
        source_text: English source sentence
        max_length: generous canvas length (e.g., 1.5× src_len)
        warmup_steps: number of MED warmup steps before scanning
        device: computation device

    Returns:
        dict with predicted_length, P(EOS) profile, warmup info
    """
    from ladit.decoding.schedules import med_schedule, compute_unmask_count

    prompt = get_prompt_prefix().format(source=source_text)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")[0]
    prompt_len = prompt_ids.size(0)
    eos_id = tokenizer.eos_token_id

    input_ids = torch.cat([
        prompt_ids.to(device),
        torch.full((max_length,), MASK_TOKEN_ID, dtype=torch.long, device=device),
    ]).unsqueeze(0)
    attention_mask = _build_attn_mask(input_ids, dtype=next(model.parameters()).dtype)

    total_target = max_length
    num_unmasked = 0

    # Warmup: run a few MED steps to build partial context
    for step in range(warmup_steps):
        target_region = input_ids[0, prompt_len:]
        masked_positions = (target_region == MASK_TOKEN_ID)
        if not masked_positions.any():
            break

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = _logits_for_positions(outputs.logits, prompt_len)

        # MED: unmask lowest-entropy positions
        schedule_kwargs = {"total_target_len": total_target}
        positions_to_unmask = med_schedule(
            logits, masked_positions, step, warmup_steps + 1,
            **schedule_kwargs,
        )

        if len(positions_to_unmask) == 0:
            continue

        # Argmax fill
        for pos in positions_to_unmask:
            input_ids[0, prompt_len + pos] = logits[pos].argmax()
        num_unmasked += len(positions_to_unmask)

    # Final scan: one forward pass with partial context
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = _logits_for_positions(outputs.logits, prompt_len)
    probs = F.softmax(logits.float(), dim=-1)

    eos_probs = probs[:, eos_id].cpu().numpy()
    entropies = -(probs * (probs + 1e-10).log()).sum(dim=-1).cpu().numpy()
    max_probs = probs.max(dim=-1).values.cpu().numpy()

    # Identify which positions are still masked
    target_region = input_ids[0, prompt_len:].cpu()
    still_masked = (target_region == MASK_TOKEN_ID).numpy()

    # Method A: First masked position where P(EOS) > threshold
    # (unmasked positions have already been "claimed" by real tokens)
    predicted_by_thresh = {}
    for thresh in [0.01, 0.05, 0.1, 0.2, 0.5]:
        found = False
        for i in range(max_length):
            if eos_probs[i] > thresh:
                predicted_by_thresh[f"thresh_{thresh}"] = i + 1  # length = pos + 1
                found = True
                break
        if not found:
            predicted_by_thresh[f"thresh_{thresh}"] = max_length

    # Method B: Among MASKED positions, find where P(EOS) is highest
    masked_eos = eos_probs.copy()
    masked_eos[~still_masked] = -1  # ignore unmasked
    if still_masked.any():
        best_masked_pos = int(np.argmax(masked_eos))
        predicted_by_thresh["argmax_masked_eos"] = best_masked_pos + 1
    else:
        predicted_by_thresh["argmax_masked_eos"] = max_length

    # Method C: First position where argmax IS eos_id
    first_eos_argmax = max_length
    for i in range(max_length):
        if logits[i].argmax().item() == eos_id:
            first_eos_argmax = i + 1
            break
    predicted_by_thresh["first_eos_argmax"] = first_eos_argmax

    # Method D: Transition detection — find where non-EOS confidence drops
    # (i.e., where the model stops being confident about content tokens)
    non_eos_confidence = max_probs.copy()
    for i in range(max_length):
        if logits[i].argmax().item() == eos_id:
            non_eos_confidence[i] = 0  # EOS positions don't count

    # Find the last position with reasonable confidence (> 0.05)
    last_confident = 0
    for i in range(max_length):
        if non_eos_confidence[i] > 0.05:
            last_confident = i
    predicted_by_thresh["last_confident"] = last_confident + 1

    return {
        "predicted_lengths": predicted_by_thresh,
        "eos_probs": eos_probs.tolist(),
        "entropies": entropies.tolist(),
        "still_masked": still_masked.tolist(),
        "num_warmup_unmasked": num_unmasked,
        "warmup_steps": warmup_steps,
        "max_length": max_length,
    }


@torch.no_grad()
def partial_unmask_batch(
    model,
    tokenizer,
    sources: List[str],
    max_ratio: float = 1.5,
    warmup_steps: int = 4,
    method: str = "first_eos_argmax",
    device: str = "cuda",
    show_progress: bool = True,
) -> Tuple[List[int], List[Dict]]:
    """Predict target lengths using Partial-Unmask EOS Scan.

    Args:
        model: LLaDA model
        tokenizer: tokenizer
        sources: source sentences
        max_ratio: canvas length as ratio of source length
        warmup_steps: MED warmup steps before scanning
        method: which prediction method to use from the scan results
        device: computation device
        show_progress: show progress bar

    Returns:
        (lengths, scan_data)
    """
    lengths = []
    scan_data = []

    iterator = enumerate(sources)
    if show_progress:
        iterator = tqdm(list(iterator), desc=f"Partial-unmask scan (w={warmup_steps})")

    for idx, src in iterator:
        src_ids = tokenizer.encode(src, add_special_tokens=False)
        max_L = max(1, int(len(src_ids) * max_ratio)) + 1

        result = partial_unmask_length_scan(
            model, tokenizer, src, max_L,
            warmup_steps=warmup_steps, device=device,
        )

        predicted = result["predicted_lengths"].get(method, max_L)
        lengths.append(predicted)
        scan_data.append(result)

    return lengths, scan_data


@torch.no_grad()
def entropy_valley_probe(
    model,
    tokenizer,
    source_text: str,
    candidate_lengths: List[int],
    device: str = "cuda",
) -> Dict:
    """Find the candidate length that minimizes mean entropy of target positions.

    At the correct length, the model is most "confident" about its predictions.
    Too short → tokens are "compressed", some positions uncertain.
    Too long → padding positions add uncertainty.

    Args:
        model: LLaDA model
        tokenizer: tokenizer
        source_text: English source
        candidate_lengths: list of lengths to test
        device: computation device

    Returns:
        dict with best_length, per-length entropy stats
    """
    prompt = get_prompt_prefix().format(source=source_text)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")[0]
    prompt_len = prompt_ids.size(0)

    results = {}

    for L in candidate_lengths:
        if L < 1:
            continue

        input_ids = torch.cat([
            prompt_ids.to(device),
            torch.full((L,), MASK_TOKEN_ID, dtype=torch.long, device=device),
        ]).unsqueeze(0)
        attention_mask = _build_attn_mask(input_ids, dtype=next(model.parameters()).dtype)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = _logits_for_positions(outputs.logits, prompt_len)
        probs = F.softmax(logits.float(), dim=-1)

        entropies = -(probs * (probs + 1e-10).log()).sum(dim=-1)

        # Mean entropy excluding last position (last is always EOS with ~0 entropy)
        if L > 1:
            mean_ent = entropies[:-1].mean().item()
        else:
            mean_ent = entropies.mean().item()

        mean_max_prob = probs.max(dim=-1).values.mean().item()

        results[L] = {
            "mean_entropy_nolast": mean_ent,
            "mean_max_prob": mean_max_prob,
        }

    if not results:
        return {"best_length": 1, "results": {}}

    best_entropy_L = min(results.keys(), key=lambda L: results[L]["mean_entropy_nolast"])

    return {
        "best_length": best_entropy_L,
        "results": results,
    }


@torch.no_grad()
def entropy_valley_batch(
    model,
    tokenizer,
    sources: List[str],
    candidate_ratios: List[float] = None,
    device: str = "cuda",
    show_progress: bool = True,
) -> Tuple[List[int], List[Dict]]:
    """Predict target lengths using entropy-valley method."""
    if candidate_ratios is None:
        candidate_ratios = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.1, 1.2, 1.3]

    lengths = []
    probe_data = []

    iterator = enumerate(sources)
    if show_progress:
        iterator = tqdm(list(iterator), desc="Entropy-valley probing")

    sorted_ratios = sorted(candidate_ratios)

    for idx, src in iterator:
        src_ids = tokenizer.encode(src, add_special_tokens=False)
        src_len = len(src_ids)

        # Map each ratio to its (deduplicated) candidate length.
        ratio_to_length = {
            r: max(1, int(src_len * r)) + 1
            for r in sorted_ratios
        }
        candidates = sorted(set(ratio_to_length.values()))

        result = entropy_valley_probe(model, tokenizer, src, candidates, device)
        best_length = result["best_length"]
        lengths.append(best_length)

        # Recover the selected ratio: the smallest ratio whose mapped length
        # equals the chosen best_length (deterministic tie-breaking on
        # length-collisions for short sources).
        selected_ratio = next(
            (r for r in sorted_ratios if ratio_to_length[r] == best_length),
            None,
        )
        # Attach back into probe data so callers (translate.py) can surface it.
        result["selected_ratio"] = selected_ratio
        result["src_len_tokens"] = src_len
        result["candidate_ratios"] = list(sorted_ratios)
        result["ratio_to_length"] = ratio_to_length
        probe_data.append(result)

    return lengths, probe_data


@torch.no_grad()
def compute_sequence_score(
    model,
    input_ids: torch.Tensor,
    prompt_len: int,
    eos_id: int,
) -> float:
    """Score a fully-decoded sequence by average token log-probability.

    Runs a fresh forward pass on the decoded sequence and computes
    the mean log P(y_i) for target tokens up to (not including) EOS.

    Args:
        model: LLaDA model
        input_ids: (1, seq_len) fully decoded input ids
        prompt_len: length of the source prompt
        eos_id: EOS token id

    Returns:
        Average token log-probability (higher = more confident)
    """
    attention_mask = _build_attn_mask(input_ids, dtype=next(model.parameters()).dtype)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = _logits_for_positions(outputs.logits, prompt_len)  # (target_len, V)

    target_ids = input_ids[0, prompt_len:]
    log_probs = F.log_softmax(logits, dim=-1)

    total_log_prob = 0.0
    count = 0

    for i, tid in enumerate(target_ids):
        tid_val = tid.item()
        if tid_val == eos_id:
            # Include the EOS token's score (model should be confident about EOS placement)
            total_log_prob += log_probs[i, tid_val].item()
            count += 1
            break
        if tid_val == MASK_TOKEN_ID:
            continue  # Skip remaining masks (shouldn't happen after full decode)
        total_log_prob += log_probs[i, tid_val].item()
        count += 1

    if count == 0:
        return float("-inf")

    return total_log_prob / count


def multi_candidate_decode(
    model,
    tokenizer,
    source_text: str,
    candidate_lengths: List[int],
    num_steps: int = 32,
    schedule_name: str = "med",
    temperature: float = 0.0,
    device: str = "cuda",
) -> dict:
    """Decode at multiple candidate lengths, return the best-scoring translation.

    Args:
        model: LLaDA model
        tokenizer: tokenizer
        source_text: source sentence
        candidate_lengths: list of lengths to try (typically 5 integer offsets {EV-2..EV+2} around the EV-selected length)
        num_steps: denoising steps per candidate
        schedule_name: decoding schedule
        temperature: sampling temperature
        device: computation device

    Returns:
        dict with best translation and comparison data
    """
    from ladit.decoding.translate import translate_single

    eos_id = tokenizer.eos_token_id
    prompt = get_prompt_prefix().format(source=source_text)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")[0]
    prompt_len = prompt_ids.size(0)

    candidates = []

    for L in candidate_lengths:
        result = translate_single(
            model, tokenizer, source_text, L,
            num_steps=num_steps,
            schedule_name=schedule_name,
            temperature=temperature,
            device=device,
        )

        # Score the decoded sequence
        # Reconstruct input_ids for scoring
        decoded_ids = tokenizer.encode(
            result["translation"], add_special_tokens=False
        )
        # Build full sequence: prompt + decoded + eos
        full_ids = (
            prompt_ids.tolist()
            + decoded_ids
            + [eos_id]
        )
        # Pad to target_length if needed
        while len(full_ids) < prompt_len + L:
            full_ids.append(eos_id)
        full_ids = full_ids[:prompt_len + L]

        score_input = torch.tensor([full_ids], dtype=torch.long, device=device)
        score = compute_sequence_score(model, score_input, prompt_len, eos_id)

        result["sequence_score"] = score
        result["candidate_length"] = L
        candidates.append(result)

    # Pick best by sequence score
    best = max(candidates, key=lambda c: c["sequence_score"])
    best["all_candidates"] = [
        {"length": c["candidate_length"], "score": c["sequence_score"],
         "bleu_approx": None}
        for c in candidates
    ]

    return best
