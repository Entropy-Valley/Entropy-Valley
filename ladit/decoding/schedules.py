"""
Order schedule implementations for masked diffusion MT decoding.

All schedules determine which positions to unmask at each denoising step.
The model forward pass is identical for all schedules — only the unmask
selection strategy differs.
"""
import torch
import numpy as np
from typing import List, Optional, Tuple


def get_schedule(name: str):
    """Get schedule function by name."""
    schedules = {
        "random": random_schedule,
        "l2r": left_to_right_schedule,
        "med": med_schedule,
        "sig_first": sig_first_schedule,
        "reverse_sig": reverse_sig_schedule,
        "hybrid_med_sig": hybrid_med_sig_schedule,
        "ew_sig": entropy_weighted_sig_schedule,
        "oracle_anchor": oracle_anchor_schedule,
    }
    if name not in schedules:
        raise ValueError(f"Unknown schedule: {name}. Choose from {list(schedules.keys())}")
    return schedules[name]


def compute_unmask_count(step: int, total_steps: int, total_masked: int,
                         curve: str = "linear") -> int:
    """Compute how many tokens to unmask at this step.

    Uses a cosine or linear reveal-rate curve so all schedules
    reveal the same number of tokens per step.
    """
    if curve == "linear":
        # Linear: unmask equal fraction each step
        already_unmasked = int(total_masked * step / total_steps)
        next_unmasked = int(total_masked * (step + 1) / total_steps)
        return next_unmasked - already_unmasked
    elif curve == "cosine":
        # Cosine: slower start, faster end
        already = int(total_masked * (1 - np.cos(np.pi * step / total_steps)) / 2)
        next_ = int(total_masked * (1 - np.cos(np.pi * (step + 1) / total_steps)) / 2)
        return next_ - already
    return max(1, total_masked // total_steps)


def random_schedule(
    logits: torch.Tensor,
    masked_positions: torch.Tensor,
    step: int,
    total_steps: int,
    **kwargs,
) -> torch.Tensor:
    """Uniform random unmask order (standard dLLM default)."""
    total_target = kwargs.get("total_target_len", masked_positions.sum().item() + step)
    num_to_unmask = compute_unmask_count(step, total_steps, total_target)

    # Random permutation of masked positions
    masked_indices = masked_positions.nonzero(as_tuple=True)[0]
    perm = torch.randperm(len(masked_indices), device=masked_indices.device)
    selected = masked_indices[perm[:num_to_unmask]]
    return selected


def left_to_right_schedule(
    logits: torch.Tensor,
    masked_positions: torch.Tensor,
    step: int,
    total_steps: int,
    **kwargs,
) -> torch.Tensor:
    """Left-to-right unmask order."""
    total_target = kwargs.get("total_target_len", masked_positions.sum().item() + step)
    num_to_unmask = compute_unmask_count(step, total_steps, total_target)

    masked_indices = masked_positions.nonzero(as_tuple=True)[0]
    # Sort by position (already sorted since we iterate forward)
    sorted_indices = masked_indices[:num_to_unmask]
    return sorted_indices


def med_schedule(
    logits: torch.Tensor,
    masked_positions: torch.Tensor,
    step: int,
    total_steps: int,
    **kwargs,
) -> torch.Tensor:
    """Minimum Entropy Decoding — unmask most confident positions first.

    This is the generic dLLM heuristic: at each step, unmask positions
    where the model is most confident (lowest entropy).
    """
    total_target = kwargs.get("total_target_len", masked_positions.sum().item() + step)
    num_to_unmask = compute_unmask_count(step, total_steps, total_target)

    masked_indices = masked_positions.nonzero(as_tuple=True)[0]

    if len(masked_indices) == 0:
        return torch.tensor([], dtype=torch.long, device=logits.device)

    # Compute entropy at masked positions (float32 for stability)
    masked_logits = logits[masked_indices].float()
    probs = masked_logits.softmax(dim=-1)  # (num_masked, V)
    entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)  # (num_masked,)

    # Select positions with lowest entropy (most confident)
    num_to_unmask = min(num_to_unmask, len(masked_indices))
    _, top_idx = entropy.topk(num_to_unmask, largest=False)
    selected = masked_indices[top_idx]
    return selected


def sig_first_schedule(
    logits: torch.Tensor,
    masked_positions: torch.Tensor,
    step: int,
    total_steps: int,
    sig_scores: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    """SIG-first schedule — unmask high-source-dependence slots first.

    Requires pre-computed SIG scores for each target position.
    Positions with higher SIG are unmasked earlier.
    """
    assert sig_scores is not None, "SIG scores required for sig_first schedule"

    total_target = kwargs.get("total_target_len", masked_positions.sum().item() + step)
    num_to_unmask = compute_unmask_count(step, total_steps, total_target)

    masked_indices = masked_positions.nonzero(as_tuple=True)[0]

    if len(masked_indices) == 0:
        return torch.tensor([], dtype=torch.long, device=logits.device)

    # Get SIG scores for masked positions
    sig_at_masked = sig_scores[masked_indices]

    # Select positions with highest SIG (most source-dependent)
    num_to_unmask = min(num_to_unmask, len(masked_indices))
    _, top_idx = sig_at_masked.topk(num_to_unmask, largest=True)
    selected = masked_indices[top_idx]
    return selected


def reverse_sig_schedule(
    logits: torch.Tensor,
    masked_positions: torch.Tensor,
    step: int,
    total_steps: int,
    sig_scores: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    """Reverse-SIG schedule — unmask LOW-source-dependence slots first.

    Sanity control: if SIG direction matters, this should perform worse.
    """
    assert sig_scores is not None, "SIG scores required for reverse_sig schedule"

    total_target = kwargs.get("total_target_len", masked_positions.sum().item() + step)
    num_to_unmask = compute_unmask_count(step, total_steps, total_target)

    masked_indices = masked_positions.nonzero(as_tuple=True)[0]

    if len(masked_indices) == 0:
        return torch.tensor([], dtype=torch.long, device=logits.device)

    sig_at_masked = sig_scores[masked_indices]

    # Select positions with LOWEST SIG
    num_to_unmask = min(num_to_unmask, len(masked_indices))
    _, top_idx = sig_at_masked.topk(num_to_unmask, largest=False)
    selected = masked_indices[top_idx]
    return selected


def hybrid_med_sig_schedule(
    logits: torch.Tensor,
    masked_positions: torch.Tensor,
    step: int,
    total_steps: int,
    sig_scores: Optional[torch.Tensor] = None,
    phase_ratio: float = 0.5,
    **kwargs,
) -> torch.Tensor:
    """Hybrid MED→SIG schedule — context-first, then source-first.

    Phase 1 (steps 0..K-1): MED — unmask most confident positions first
        to build a context skeleton (function words, easy tokens).
    Phase 2 (steps K..T-1): SIG-first — unmask highest source-dependence
        positions, now benefiting from the context built in Phase 1.

    The insight: SIG-first fails because it reveals hard tokens without
    context. By letting MED build context first, the hard source-dependent
    tokens can be generated with surrounding support.
    """
    assert sig_scores is not None, "SIG scores required for hybrid_med_sig schedule"
    assert 0.0 < phase_ratio < 1.0, f"phase_ratio must be in (0, 1), got {phase_ratio}"

    phase_boundary = int(total_steps * phase_ratio)

    if step < phase_boundary:
        # Phase 1: MED (confidence-first)
        return med_schedule(logits, masked_positions, step, total_steps, **kwargs)
    else:
        # Phase 2: SIG-first (source-dependence-first)
        return sig_first_schedule(
            logits, masked_positions, step, total_steps,
            sig_scores=sig_scores, **kwargs
        )


def entropy_weighted_sig_schedule(
    logits: torch.Tensor,
    masked_positions: torch.Tensor,
    step: int,
    total_steps: int,
    sig_scores: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    """Entropy-weighted SIG schedule — blend confidence and source info.

    Score(i) = α(t) * SIG(i) + (1-α(t)) * (-entropy(i))

    α(t) increases linearly from 0 to 1 over decoding steps:
    early steps prioritize confidence (low entropy), late steps
    prioritize source dependence (high SIG).

    This is a smooth version of the hybrid schedule — no hard boundary.
    """
    assert sig_scores is not None, "SIG scores required for entropy_weighted_sig"

    total_target = kwargs.get("total_target_len", masked_positions.sum().item() + step)
    num_to_unmask = compute_unmask_count(step, total_steps, total_target)

    masked_indices = masked_positions.nonzero(as_tuple=True)[0]

    if len(masked_indices) == 0:
        return torch.tensor([], dtype=torch.long, device=logits.device)

    # Compute entropy at masked positions
    masked_logits = logits[masked_indices].float()
    probs = masked_logits.softmax(dim=-1)
    entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)  # (num_masked,)

    # Clamp negative SIG to 0 before normalizing — preserves zero baseline
    # (SIG < 0 means distractor is MORE informative, which is noise)
    sig_at_masked = sig_scores[masked_indices].clamp(min=0)

    sig_max = sig_at_masked.max()
    if sig_max > 0:
        sig_norm = sig_at_masked / sig_max  # [0, 1] with 0 = no source dependence
    else:
        sig_norm = torch.zeros_like(sig_at_masked)

    ent_min, ent_max = entropy.min(), entropy.max()
    if ent_max > ent_min:
        ent_norm = (entropy - ent_min) / (ent_max - ent_min)
    else:
        ent_norm = torch.zeros_like(entropy)

    # α increases linearly: 0 at step 0, 1 at step T-1
    alpha = step / max(total_steps - 1, 1)

    # Combined score: higher = unmask first
    # SIG: higher is better (more source-dependent)
    # Entropy: LOWER is better (more confident) → use negative
    combined = alpha * sig_norm + (1 - alpha) * (1 - ent_norm)

    num_to_unmask = min(num_to_unmask, len(masked_indices))
    _, top_idx = combined.topk(num_to_unmask, largest=True)
    selected = masked_indices[top_idx]
    return selected


def oracle_anchor_schedule(
    logits: torch.Tensor,
    masked_positions: torch.Tensor,
    step: int,
    total_steps: int,
    anchor_positions: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    """Oracle-anchor schedule — gold-aligned anchors first, then MED.

    Upper bound analysis: reveals lexically aligned positions first,
    then falls back to MED for remaining positions.
    """
    total_target = kwargs.get("total_target_len", masked_positions.sum().item() + step)
    num_to_unmask = compute_unmask_count(step, total_steps, total_target)

    masked_indices = masked_positions.nonzero(as_tuple=True)[0]

    if len(masked_indices) == 0:
        return torch.tensor([], dtype=torch.long, device=logits.device)

    if anchor_positions is not None:
        # Prioritize anchor positions that are still masked
        anchor_mask = torch.zeros_like(masked_positions)
        anchor_mask[anchor_positions] = True
        anchor_and_masked = masked_positions & anchor_mask
        anchor_indices = anchor_and_masked.nonzero(as_tuple=True)[0]

        if len(anchor_indices) >= num_to_unmask:
            return anchor_indices[:num_to_unmask]

        # Fill remaining with MED
        non_anchor_masked = masked_positions & ~anchor_mask
        remaining = num_to_unmask - len(anchor_indices)
        non_anchor_indices = non_anchor_masked.nonzero(as_tuple=True)[0]

        if len(non_anchor_indices) > 0 and remaining > 0:
            probs = logits[non_anchor_indices].softmax(dim=-1)
            entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)
            _, top_idx = entropy.topk(min(remaining, len(non_anchor_indices)), largest=False)
            med_selected = non_anchor_indices[top_idx]
            return torch.cat([anchor_indices, med_selected])

        return anchor_indices

    # Fallback to MED if no anchors
    return med_schedule(logits, masked_positions, step, total_steps, **kwargs)
