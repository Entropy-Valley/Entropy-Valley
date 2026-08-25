"""
Source Information Gain (SIG) computation.

SIG(i) = [H(p(y_i | distractor, [M]^L)) - H(p(y_i | true_source, [M]^L))] / log|V|

Measures how much the true source reduces uncertainty at each target slot
compared to an unrelated source. High SIG = strong source constraint.
"""
import random
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Compute token-distribution entropy for each position.

    Args:
        logits: (L, V) logits at each position
    Returns:
        entropy: (L,) per-position entropy
    """
    # Cast to float32 for numerical stability in softmax/log
    logits_f32 = logits.float()
    probs = F.softmax(logits_f32, dim=-1)
    log_probs = F.log_softmax(logits_f32, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy


@torch.no_grad()
def compute_sig_scores(
    model,
    tokenizer,
    source_ids: torch.Tensor,     # (prompt_len,)
    target_length: int,
    distractor_ids: torch.Tensor,  # (distractor_prompt_len,)
    mask_token_id: int = 126336,
    device: str = "cuda",
) -> torch.Tensor:
    """Compute SIG(i) for each target position.

    Two forward passes:
    1. H(y_i | true_source, [M]^L) — entropy with true source
    2. H(y_i | distractor, [M]^L) — entropy with distractor source

    SIG(i) = [H_distractor(i) - H_true(i)] / log|V|

    Args:
        model: LLaDA model
        tokenizer: tokenizer
        source_ids: tokenized source prompt (including "Translate..." prefix)
        target_length: number of target tokens to generate
        distractor_ids: tokenized distractor prompt
        mask_token_id: LLaDA mask token ID
        device: computation device

    Returns:
        sig_scores: (target_length,) SIG score per target position
    """
    vocab_size = model.config.vocab_size if hasattr(model.config, 'vocab_size') else 126464
    log_vocab = np.log(vocab_size)

    # Build all-masked target canvas
    mask_canvas = torch.full((target_length,), mask_token_id, dtype=torch.long, device=device)

    # Forward pass 1: true source
    input_true = torch.cat([source_ids.to(device), mask_canvas]).unsqueeze(0)
    attn_true = torch.ones(1, input_true.size(1), dtype=torch.long, device=device)
    outputs_true = model(input_ids=input_true, attention_mask=attn_true)
    logits_true = outputs_true.logits[0]  # (L, V)

    # Extract target positions only
    prompt_len = source_ids.size(0)
    target_logits_true = logits_true[prompt_len:prompt_len + target_length]
    h_true = compute_entropy(target_logits_true)  # (target_length,)

    # Forward pass 2: distractor source
    input_dist = torch.cat([distractor_ids.to(device), mask_canvas]).unsqueeze(0)
    attn_dist = torch.ones(1, input_dist.size(1), dtype=torch.long, device=device)
    outputs_dist = model(input_ids=input_dist, attention_mask=attn_dist)
    logits_dist = outputs_dist.logits[0]

    dist_prompt_len = distractor_ids.size(0)
    target_logits_dist = logits_dist[dist_prompt_len:dist_prompt_len + target_length]
    h_dist = compute_entropy(target_logits_dist)  # (target_length,)

    # SIG = (H_distractor - H_true) / log|V|
    sig_scores = (h_dist - h_true) / log_vocab

    return sig_scores


def compute_sig_concentration(sig_scores: torch.Tensor) -> float:
    """Compute SIG-Concentration using Gini coefficient.

    High Gini = source info concentrated in few slots = order matters more.
    Low Gini = source info uniformly distributed = any order similar.

    Args:
        sig_scores: (L,) SIG scores for each target position
    Returns:
        gini: float in [0, 1]
    """
    scores = sig_scores.cpu().numpy()
    # Clamp negative SIG values to 0 for Gini computation
    scores = np.maximum(scores, 0)

    if scores.sum() == 0:
        return 0.0

    n = len(scores)
    sorted_scores = np.sort(scores)
    cumsum = np.cumsum(sorted_scores)
    gini = (2 * np.sum((np.arange(1, n + 1) * sorted_scores)) / (n * scores.sum())) - (n + 1) / n
    return float(gini)


def select_distractor(
    source_text: str,
    corpus: List[str],
    length_tolerance: float = 0.2,
    seed: Optional[int] = None,
) -> str:
    """Select a distractor source sentence from corpus.

    Protocol: randomly sample a source from the same corpus,
    matched to within +-20% of the true source length.
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()

    src_len = len(source_text.split())
    min_len = int(src_len * (1 - length_tolerance))
    max_len = int(src_len * (1 + length_tolerance))

    candidates = [s for s in corpus if min_len <= len(s.split()) <= max_len and s != source_text]

    if not candidates:
        # Fallback: any sentence different from source
        candidates = [s for s in corpus if s != source_text]

    if not candidates:
        return source_text  # Last resort

    return rng.choice(candidates)
