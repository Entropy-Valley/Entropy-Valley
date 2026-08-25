"""
Challenge set construction for diagnostic MT evaluation
(paper Appendix: order-vs-length challenge subsets).

Builds targeted subsets from WMT En→Zh test data to evaluate
translation quality on specific phenomena:

Coverage-Zh: sentences with numbers, dates, named entities, enumerations
  — tests whether the model preserves factual content accurately.

Length-Outliers: sentences with unusual source-target length ratios
  — tests robustness of the decoding pipeline.

Also computes optimal length ratio statistics for analysis of the
fixed per-direction ratio sets used by the Entropy-Valley candidate grid.
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import List, Dict, Tuple


# --- Coverage-Zh filters ---

def has_numbers(text: str) -> bool:
    """Detect sentences with significant numbers (years, amounts, percentages)."""
    patterns = [
        r'\b\d{4}\b',          # years like 2024
        r'\b\d+[,\.]\d+\b',   # decimal/comma numbers like 3.14, 1,000
        r'\d+%',               # percentages
        r'\$\d+',              # dollar amounts
        r'\d+\s*(million|billion|thousand|hundred)', # large numbers
        r'\b\d{2,}\b',         # any number with 2+ digits
    ]
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def has_dates(text: str) -> bool:
    """Detect sentences with dates."""
    patterns = [
        r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d+',
        r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
        r'\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b',
        r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b',
    ]
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def has_named_entities(text: str) -> bool:
    """Detect sentences likely containing named entities (proper nouns).

    Uses capitalization heuristic — not perfect but fast and no dependencies.
    """
    words = text.split()
    if len(words) < 3:
        return False

    # Count capitalized words that aren't sentence-initial
    cap_count = sum(1 for i, w in enumerate(words[1:], 1)
                    if w[0].isupper() and not w.isupper()  # not ALL-CAPS
                    and len(w) > 1)
    return cap_count >= 2


def has_enumeration(text: str) -> bool:
    """Detect sentences with enumerations/lists."""
    patterns = [
        r'\b(first|second|third|fourth|fifth)\b.*\b(first|second|third|fourth|fifth)\b',
        r'(?:,\s*(?:and|or)\s)',  # Oxford comma pattern
        r'(?:\w+,\s+\w+,\s+(?:and|or)\s+\w+)',  # A, B, and C
        r'\b\d+\)\s',  # 1) 2) 3)
    ]
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def has_fertility_mismatch(src: str, tgt: str, threshold: float = 0.5) -> bool:
    """Detect large fertility mismatch (char-level ratio far from median)."""
    src_len = len(src)
    tgt_len = len(tgt)
    if src_len == 0:
        return False
    ratio = tgt_len / src_len
    # Chinese chars are denser — typical ratio for En→Zh is 0.3-0.6 (char level)
    return ratio < 0.15 or ratio > 0.8


# --- Challenge set builder ---

def classify_example(src: str, tgt: str) -> List[str]:
    """Return list of challenge categories this example belongs to."""
    categories = []
    if has_numbers(src):
        categories.append("numbers")
    if has_dates(src):
        categories.append("dates")
    if has_named_entities(src):
        categories.append("named_entities")
    if has_enumeration(src):
        categories.append("enumeration")
    if has_fertility_mismatch(src, tgt):
        categories.append("fertility_mismatch")
    return categories


def build_challenge_sets(data: List[Dict], max_per_category: int = 200) -> Dict[str, List[Dict]]:
    """Build challenge sets from parallel data."""
    challenge_sets = {
        "coverage_zh": [],       # Union of numbers + dates + entities + enumeration
        "numbers": [],
        "dates": [],
        "named_entities": [],
        "enumeration": [],
        "fertility_mismatch": [],
    }
    seen_coverage = set()

    for i, example in enumerate(data):
        src = example["en"]
        tgt = example.get("zh", "")
        categories = classify_example(src, tgt)

        for cat in categories:
            if len(challenge_sets[cat]) < max_per_category:
                challenge_sets[cat].append({**example, "challenge_category": cat, "idx": i})

        # Coverage-Zh = union of number/date/entity/enumeration categories
        if any(c in categories for c in ["numbers", "dates", "named_entities", "enumeration"]):
            if i not in seen_coverage and len(challenge_sets["coverage_zh"]) < max_per_category * 2:
                challenge_sets["coverage_zh"].append({
                    **example,
                    "challenge_categories": [c for c in categories if c != "fertility_mismatch"],
                    "idx": i,
                })
                seen_coverage.add(i)

    return challenge_sets


def compute_length_statistics(data: List[Dict], tokenizer=None) -> Dict:
    """Compute length ratio statistics for better length prediction."""
    char_ratios = []
    token_ratios = []

    for ex in data:
        src = ex["en"]
        tgt = ex.get("zh", "")
        if not tgt:
            continue

        # Character-level ratio
        if len(src) > 0:
            char_ratios.append(len(tgt) / len(src))

        # Token-level ratio (if tokenizer available)
        if tokenizer:
            src_ids = tokenizer.encode(src, add_special_tokens=False)
            tgt_ids = tokenizer.encode(tgt, add_special_tokens=False)
            if len(src_ids) > 0:
                token_ratios.append(len(tgt_ids) / len(src_ids))

    import numpy as np
    stats = {
        "char_ratio_mean": float(np.mean(char_ratios)),
        "char_ratio_median": float(np.median(char_ratios)),
        "char_ratio_std": float(np.std(char_ratios)),
        "char_ratio_p25": float(np.percentile(char_ratios, 25)),
        "char_ratio_p75": float(np.percentile(char_ratios, 75)),
        "num_examples": len(char_ratios),
    }
    if token_ratios:
        stats.update({
            "token_ratio_mean": float(np.mean(token_ratios)),
            "token_ratio_median": float(np.median(token_ratios)),
            "token_ratio_std": float(np.std(token_ratios)),
            "token_ratio_p25": float(np.percentile(token_ratios, 25)),
            "token_ratio_p75": float(np.percentile(token_ratios, 75)),
        })
    return stats


def main():
    parser = argparse.ArgumentParser(description="Build diagnostic challenge sets")
    parser.add_argument("--input_file", type=str, required=True,
                        help="JSONL with 'en' and 'zh' fields")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_per_category", type=int, default=200)
    parser.add_argument("--compute_length_stats", action="store_true", default=True)
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="Path to tokenizer for token-level length stats")
    args = parser.parse_args()

    # Load data
    data = []
    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    print(f"Loaded {len(data)} examples from {args.input_file}")

    # Build challenge sets
    challenge_sets = build_challenge_sets(data, max_per_category=args.max_per_category)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save each challenge set
    summary = {}
    for name, examples in challenge_sets.items():
        if not examples:
            continue
        output_path = output_dir / f"challenge_{name}.jsonl"
        with open(output_path, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        summary[name] = len(examples)
        print(f"  {name}: {len(examples)} examples → {output_path}")

    # Length statistics
    if args.compute_length_stats:
        tokenizer = None
        if args.tokenizer_path:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
            print(f"Tokenizer loaded for length stats: {args.tokenizer_path}")

        length_stats = compute_length_statistics(data, tokenizer)
        stats_path = output_dir / "length_statistics.json"
        with open(stats_path, "w") as f:
            json.dump(length_stats, f, indent=2)
        print(f"\nLength statistics saved to {stats_path}")
        print(f"  Char ratio: mean={length_stats['char_ratio_mean']:.3f}, "
              f"median={length_stats['char_ratio_median']:.3f}")
        if "token_ratio_mean" in length_stats:
            print(f"  Token ratio: mean={length_stats['token_ratio_mean']:.3f}, "
                  f"median={length_stats['token_ratio_median']:.3f}")

    # Save summary
    summary_path = output_dir / "challenge_sets_summary.json"
    with open(summary_path, "w") as f:
        json.dump({"source": args.input_file, "sets": summary,
                    "max_per_category": args.max_per_category}, f, indent=2)
    print(f"\nSummary: {summary}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
