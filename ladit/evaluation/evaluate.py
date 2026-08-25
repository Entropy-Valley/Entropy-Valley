"""
Evaluation metrics for LaDiT MT experiments.

Computes: COMET-22, BLEU, and optional per-sentence scores.
All metrics use SacreBLEU detokenization for reproducibility.
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def compute_bleu(hypotheses: List[str], references: List[str],
                 target_lang: str = "zh") -> dict:
    """Compute BLEU score using SacreBLEU with proper tokenization."""
    import sacrebleu
    tokenize = "zh" if target_lang == "zh" else "13a"
    bleu = sacrebleu.corpus_bleu(hypotheses, [references], tokenize=tokenize)
    return {
        "bleu": bleu.score,
        "bleu_signature": str(bleu),
    }


def compute_comet(
    sources: List[str],
    hypotheses: List[str],
    references: List[str],
    model_name: str = "Unbabel/wmt22-comet-da",
    batch_size: int = 32,
    gpus: int = 1,
) -> dict:
    """Compute COMET-22 score.

    Returns both corpus-level and sentence-level scores.
    """
    from comet import download_model, load_from_checkpoint

    model_path = download_model(model_name)
    model = load_from_checkpoint(model_path)

    data = [
        {"src": s, "mt": h, "ref": r}
        for s, h, r in zip(sources, hypotheses, references)
    ]

    output = model.predict(data, batch_size=batch_size, gpus=gpus)

    return {
        "comet22": output.system_score,
        "comet22_scores": output.scores,  # per-sentence
        "comet22_model": model_name,
    }


def compute_all_metrics(
    sources: List[str],
    hypotheses: List[str],
    references: List[str],
    compute_comet_flag: bool = True,
    comet_model: str = "Unbabel/wmt22-comet-da",
    target_lang: str = "zh",
) -> dict:
    """Compute all standard MT metrics."""
    results = {}

    # BLEU
    bleu_results = compute_bleu(hypotheses, references, target_lang=target_lang)
    results.update(bleu_results)

    # COMET
    if compute_comet_flag:
        try:
            comet_results = compute_comet(sources, hypotheses, references,
                                          model_name=comet_model)
            results["comet22"] = comet_results["comet22"]
            results["comet22_per_sentence"] = comet_results["comet22_scores"]
        except Exception as e:
            print(f"COMET computation failed: {e}")
            results["comet22"] = None
            results["comet22_error"] = str(e)

    # Summary stats
    results["num_examples"] = len(hypotheses)
    avg_hyp_len = np.mean([len(h) for h in hypotheses])
    avg_ref_len = np.mean([len(r) for r in references])
    results["avg_hyp_length"] = float(avg_hyp_len)
    results["avg_ref_length"] = float(avg_ref_len)
    results["length_ratio"] = float(avg_hyp_len / max(avg_ref_len, 1))

    return results


def main():
    parser = argparse.ArgumentParser(description="LaDiT MT Evaluation")
    parser.add_argument("--translations_file", type=str, required=True,
                        help="JSON file from translate.py")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Output JSON for metrics")
    parser.add_argument("--no_comet", action="store_true",
                        help="Skip COMET (faster, for debugging)")
    parser.add_argument("--comet_model", type=str,
                        default="Unbabel/wmt22-comet-da")
    args = parser.parse_args()

    # Load translations
    with open(args.translations_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    sources = [d["source"] for d in data]
    hypotheses = [d["translation"] for d in data]
    references = [d["reference"] for d in data if "reference" in d]

    if len(references) != len(hypotheses):
        print("WARNING: Not all examples have references. Skipping reference-based metrics.")
        return

    # Compute metrics
    print(f"Evaluating {len(hypotheses)} translations...")
    metrics = compute_all_metrics(
        sources, hypotheses, references,
        compute_comet_flag=not args.no_comet,
        comet_model=args.comet_model,
    )

    # Add metadata
    if data:
        metrics["schedule"] = data[0].get("schedule", "unknown")
        metrics["num_steps"] = data[0].get("num_steps", -1)
        metrics["length_method"] = data[0].get("length_method", "unknown")

    # Save
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)

    # Print summary
    print(f"\n{'='*50}")
    print(f"  Schedule: {metrics.get('schedule', 'N/A')}")
    print(f"  Examples: {metrics['num_examples']}")
    print(f"  BLEU:     {metrics['bleu']:.2f}")
    if metrics.get('comet22') is not None:
        print(f"  COMET-22: {metrics['comet22']:.4f}")
    print(f"  Len ratio:{metrics['length_ratio']:.3f}")
    print(f"{'='*50}")
    print(f"Results saved to {output_path}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
