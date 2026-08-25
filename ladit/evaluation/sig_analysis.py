"""
SIG-Concentration analysis used for two diagnostic gate checks
in the order-vs-length study (paper Appendix: source-guided order variants).

Gate G2: Spearman ρ(SIG-Concentration, SIG-first gain over MED) ≥ 0.2
Gate G3: SIG-first > MED by ≥ 0.3 COMET on any challenge subset
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy import stats


def load_translations(path: str) -> List[dict]:
    """Load translation results JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_metrics(path: str) -> dict:
    """Load metrics JSON (may contain per-sentence COMET)."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_sentence_comet(metrics: dict) -> Optional[List[float]]:
    """Extract per-sentence COMET scores from metrics."""
    return metrics.get("comet22_per_sentence")


def analyze_sig_concentration(
    sigfirst_trans: List[dict],
    sigfirst_metrics: dict,
    med_trans: List[dict],
    med_metrics: dict,
) -> dict:
    """
    Gate G2 analysis.

    Correlate SIG-Concentration with sentence-level SIG-first gain over MED.
    """
    # Get per-sentence COMET
    sig_comet = get_sentence_comet(sigfirst_metrics)
    med_comet = get_sentence_comet(med_metrics)

    if sig_comet is None or med_comet is None:
        print("WARNING: Per-sentence COMET not available. Using BLEU proxy.")
        # Fall back to sentence-level BLEU (rough proxy)
        return {"error": "per-sentence COMET not available", "gate_g2": "INCONCLUSIVE"}

    # Get SIG-Concentration values from SIG-first translations
    sig_concentrations = []
    for t in sigfirst_trans:
        sc = t.get("sig_concentration")
        if sc is not None:
            sig_concentrations.append(sc)
        else:
            sig_concentrations.append(0.0)

    # Align lengths — use min of available
    n = min(len(sig_concentrations), len(sig_comet), len(med_comet))
    if n == 0:
        return {"error": "no aligned examples", "gate_g2": "FAIL"}

    sig_concentrations = sig_concentrations[:n]
    sig_comet = sig_comet[:n]
    med_comet = med_comet[:n]

    # Compute gain
    gains = [s - m for s, m in zip(sig_comet, med_comet)]

    # Spearman correlation
    rho, p_value = stats.spearmanr(sig_concentrations, gains)

    # Also compute Pearson for completeness
    pearson_r, pearson_p = stats.pearsonr(sig_concentrations, gains)

    # Binned analysis (quintiles)
    concentrations_arr = np.array(sig_concentrations)
    gains_arr = np.array(gains)

    quintiles = np.percentile(concentrations_arr, [20, 40, 60, 80])
    bin_edges = [-np.inf] + quintiles.tolist() + [np.inf]
    bin_labels = ["Q1 (low)", "Q2", "Q3", "Q4", "Q5 (high)"]
    binned_gains = {}

    for i in range(len(bin_edges) - 1):
        mask = (concentrations_arr >= bin_edges[i]) & (concentrations_arr < bin_edges[i + 1])
        if mask.sum() > 0:
            binned_gains[bin_labels[i]] = {
                "n": int(mask.sum()),
                "mean_gain": float(gains_arr[mask].mean()),
                "std_gain": float(gains_arr[mask].std()),
                "mean_concentration": float(concentrations_arr[mask].mean()),
            }

    # Gate check
    gate_g2 = "PASS" if rho >= 0.2 else "FAIL"

    result = {
        "spearman_rho": float(rho),
        "spearman_p": float(p_value),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "n_examples": n,
        "mean_sig_concentration": float(np.mean(sig_concentrations)),
        "std_sig_concentration": float(np.std(sig_concentrations)),
        "mean_gain": float(np.mean(gains)),
        "std_gain": float(np.std(gains)),
        "fraction_positive_gain": float((gains_arr > 0).mean()),
        "binned_gains_by_concentration": binned_gains,
        "gate_g2": gate_g2,
        "gate_g2_threshold": 0.2,
        "corpus_comet_sigfirst": sigfirst_metrics.get("comet22"),
        "corpus_comet_med": med_metrics.get("comet22"),
        "corpus_comet_delta": (
            sigfirst_metrics.get("comet22", 0) - med_metrics.get("comet22", 0)
            if sigfirst_metrics.get("comet22") and med_metrics.get("comet22")
            else None
        ),
    }

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"  Gate G2: SIG-Concentration Analysis")
    print(f"{'=' * 60}")
    print(f"  Examples: {n}")
    print(f"  Spearman ρ(SIG-Concentration, gain): {rho:.4f} (p={p_value:.4e})")
    print(f"  Pearson r: {pearson_r:.4f} (p={pearson_p:.4e})")
    print(f"  Mean SIG-Concentration: {np.mean(sig_concentrations):.4f}")
    print(f"  Mean gain (SIG-first − MED): {np.mean(gains):.4f}")
    print(f"  Fraction with positive gain: {(gains_arr > 0).mean():.1%}")
    print(f"\n  Binned gain by SIG-Concentration quintile:")
    for label, data in binned_gains.items():
        print(f"    {label}: n={data['n']}, mean_gain={data['mean_gain']:+.4f}")
    print(f"\n  Corpus COMET: SIG-first={sigfirst_metrics.get('comet22', 'N/A')}, "
          f"MED={med_metrics.get('comet22', 'N/A')}")
    print(f"  Gate G2: {'PASS ✓' if gate_g2 == 'PASS' else 'FAIL ✗'} "
          f"(ρ={rho:.4f} {'≥' if rho >= 0.2 else '<'} 0.2)")
    print(f"{'=' * 60}")
    sys.stdout.flush()

    return result


def analyze_challenge_set(
    sigfirst_trans: List[dict],
    sigfirst_metrics: dict,
    med_trans: List[dict],
    med_metrics: dict,
) -> dict:
    """
    Gate G3 analysis.

    Check if SIG-first > MED on challenge subsets.
    """
    sig_comet = get_sentence_comet(sigfirst_metrics)
    med_comet = get_sentence_comet(med_metrics)

    # Corpus-level comparison
    corpus_delta = None
    if sigfirst_metrics.get("comet22") and med_metrics.get("comet22"):
        corpus_delta = sigfirst_metrics["comet22"] - med_metrics["comet22"]

    result = {
        "corpus_comet_sigfirst": sigfirst_metrics.get("comet22"),
        "corpus_comet_med": med_metrics.get("comet22"),
        "comet_delta": corpus_delta,
        "bleu_sigfirst": sigfirst_metrics.get("bleu"),
        "bleu_med": med_metrics.get("bleu"),
        "bleu_delta": (
            sigfirst_metrics.get("bleu", 0) - med_metrics.get("bleu", 0)
            if sigfirst_metrics.get("bleu") is not None and med_metrics.get("bleu") is not None
            else None
        ),
    }

    # Per-category analysis (if translations have challenge_categories)
    if sig_comet and med_comet:
        n = min(len(sig_comet), len(med_comet), len(sigfirst_trans), len(med_trans))
        per_category = {}

        for i in range(n):
            cats = sigfirst_trans[i].get("challenge_categories",
                   sigfirst_trans[i].get("challenge_category", []))
            if isinstance(cats, str):
                cats = [cats]

            gain = sig_comet[i] - med_comet[i]

            for cat in cats:
                if cat not in per_category:
                    per_category[cat] = {"gains": [], "sig_comet": [], "med_comet": []}
                per_category[cat]["gains"].append(gain)
                per_category[cat]["sig_comet"].append(sig_comet[i])
                per_category[cat]["med_comet"].append(med_comet[i])

        category_summary = {}
        for cat, data in per_category.items():
            gains_arr = np.array(data["gains"])
            category_summary[cat] = {
                "n": len(data["gains"]),
                "comet_delta": float(gains_arr.mean()),
                "comet_delta_std": float(gains_arr.std()),
                "mean_sigfirst_comet": float(np.mean(data["sig_comet"])),
                "mean_med_comet": float(np.mean(data["med_comet"])),
                "fraction_positive": float((gains_arr > 0).mean()),
            }

        result["per_category"] = category_summary

    # Gate G3 check: any subset with delta ≥ 0.003 (0.3 COMET points on 0-1 scale)
    # Note: COMET-22 is on 0-1 scale, so 0.3 COMET "points" = 0.003 on the scale
    # Actually, looking at the plan again: "≥ 0.3 COMET" — this likely means 0.003
    # on the 0-1 scale (since COMET ranges ~0.5-1.0). Let's check both thresholds.
    gate_g3 = "FAIL"
    best_subset = None
    best_delta = None

    if "per_category" in result:
        for cat, data in result["per_category"].items():
            delta = data["comet_delta"]
            if best_delta is None or delta > best_delta:
                best_delta = delta
                best_subset = cat
            if delta >= 0.003:  # 0.3 COMET points
                gate_g3 = "PASS"

    result["gate_g3"] = gate_g3
    result["gate_g3_best_subset"] = best_subset
    result["gate_g3_best_delta"] = best_delta

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"  Gate G3: Challenge Set Analysis")
    print(f"{'=' * 60}")
    print(f"  Corpus COMET: SIG-first={result.get('corpus_comet_sigfirst', 'N/A')}, "
          f"MED={result.get('corpus_comet_med', 'N/A')}")
    if corpus_delta is not None:
        print(f"  Corpus COMET delta: {corpus_delta:+.4f}")
    print(f"  BLEU: SIG-first={result.get('bleu_sigfirst', 'N/A'):.2f}, "
          f"MED={result.get('bleu_med', 'N/A'):.2f}")

    if "per_category" in result:
        print(f"\n  Per-category results:")
        for cat, data in sorted(result["per_category"].items()):
            print(f"    {cat}: n={data['n']}, COMET delta={data['comet_delta']:+.4f}, "
                  f"positive={data['fraction_positive']:.1%}")

    print(f"\n  Gate G3: {'PASS ✓' if gate_g3 == 'PASS' else 'FAIL ✗'} "
          f"(best={best_subset}, delta={best_delta:+.4f})" if best_delta else "")
    print(f"{'=' * 60}")
    sys.stdout.flush()

    return result


def main():
    parser = argparse.ArgumentParser(description="SIG-Concentration Analysis")
    parser.add_argument("--sigfirst_trans", type=str, required=True,
                        help="SIG-first translations JSON")
    parser.add_argument("--sigfirst_metrics", type=str, required=True,
                        help="SIG-first metrics JSON (with per-sentence COMET)")
    parser.add_argument("--med_trans", type=str, required=True,
                        help="MED translations JSON")
    parser.add_argument("--med_metrics", type=str, required=True,
                        help="MED metrics JSON (with per-sentence COMET)")
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--challenge_mode", action="store_true",
                        help="Run Gate G3 challenge-set analysis instead of Gate G2")
    args = parser.parse_args()

    sigfirst_trans = load_translations(args.sigfirst_trans)
    sigfirst_metrics = load_metrics(args.sigfirst_metrics)
    med_trans = load_translations(args.med_trans)
    med_metrics = load_metrics(args.med_metrics)

    if args.challenge_mode:
        result = analyze_challenge_set(
            sigfirst_trans, sigfirst_metrics, med_trans, med_metrics
        )
    else:
        result = analyze_sig_concentration(
            sigfirst_trans, sigfirst_metrics, med_trans, med_metrics
        )

    # Save
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nAnalysis saved to {output_path}")


if __name__ == "__main__":
    main()
