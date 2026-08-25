"""
Statistical significance testing for LaDiT experiments.

Implements paired bootstrap resampling (WMT standard) and Wilcoxon signed-rank test
for comparing per-sentence COMET scores between methods.

Usage:
    python -m ladit.evaluation.significance_test \
        --enzh_scores eval_results/ladit_enzh_seed42/per_sentence_comet.json \
        --ende_scores eval_results/ladit_ende_seed42/per_sentence_comet.json \
        --output      eval_results/significance_results.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats


def paired_bootstrap_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> dict:
    """Paired bootstrap resampling test (Koehn 2004, WMT standard).

    Tests H0: mean(scores_a) <= mean(scores_b).
    Returns p-value = fraction of bootstrap samples where A does NOT beat B.
    """
    rng = np.random.RandomState(seed)
    n = len(scores_a)
    diff = scores_a - scores_b
    observed_diff = diff.mean()

    count_a_better = 0
    boot_diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        boot_diff = diff[idx].mean()
        boot_diffs[i] = boot_diff
        if boot_diff > 0:
            count_a_better += 1

    p_value = 1.0 - count_a_better / n_bootstrap
    ci_lower, ci_upper = np.percentile(boot_diffs, [2.5, 97.5])

    return {
        "observed_diff": float(observed_diff),
        "p_value": float(p_value),
        "ci_95_lower": float(ci_lower),
        "ci_95_upper": float(ci_upper),
        "n_bootstrap": n_bootstrap,
        "significant_p05": bool(p_value < 0.05),
        "significant_p01": bool(p_value < 0.01),
    }


def wilcoxon_test(scores_a: np.ndarray, scores_b: np.ndarray) -> dict:
    """Wilcoxon signed-rank test (non-parametric paired test)."""
    statistic, p_value = stats.wilcoxon(scores_a, scores_b)
    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "significant_p05": bool(p_value < 0.05),
        "significant_p01": bool(p_value < 0.01),
    }


def effect_size_cohens_d(scores_a: np.ndarray, scores_b: np.ndarray) -> dict:
    """Paired Cohen's d effect size."""
    diff = scores_a - scores_b
    d = diff.mean() / diff.std() if diff.std() > 0 else 0.0
    magnitude = "negligible" if abs(d) < 0.2 else "small" if abs(d) < 0.5 else "medium" if abs(d) < 0.8 else "large"
    return {"cohens_d": float(d), "magnitude": magnitude}


def win_loss_tie(scores_a: np.ndarray, scores_b: np.ndarray, tol: float = 1e-6) -> dict:
    """Count wins, losses, and ties."""
    diff = scores_a - scores_b
    wins = int((diff > tol).sum())
    losses = int((diff < -tol).sum())
    ties = int(len(diff) - wins - losses)
    n = len(diff)
    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_pct": float(wins / n * 100),
        "loss_pct": float(losses / n * 100),
        "tie_pct": float(ties / n * 100),
    }


def compare_pair(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    label_a: str,
    label_b: str,
    n_bootstrap: int = 10000,
) -> dict:
    """Full comparison between two score arrays."""
    result = {
        "system_a": label_a,
        "system_b": label_b,
        "n": len(scores_a),
        "mean_a": float(scores_a.mean()),
        "mean_b": float(scores_b.mean()),
        "mean_diff": float((scores_a - scores_b).mean()),
        "bootstrap": paired_bootstrap_test(scores_a, scores_b, n_bootstrap),
        "wilcoxon": wilcoxon_test(scores_a, scores_b),
        "effect_size": effect_size_cohens_d(scores_a, scores_b),
        "win_loss_tie": win_loss_tie(scores_a, scores_b),
    }
    return result


def analyze_lang_pair(scores_dict: dict, lang: str) -> dict:
    """Analyze all pairwise comparisons for one language pair."""
    if lang == "enzh":
        ev = np.array(scores_dict["ev_scores"])
        ratio = np.array(scores_dict["ratio_scores"])
        oracle = np.array(scores_dict["oracle_scores"])
        ratio_label = "ratio_0.8"
    else:
        ev = np.array(scores_dict["entropy_valley"])
        ratio = np.array(scores_dict["ratio_1.8"])
        oracle = np.array(scores_dict["oracle"])
        ratio_label = "ratio_1.8"

    results = {
        "lang_pair": "En→Zh" if lang == "enzh" else "En→De",
        "n_sentences": len(ev),
        "ev_vs_ratio": compare_pair(ev, ratio, "EV", ratio_label),
        "oracle_vs_ev": compare_pair(oracle, ev, "oracle", "EV"),
        "oracle_vs_ratio": compare_pair(oracle, ratio, "oracle", ratio_label),
    }
    return results


def print_results(results: dict):
    """Pretty-print significance test results."""
    for lang_key in ["enzh", "ende"]:
        if lang_key not in results:
            continue
        r = results[lang_key]
        print(f"\n{'=' * 70}")
        print(f"  {r['lang_pair']} — Statistical Significance Tests (N={r['n_sentences']})")
        print(f"{'=' * 70}")

        for comp_key, comp_label in [
            ("ev_vs_ratio", "EV vs Ratio (main claim)"),
            ("oracle_vs_ev", "Oracle vs EV"),
            ("oracle_vs_ratio", "Oracle vs Ratio"),
        ]:
            c = r[comp_key]
            print(f"\n  --- {comp_label}: {c['system_a']} vs {c['system_b']} ---")
            print(f"  Mean: {c['system_a']}={c['mean_a']:.4f}, {c['system_b']}={c['mean_b']:.4f}, diff={c['mean_diff']:+.4f}")
            bs = c["bootstrap"]
            print(f"  Bootstrap: p={bs['p_value']:.6f}, 95% CI [{bs['ci_95_lower']:.4f}, {bs['ci_95_upper']:.4f}]")
            wt = c["wilcoxon"]
            print(f"  Wilcoxon:  p={wt['p_value']:.2e}")
            es = c["effect_size"]
            print(f"  Effect:    Cohen's d={es['cohens_d']:.4f} ({es['magnitude']})")
            wlt = c["win_loss_tie"]
            print(f"  Win/Loss/Tie: {wlt['wins']}/{wlt['losses']}/{wlt['ties']} ({wlt['win_pct']:.1f}%/{wlt['loss_pct']:.1f}%/{wlt['tie_pct']:.1f}%)")

            sig_marker = ""
            if bs["significant_p01"]:
                sig_marker = "*** (p<0.01)"
            elif bs["significant_p05"]:
                sig_marker = "**  (p<0.05)"
            else:
                sig_marker = "    (n.s.)"
            print(f"  Verdict:   {sig_marker}")

    print(f"\n{'=' * 70}")
    print("  Paper-ready LaTeX table:")
    print(f"{'=' * 70}")
    print_latex_table(results)
    sys.stdout.flush()


def print_latex_table(results: dict):
    """Generate LaTeX table for paper."""
    print(r"  \begin{tabular}{lcccccc}")
    print(r"  \toprule")
    print(r"  & \multicolumn{3}{c}{En$\to$Zh} & \multicolumn{3}{c}{En$\to$De} \\")
    print(r"  \cmidrule(lr){2-4} \cmidrule(lr){5-7}")
    print(r"  Comparison & $\Delta$COMET & 95\% CI & $p$ & $\Delta$COMET & 95\% CI & $p$ \\")
    print(r"  \midrule")

    for comp_key, label in [
        ("ev_vs_ratio", r"EV $>$ Ratio"),
        ("oracle_vs_ev", r"Oracle $>$ EV"),
        ("oracle_vs_ratio", r"Oracle $>$ Ratio"),
    ]:
        parts = [f"  {label}"]
        for lang in ["enzh", "ende"]:
            if lang in results:
                c = results[lang][comp_key]
                bs = c["bootstrap"]
                diff = c["mean_diff"]
                ci_lo = bs["ci_95_lower"]
                ci_hi = bs["ci_95_upper"]
                p = bs["p_value"]
                p_str = f"{p:.4f}" if p >= 0.0001 else r"$<$0.0001"
                parts.append(f"{diff:+.4f}")
                parts.append(f"[{ci_lo:.4f}, {ci_hi:.4f}]")
                parts.append(p_str)
        print(" & ".join(parts) + r" \\")

    print(r"  \bottomrule")
    print(r"  \end{tabular}")


def main():
    parser = argparse.ArgumentParser(description="Statistical significance tests for LaDiT")
    parser.add_argument("--enzh_scores", type=str,
                        help="Per-sentence COMET JSON for En→Zh")
    parser.add_argument("--ende_scores", type=str,
                        help="Per-sentence COMET JSON for En→De")
    parser.add_argument("--output", type=str, default="eval_results/significance_results.json")
    parser.add_argument("--n_bootstrap", type=int, default=10000)
    args = parser.parse_args()

    results = {}

    if args.enzh_scores:
        with open(args.enzh_scores) as f:
            enzh = json.load(f)
        results["enzh"] = analyze_lang_pair(enzh, "enzh")

    if args.ende_scores:
        with open(args.ende_scores) as f:
            ende = json.load(f)
        results["ende"] = analyze_lang_pair(ende, "ende")

    print_results(results)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
