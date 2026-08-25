# Human evaluation template (two-direction protocol)

This directory provides the **protocol, annotator guidelines, and form templates** used in the paper's two-direction three-expert human evaluation (**En→Zh** and **Zh→En**, 100 sentences each, the same three bilingual annotators across both directions), so future work can reproduce a directly comparable study.

The annotator guidelines come in **two complementary layers**, not three redundant copies: `annotation_guidelines.md` is a *direction-agnostic English template* intended for researchers reproducing the protocol on other language pairs, while `annotation_guidelines_enzh_zh.md` (En→Zh) and `annotation_guidelines_zhen_zh.md` (Zh→En) are the *Chinese, direction-specific guidelines actually distributed to the three bilingual annotators in the paper*, retained verbatim for reproducibility. The Chinese versions add per-direction scoring tips (e.g.\ 量词 and 「的地得」 for En→Zh; 冠词, 时态, and 主谓一致 for Zh→En) that the agnostic template intentionally omits.

> **No individual annotator scores from the paper are released here** for privacy reasons. The paper reports only aggregated statistics (per-direction 3-expert mean Δ-Adequacy, Δ-Fluency, Fleiss κ, Kendall W, Spearman ρ with COMET-22, and bucket-conditioned summaries). The actual system-to-slot mapping used in the paper is also not released; `system_mapping_example.json` is a small fictitious example.

## Files

| File | Purpose |
|------|---------|
| `evaluation_form_template.csv` | 100-row blank CSV template. Direction-agnostic: the `Source` and `Reference` columns hold whichever language is the source / reference for that direction. Annotators fill in the five rating columns plus a free-text `Notes` field. |
| `annotation_guidelines.md` | English-language, direction-agnostic guideline (1–5 Likert scales for Adequacy and Fluency, A/B/Tie Preference, blinding protocol, edge cases). |
| `annotation_guidelines_enzh_zh.md` | Chinese-language **En→Zh** guideline as distributed to annotators, with En→Zh-specific scoring tips (e.g. 量词, 「的地得」用法). |
| `annotation_guidelines_zhen_zh.md` | Chinese-language **Zh→En** guideline as distributed to annotators, with Zh→En-specific scoring tips (e.g. 英文时态, 冠词, 主谓一致). |
| `system_mapping_example.json` | Fictitious example illustrating the per-row system-to-slot mapping plus the stratification bucket label (`EV-best`, `Ratio-best`, `Tied`, `Random`). The real mapping from the paper is **not** released. |

## Stratification (per direction)

Following the paper, for each direction we stratify the test set (WMT22, N=2037) by the EV − Ratio ΔCOMET-22 per sentence and draw 100 items:

| Bucket      | N  | Criterion                                           |
|-------------|----|-----------------------------------------------------|
| EV-best     | 25 | Top-25 largest positive ΔCOMET (EV clearly better)  |
| Ratio-best  | 25 | Bottom-25 (most negative ΔCOMET, Ratio clearly better) |
| Tied        | 25 | 25 smallest **non-zero** \|ΔCOMET\| (excludes byte-identical outputs) |
| Random      | 25 | Uniform random draw from the remaining indices      |

This design stress-tests three concerns simultaneously: (i) whether humans agree with COMET on clear wins (EV-best, Ratio-best); (ii) whether COMET ties are perceived as real ties (Tied); and (iii) the natural-distribution baseline (Random).

## Blinding protocol

Per row, an **independent Bernoulli(0.5) flip** decides which system is rendered as `System A` and which as `System B`. Within a 100-sentence pack this yields an approximately 50/50 A/B distribution for each method, but the flip is independent across rows so a system's identity cannot be inferred from row order. Rows within each pack are additionally shuffled so bucket boundaries are not visible to the annotator.

Annotators never see: the stratification bucket, the ΔCOMET, the A/B flip, or the mapping to EV / Ratio. Only the paper authors hold the de-blinding mapping, and they apply it only after all three annotators have returned their filled CSVs.

## Workflow

1. **Generate system outputs** for the two length methods you want to compare (typically Entropy-Valley and a fixed-ratio baseline) on the same test split.
2. **Stratify**: for each direction, compute per-sentence ΔCOMET (or your automatic metric of choice), draw 100 items using the four-bucket rule above, and write the mapping (bucket, system-to-slot, source index) to a private JSON mirroring `system_mapping_example.json`.
3. **Shuffle** row order within each pack so bucket labels are not visible.
4. **Hand each annotator** the same blank `evaluation_form_template.csv` (with columns 1–5 pre-filled: ID, Source, Reference, System A, System B) and the direction-appropriate guideline. Annotators work independently, on disjoint passes, with no cross-annotator discussion during rating.
5. **After all three annotations return**, de-blind using the saved mapping and compute per-annotator and aggregated statistics, matching the paper's schema (per-annotator Δ-Adequacy, Δ-Fluency, Wilcoxon p; aggregated means, majority preference and sign test, Fleiss κ, Kendall W, Spearman ρ against ΔCOMET).

See `ladit/evaluation/significance_test.py` in the top-level repository for a reference implementation of the paired bootstrap test used for the automatic metrics, and the same directory for the sentence-level Wilcoxon / Spearman helpers used in the human-eval aggregation.

## What the paper uses this template for (reference)

The paper runs the protocol twice (En→Zh and Zh→En), with the same three Chinese–English bilingual annotators (anonymised as A, B, C) on separate days for the two directions. The aggregated quantities — three-expert mean $\Delta$-Adequacy (EV − Ratio), paired Wilcoxon $p$ on adequacy, majority preference counts and sign-test $p$, Fleiss $\kappa$ on the three-class preference, and Spearman $\rho$ between sentence-level $\Delta$COMET and $\Delta$Adequacy — are reported in the paper's Table 3 (headline) and Appendix Table `tab:human_eval_details` (per-annotator and per-direction). Raw per-row annotator scores are not included in this directory and remain with the paper authors.
