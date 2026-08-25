# Annotation Guidelines — Human Evaluation of Machine Translation

*This is a direction-agnostic English version of the guideline, suitable for any language pair. The Chinese-language guidelines actually distributed to annotators in the paper (one per direction, each with direction-specific scoring tips) are also included in this directory: `annotation_guidelines_enzh_zh.md` for **En→Zh** and `annotation_guidelines_zhen_zh.md` for **Zh→En**. Adapt the language names where appropriate when applying this template to other directions.*

## Task overview

You will evaluate 100 machine-translated sentences. Each sentence is presented with two system outputs labelled **System A** and **System B**. You will:

1. Give each system a **1–5 Adequacy** score (does the translation convey the source meaning?).
2. Give each system a **1–5 Fluency** score (is the target-language output natural?).
3. Indicate an **overall Preference**: A / B / Tie.
4. Optionally leave a Note on common error types.

The mapping from System A/B to the actual systems is **hidden** — please do not try to guess which is which. A random independent flip is applied per sentence, so A in one row may be a different system than A in another.

If you notice two translations look equally good (or bad), use **Tie**. Tied is a legitimate, useful signal.

---

## Rating scales

### Adequacy (1–5)

Measures whether all information in the source is faithfully rendered.

| Score | Meaning |
|-------|---------|
| 5 | All source meaning is present. No omissions, no hallucinations, no mistranslations. |
| 4 | Nearly all meaning is preserved. A few minor omissions or slight imprecision that do not hinder comprehension. |
| 3 | The main meaning is present, but there are noticeable omissions, minor mistranslations, or partial information loss. |
| 2 | Only a small fraction of the source meaning is correctly conveyed. Significant mistranslations or major omissions. |
| 1 | The translation is essentially unrelated to the source or incomprehensible. |

Scoring tips:
- Focus on **information completeness**: do all entities, numbers, and relations in the source appear in the translation?
- Focus on **accuracy**: are nouns, verbs and modifiers mapped correctly?
- Do **not** deduct points for a different style from the reference — a different-but-correct translation is fine.

### Fluency (1–5)

Measures naturalness and grammaticality of the target-language output, independent of whether it matches the source.

| Score | Meaning |
|-------|---------|
| 5 | Natural and idiomatic. Reads as if written by a fluent native speaker. |
| 4 | Generally fluent. Occasional minor unnaturalness (e.g., slightly awkward word order) that does not impede reading. |
| 3 | Understandable but with multiple unnatural constructions or grammatical oddities. |
| 2 | Hard to read. Many grammar or word-choice errors require multiple readings to understand. |
| 1 | Unreadable. Severe grammatical errors, garbled output, or incoherent word sequences. |

Scoring tips:
- **Grammaticality**: agreement, proper particle usage, correct measure words.
- **Naturalness**: avoid translationese (e.g., overuse of passive voice, unnaturally long sentences).
- **Word choice**: are specialised terms rendered with appropriate target-language equivalents?
- Fluency is **independent of Adequacy**: a translation can be very fluent but miss meaning (high fluency, low adequacy), or information-complete but awkward (high adequacy, low fluency).

### Overall Preference (A / B / Tie)

Combining Adequacy and Fluency, which translation do you prefer?

| Choice | Meaning |
|--------|---------|
| A | System A's translation is clearly better (considering both adequacy and fluency). |
| B | System B's translation is clearly better. |
| Tie | The two are of roughly equal quality, with no clear preference. |

Decision rules:
- If one system is better on both dimensions → pick that system.
- If one is more adequate but less fluent than the other → use your holistic judgment.
- If the difference is small (e.g., 4/4 vs 4/5) → Tie is usually correct.

---

## Process per sentence

1. Read the **Source** carefully and form an understanding of the intended meaning.
2. Read the **Reference** translation — but remember the reference is only one valid translation, not the sole truth.
3. Rate **System A**: Adequacy, then Fluency.
4. Rate **System B**: Adequacy, then Fluency.
5. Choose **Preference**: A / B / Tie.
6. (Optional) Record any notable observations in the Notes column.

---

## Edge cases

- **Empty output**: if a translation is entirely empty, score Adequacy 1 and Fluency 1.
- **Repetitions** (same phrase repeated many times): deduct under Fluency.
- **Correct meaning in a very different style** (e.g., rephrased, different particle choice): do not penalise — different styles can both be correct.
- **Exact match with reference**: full marks on Adequacy are fine even if Fluency is imperfect.

---

## Time estimate

- Roughly 1–2 minutes per sentence.
- 100 sentences ≈ 2–3 hours total.
- We recommend splitting across 2–3 sessions to avoid fatigue.

---

## Submission

Save the filled CSV and return it. Aggregation, anonymisation and de-blinding will be handled by the researchers.
