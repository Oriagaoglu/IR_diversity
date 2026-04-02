# LLM Judge Validation

## Overview

Our primary metric, Coverage@K, relies on an LLM judge (Claude Haiku) to determine whether a paragraph covers a perspective claim. This document describes how we validated that judge against human annotations.

## Annotation Setup

**Task:** Given a perspective claim and a paragraph from a news article, annotators answer: *does this paragraph express or support this claim?* (Yes/No)

**Sample:** 100 paragraph-claim pairs, stratified to include 50 LLM-positive and 50 LLM-negative pairs to ensure sufficient representation of both classes. Pairs were drawn from 64 events across the full DiverseSumm benchmark.

**Annotators:** 4 group members (Orhan, Melani, Constantinos, Zhiyong) independently annotated all 100 pairs. Annotators were blind to the LLM labels.

**Tool:** Custom PyQt6 annotation app (`scripts/annotator_app.py`) displaying claim and paragraph side by side, with keyboard shortcuts for rapid annotation.

## Inter-Annotator Agreement

| Pair | Cohen's κ | Interpretation |
|------|-----------|----------------|
| Orhan – Melani | 0.411 | Moderate |
| Orhan – Constantinos | 0.361 | Fair |
| Melani – Zhiyong | 0.297 | Fair |
| Melani – Constantinos | 0.197 | Slight |
| Orhan – Zhiyong | 0.175 | Slight |
| Constantinos – Zhiyong | 0.077 | Slight |

**Fleiss' κ (all 4 annotators): 0.250 (fair)**

The task is inherently subjective: whether a paragraph "supports" a claim involves judgment about framing, implication, and emphasis. Annotators agreed unanimously on only 32 of 100 pairs. This level of agreement is consistent with other perspective/stance annotation tasks in NLP, where fine-grained framing distinctions are notoriously difficult to adjudicate.

## LLM Judge vs Human Majority Vote

We compare the LLM judge's labels against the human majority vote (≥3 of 4 annotators agree).

|  | Human Yes | Human No |
|--|-----------|----------|
| **LLM Yes** | 29 | 21 |
| **LLM No** | 15 | 35 |

- **Cohen's κ (LLM vs human majority): 0.280 (fair)**
- Precision: 58.0% (when LLM says covered, humans agree 58% of the time)
- Recall: 65.9% (when humans say covered, LLM catches 66%)
- F1: 0.617

## Interpretation

The LLM judge achieves κ = 0.280 against human majority vote, which is comparable to the mean pairwise agreement among human annotators themselves (κ = 0.253). This means the LLM performs at the level of an individual human annotator on this task — it is no worse than replacing one group member with the LLM.

The moderate disagreement reflects the genuine difficulty of the task, not a failure of the LLM. Perspective coverage is not a binary factual question — it requires interpreting whether a paragraph's framing aligns with a specific claim, which reasonable annotators disagree on.

Critically, the same LLM labels are used to evaluate **all** methods in our study. Any noise in Coverage@K affects all methods equally and does not systematically favor one pipeline over another. The relative ordering of methods — which is what our research questions address — is robust to label noise because it is measured through paired statistical tests (Wilcoxon signed-rank) on per-event scores.

## Files

| File | Description |
|------|-------------|
| `data/annotations/annotator_orhan.csv` | Orhan's 100 annotations |
| `data/annotations/annotator_melani.csv` | Melani's 100 annotations |
| `data/annotations/annotator_constantinos.csv` | Constantinos's 100 annotations |
| `data/annotations/annotator_zhiyong.csv` | Zhiyong's 100 annotations |
| `data/annotations/annotations_combined.csv` | Combined annotations with majority vote |
| `data/processed/annotation_100pairs.json` | The 100 pairs (blind, no LLM labels) |
| `data/processed/annotation_llm_ground_truth.json` | LLM labels for the 100 pairs |
| `scripts/annotator_app.py` | Annotation tool (PyQt6) |
| `scripts/compute_annotation_agreement.py` | Agreement computation script |
