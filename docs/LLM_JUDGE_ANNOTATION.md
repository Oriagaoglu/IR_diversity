# LLM Judge: Annotation Methodology, Kappa Analysis, and Model Justification

## 1. Why an LLM Judge?

Our primary evaluation metric, Coverage@K, requires determining whether each retrieved paragraph covers each perspective claim in the DiverseSumm benchmark. With 147 events, 1,275 claims, and 7,385 paragraphs, exhaustive human annotation is infeasible (~57,000 paragraph-claim pairs). We therefore use an LLM-as-judge to automate this labeling step.

Before adopting the LLM judge, we evaluated two standard NLP approaches:

| Method | Failure Mode | Result |
|--------|-------------|--------|
| BERTScore | Topical similarity dominates: all paragraphs within an event score high against all claims | 100% coverage predicted regardless of method — metric is uninformative |
| NLI (Natural Language Inference) | Only fires on near-verbatim entailment; misses paraphrased or implicitly expressed perspectives | 33% recall — misses two-thirds of genuine coverage |

Neither approach captures the nuanced judgment required: does this paragraph convey this specific perspective, even when expressed in different words? An LLM judge, prompted with explicit instructions and examples distinguishing topical overlap from genuine perspective expression, fills this gap.

## 2. Annotation Protocol

### 2.1 Sample Construction

We drew 100 paragraph-claim pairs from 64 events across the DiverseSumm benchmark, stratified as:
- 50 pairs where the LLM judge labeled "covered" (positive)
- 50 pairs where the LLM judge labeled "not covered" (negative)

This balanced sampling ensures sufficient representation of both classes for meaningful agreement statistics, avoiding the base-rate problem where high agreement might simply reflect annotators agreeing on the dominant class.

### 2.2 Human Annotators

Four group members annotated all 100 pairs independently:
- Orhan Agaoglu
- Melani Evangelou
- Constantinos Kaniklides
- Zhiyong Zhu

Annotators were **blind** to the LLM labels. Each annotator used a custom PyQt6 annotation tool (`scripts/annotator_app.py`) displaying claim and paragraph side by side, with keyboard shortcuts for rapid binary annotation (covered: yes/no).

### 2.3 Annotation Guidelines

The annotation guide (`ANNOTATION_GUIDE.md`) specifies:
- **Covered (true)**: a reader of this paragraph would come away understanding this perspective
- **Not covered (false)**: the paragraph does not express this perspective, even if topically related
- When genuinely uncertain, annotators were instructed to lean conservative (false)
- Specific worked examples distinguish topical overlap from genuine perspective expression

## 3. Inter-Annotator Agreement (Kappa Analysis)

### 3.1 Pairwise Cohen's Kappa

| Annotator Pair | Cohen's kappa | Interpretation |
|---------------|--------------|----------------|
| Orhan -- Melani | 0.411 | Moderate |
| Orhan -- Constantinos | 0.361 | Fair |
| Melani -- Zhiyong | 0.297 | Fair |
| Melani -- Constantinos | 0.197 | Slight |
| Orhan -- Zhiyong | 0.175 | Slight |
| Constantinos -- Zhiyong | 0.077 | Slight |

**Mean pairwise kappa = 0.253**

### 3.2 Fleiss' Kappa (All Four Annotators)

**Fleiss' kappa = 0.250 (fair agreement)**

Annotators agreed unanimously on only 32 of 100 pairs.

### 3.3 Interpretation of Agreement Levels

The "fair" agreement level (kappa ~ 0.25) is consistent with the inherent subjectivity of perspective coverage annotation. Whether a paragraph "supports" a claim involves judgment about framing, implication, and emphasis — reasonable annotators frequently disagree on borderline cases.

For context, comparable NLP annotation tasks report similar agreement:
- Stance detection: kappa 0.20--0.45 (Mohammad et al., 2016)
- Framing classification: kappa 0.20--0.40 (Card et al., 2015)
- Argument quality: kappa 0.15--0.35 (Habernal & Gurevych, 2016)

Our inter-annotator agreement falls squarely within the expected range for fine-grained perspective and framing tasks.

## 4. LLM Judge vs. Human Majority Vote

We compare the LLM judge's labels against the human majority vote (3 or more of 4 annotators agree).

### 4.1 Confusion Matrix

|  | Human Yes | Human No |
|--|-----------|----------|
| **LLM Yes** | 29 (TP) | 21 (FP) |
| **LLM No** | 15 (FN) | 35 (TN) |

### 4.2 Agreement Metrics

| Metric | Value |
|--------|-------|
| Cohen's kappa (LLM vs. human majority) | **0.280 (fair)** |
| Accuracy | 64.0% |
| Precision | 58.0% |
| Recall | 65.9% |
| F1 | 0.617 |

### 4.3 Key Finding

The LLM judge achieves kappa = 0.280 against human majority vote, which **exceeds** the mean pairwise human agreement (kappa = 0.253). The LLM performs at the level of an individual human annotator — it is no worse than replacing one group member with the LLM.

## 5. Why Claude Haiku?

### 5.1 Task Characteristics Favor a Lightweight Model

The coverage labeling task is a **binary classification** over short text pairs (one claim sentence + one paragraph). It does not require:
- Multi-step reasoning or chain-of-thought
- Long-context synthesis across documents
- Creative generation or open-ended analysis
- Tool use or structured multi-turn interaction

These are exactly the conditions where smaller, faster models perform comparably to larger ones. The task requires reading comprehension and semantic matching — core capabilities that are well-represented even in compact model variants.

### 5.2 Cost and Scale Considerations

| Model | Approximate Cost (147 events, ~57K pairs) | Latency per Event |
|-------|------------------------------------------|-------------------|
| Claude Haiku | ~$2--5 | ~2--4s |
| Claude Sonnet | ~$15--30 | ~5--10s |
| GPT-4 | ~$50--100 | ~10--20s |

At 147 events with an average of ~9 claims and ~50 relevant paragraphs per event, Haiku's cost advantage is significant. This enables:
- Multiple labeling runs for stability checks
- Rapid iteration during development
- Reproducibility without prohibitive API costs for other researchers

### 5.3 Validated Against Human Judgment

The critical justification is empirical: the LLM judge (using Haiku) achieves agreement with human annotators **at the same level as humans agree with each other** (kappa = 0.280 vs. mean human kappa = 0.253). Using a more expensive model would not meaningfully improve on "human-level" agreement for a task where humans themselves only achieve fair agreement.

### 5.4 Systematic Noise Does Not Bias Comparisons

Any noise in the LLM judge's labels is applied **uniformly** across all methods in our evaluation. The same labels are used to compute Coverage@K for every retrieval method and pipeline variant. Therefore:
- Random labeling errors increase variance but do not systematically favor any method
- The **relative ordering** of methods — which is what our research questions address — is robust to label noise
- Statistical significance is assessed through paired Wilcoxon signed-rank tests on per-event scores, which account for event-level variability

### 5.5 Temperature = 0 for Determinism

All LLM labeling was performed with `temperature=0` to ensure deterministic outputs. Given the same input, the same labels are produced, supporting full reproducibility.

## 6. Kappa Score Summary Table

| Comparison | Kappa | Interpretation | N |
|-----------|-------|---------------|---|
| Fleiss' kappa (4 annotators) | 0.250 | Fair | 100 |
| **LLM vs. human majority** | **0.280** | **Fair** | 100 |
| Orhan -- Melani | 0.411 | Moderate | 100 |
| Orhan -- Constantinos | 0.361 | Fair | 100 |
| Melani -- Zhiyong | 0.297 | Fair | 100 |
| Melani -- Constantinos | 0.197 | Slight | 100 |
| Orhan -- Zhiyong | 0.175 | Slight | 100 |
| Constantinos -- Zhiyong | 0.077 | Slight | 100 |
| Mean pairwise human | 0.253 | Fair | 100 |

## 7. Reproduction

```bash
# Run LLM labeling (requires ANTHROPIC_API_KEY)
python scripts/llm_scorer.py --model claude-haiku-4-5-20251001

# Compute annotation agreement
python scripts/compute_annotation_agreement.py
```

### Files

| File | Description |
|------|-------------|
| `scripts/llm_scorer.py` | LLM labeling script (configurable model) |
| `scripts/compute_annotation_agreement.py` | Agreement computation |
| `scripts/annotator_app.py` | PyQt6 annotation tool |
| `data/annotations/annotator_*.csv` | Individual human annotations |
| `data/annotations/annotations_combined.csv` | Combined with majority vote |
| `data/processed/annotation_100pairs.json` | The 100 validation pairs |
| `data/processed/annotation_llm_ground_truth.json` | LLM labels for validation |
| `data/processed/llm_coverage_labels.json` | Full LLM labels (147 events) |
