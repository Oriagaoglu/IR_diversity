# IR Diversity Project — Findings & Progress Report

## Project Overview

We evaluate NEWSCOPE's geometric diversity metrics (APD, PCC, IDR) as proxies for **perspective coverage** in multi-perspective news retrieval. We propose **Coverage@K** — the fraction of human-identified perspective claims covered by the top-K retrieved paragraphs — as a ground-truth metric grounded in DiverseSumm annotations.

**Research Questions:**
- **RQ1**: Do NEWSCOPE's geometric metrics (APD, PCC) predict actual perspective coverage?
- **RQ2**: Which diversity mechanism best maximizes perspective coverage?

**Dataset:** 147 news events from DiverseSumm × DSGlobal, 363 questions, 1275 perspective claims, 7385 paragraphs (83.3% annotated as relevant).

---

## Key Findings

### 1. NEWSCOPE's APD Metric is Negatively Correlated with Coverage

APD (Average Pairwise Distance) — NEWSCOPE's primary diversity metric — is **negatively** correlated with Coverage@K across all methods and all K values. Higher geometric diversity actually predicts **worse** perspective coverage.

| Method | K | APD↔Coverage ρ | p-value |
|--------|---|----------------|---------|
| GreedySCS | 10 | -0.291 | 0.0003*** |
| FacLoc | 10 | -0.218 | 0.008** |
| SentNovelty | 10 | -0.404 | <0.0001*** |

**Code:** `scripts/run_newscope_faithful.py` — Stage 3 (lines 466-514) computes correlations.

### 2. The Cross-Encoder Reranker is Biased Against Minority Perspectives

The BGE reranker (`BAAI/bge-reranker-large`) systematically assigns lower scores to paragraphs carrying unique perspectives.

| Paragraph type | N | Mean reranker score |
|---------------|---|-------------------|
| Singleton claim (unique perspective) | 460 | 0.512 |
| Shared claim (2+ paragraphs) | 1451 | 0.628 |
| No claim covered | 4240 | 0.464 |

- Mann-Whitney U test (singleton < shared): **p = 1.91e-10***
- Answer group size correlates positively with reranker score: **ρ = +0.175, p < 0.0001**
- 46% of perspective claims are covered by only 1 paragraph — miss it, miss the perspective
- 40% of these singleton paragraphs score below 0.3 (bottom of the ranking)

**Code:** `scripts/reranker_bias_analysis.py` — Part 1 (reranker score distributions), Part 2 (rank quintile analysis).

### 3. Minority Perspectives are Geometrically Marginalized

In embedding space, minority perspectives sit further from the event centroid than majority perspectives.

- Majority distance from centroid: 0.178
- Minority distance from centroid: 0.195
- Mann-Whitney U (minority farther): **p = 0.004**

This means even pure embedding-based diversity methods have an inherent bias toward majority perspectives — "representative" favors the center where majority views cluster.

**Code:** `scripts/qbias_analysis.py` — Part 3 (embedding-space geometry).

### 4. Removing the Reranker Improves Both Coverage and Fairness

Even within NEWSCOPE's own OPTICS clustering framework, removing the cross-encoder improves results:

| Method | K=5 | K=10 | K=20 |
|--------|-----|------|------|
| GreedySCS (original) | 35.8% | 53.6% | 72.2% |
| SCS-NoRerank (same OPTICS, no reranker) | 36.4% | 54.5% | 74.6% |

QBias analysis (K=10):

| Method | Majority | Minority | Bias Gap |
|--------|----------|----------|----------|
| GreedySCS | 50.4% | 40.8% | 9.6pp |
| SCS-NoRerank | 49.6% | 43.3% | **6.3pp** |

The reranker's tiebreaker within clusters actively selects majority-favoring paragraphs.

**Code:** `scripts/pipeline_variants.py` — `scs_norerank_select()` (line 68).

### 5. Diversify-First Pipeline Beats Standard Pipeline

Reversing the pipeline order — diversify first, then soft-rerank — yields the best results:

**Standard pipeline:** retrieve → rerank → diversify
**Our pipeline:** retrieve → diversify (FacLoc, 2K candidates) → soft rerank (0.3 × reranker + 0.7 × diversity rank)

| Method | K=5 | K=10 | K=20 | Compute |
|--------|-----|------|------|---------|
| Reranker only | 27.9% | 41.7% | 67.5% | CrossEnc |
| GreedySCS | 35.8% | 53.6% | 72.2% | Full NEWSCOPE |
| Emb+TF-FacLoc | 38.2% | 55.7% | 74.3% | Para emb + TF-IDF |
| **Div2K→SoftRerank** | **38.2%** | **57.4%★** | **73.9%** | Para emb + TF-IDF + CrossEnc(soft) |

★ Statistically significant vs GreedySCS at K=10 (Wilcoxon p = 0.017)

QBias (K=10):

| Method | Majority | Minority | Bias Gap | Full Q |
|--------|----------|----------|----------|--------|
| Reranker | 43.5% | 33.1% | 10.5pp | 12.1% |
| GreedySCS | 50.4% | 40.8% | 9.6pp | 17.1% |
| **Div2K→SoftRerank** | **52.3%** | **45.2%** | **7.2pp** | **19.6%** |

**Code:** `scripts/pipeline_variants.py` — `div2k_soft_rerank_select()` (line 101).

### 6. TF-IDF Methods Match NEWSCOPE with Zero GPU Compute

Pure TF-IDF methods — requiring no neural models, no GPU — match GreedySCS:

| Method | K=5 | K=10 | K=20 | GPU needed? |
|--------|-----|------|------|-------------|
| GreedySCS | 35.8% | 53.6% | 72.2% | Yes (sentence embeddings + cross-encoder) |
| TF-FacLoc | 36.1% | 53.2% | 73.1% | **No** |
| TF-OPTICS | 36.8% | 53.9% | 70.2% | **No** |
| TF-Hybrid | 34.1% | 52.8% | 76.1%★ | **No** |

★ TF-Hybrid statistically beats GreedySCS at K=20 (p = 0.013)

**Code:** `scripts/cheap_coverage_methods.py` — all TF-IDF and embedding methods.

### 7. Retrieval is Not the Bottleneck

Dense retrieval performance (cosine similarity to headline):

| K | Recall@K | Precision@K |
|---|----------|-------------|
| 5 | 13.3% | 100.0% |
| 10 | 26.6% | 99.8% |
| 20 | 51.8% | 98.2% |

With 83.3% of paragraphs annotated as relevant and MRR = 1.0, retrieval is trivially easy. The challenge is entirely in **ranking within the relevant set for diversity**.

### 8. QBias Political Balance: Bias Lives in Representations, Not Just the Reranker

We ran all methods on QBias (3,976 balanced events, one article per left/center/right). χ² tests uniformity of which leaning gets ranked first (Bonferroni-corrected over 10 methods).

| Method | Left % | Center % | Right % | χ² | Sig |
|--------|--------|----------|---------|-----|-----|
| Reranker | 28.9% | 39.2% | 31.9% | 67.3 | *** |
| BM25 | 28.9% | 37.7% | 33.4% | 46.5 | *** |
| MMR(λ=0.5) | 28.9% | 39.2% | 31.9% | 67.3 | *** |
| FacLoc (emb) | 33.7% | 37.3% | 29.0% | 40.6 | *** |
| GreedySCS | 29.3% | 37.5% | 33.2% | 40.1 | *** |
| GreedyPlus | 29.0% | 39.0% | 32.0% | — | *** |
| Div-First | 33.7% | 37.3% | 29.0% | 40.6 | *** |
| **MaxDiv** | **33.9%** | **33.5%** | **32.7%** | **0.8** | **ns** |
| **TF-FacLoc** | **33.9%** | **33.5%** | **32.7%** | **0.8** | **ns** |
| **NoRerank** | **35.0%** | **33.7%** | **31.3%** | **8.4** | **ns** |

**Key findings:**

1. **Representation bias (FacLoc vs TF-FacLoc):** Same facility location algorithm, different representation. FacLoc with neural embeddings is biased (χ²=40.6***) — centrist articles sit closer to the geometric center of embedding space, so FacLoc picks them first. TF-FacLoc with TF-IDF vectors is perfectly balanced (χ²=0.8, ns) — word-frequency overlap doesn't systematically favor any leaning. **The bias is in the representation, not the algorithm.**

2. **Reranker bias (GreedySCS vs NoRerank):** Same OPTICS clustering, remove reranker tiebreaker. GreedySCS shows center bias (37.5% center, χ²=40.1***). NoRerank is near-balanced (33.7% center, χ²=8.4, ns). **The reranker introduces the center-favoring imbalance.**

3. **Identical pairs explained:** MMR first-pick = Reranker (diversity penalty is zero on first selection). Div-First = FacLoc (α=0.3 too weak to override among only 3 items).

4. **The tension:** Methods achieving political balance (MaxDiv, TF-FacLoc) are NOT the best on DiverseSumm. MaxDiv scores only 42.7% Coverage@10, below BM25. No single method wins on both benchmarks.

**Code:** `scripts/qbias_transfer.py`

### 9. LLM Judge Performs at Human Annotator Level

We validated the Claude Haiku LLM judge against 4 independent human annotators on 100 paragraph-claim pairs. The LLM achieves higher agreement with the human majority vote than the average human annotator pair:

| Comparison | Cohen's kappa | Interpretation |
|-----------|--------------|----------------|
| LLM vs human majority | **0.280** | Fair |
| Mean human pairwise | 0.253 | Fair |
| Best human pair (Orhan-Melani) | 0.411 | Moderate |
| Worst human pair (Constantinos-Zhiyong) | 0.077 | Slight |

Fleiss' kappa across all 4 annotators: 0.250 (fair). Unanimous agreement on only 32/100 pairs, confirming the inherent subjectivity of perspective coverage annotation. The LLM judge's noise is no worse than replacing one human annotator, and the same labels are applied to all methods — relative method rankings are unaffected.

**Full analysis:** `docs/LLM_JUDGE_ANNOTATION.md`
**Code:** `scripts/compute_annotation_agreement.py`

### 10. Oracle Ceiling Shows Massive Headroom

| K | Oracle | GreedySCS | Our Best | Gap to Oracle |
|---|--------|-----------|----------|---------------|
| 5 | 84.1% | 35.8% | 38.2% | 45.9pp |
| 10 | 92.4% | 53.6% | 57.4% | 35.0pp |
| 20 | 94.3% | 72.2% | 73.9% | 20.4pp |

The oracle picks the K paragraphs that greedily maximize claim coverage. The 35pp gap at K=10 suggests there is significant room for improvement, likely requiring perspective-aware models rather than generic diversity.

---

## Complete Method Comparison

### Coverage@K (all methods tested)

| Method | K=5 | K=10 | K=20 | Compute |
|--------|-----|------|------|---------|
| **Div2K→SoftRerank** | **38.2%** | **57.4%** | **73.9%** | Para emb + TF-IDF + CrossEnc(soft) |
| Emb+TF-FacLoc | 38.2% | 55.7% | 74.3% | Para emb + TF-IDF |
| TF-OPTICS | 36.8% | 53.9% | 70.2% | TF-IDF only |
| SCS-NoRerank | 36.4% | 54.5% | 74.6% | Emb+TF+Stanza+OPTICS |
| TF-FacLoc | 36.1% | 53.2% | 73.1% | TF-IDF only |
| GreedySCS | 35.8% | 53.6% | 72.2% | Full NEWSCOPE |
| TF-Hybrid | 34.1% | 52.8% | 76.1% | TF-IDF only |
| GreedyPlus | 33.9% | 52.1% | 70.6% | Emb+OPTICS |
| FacLoc | 33.7% | 48.8% | 72.6% | Para emb + reranker |
| SentNovelty | 32.5% | 50.7% | 73.2% | Sent emb |
| LogDet | 32.4% | 49.7% | 71.5% | Para emb + reranker |
| InfoGain | 31.8% | 51.2% | 71.9% | Sent emb |
| MMR | 30.9% | 48.1% | 69.3% | Para emb + reranker |
| DPP | 30.2% | 48.9% | 71.3% | Para emb + reranker |
| Reranker | 27.9% | 41.7% | 67.5% | CrossEnc |
| DenseRetrieval | 26.5% | 40.1% | 63.3% | Emb |

---

## Code Reference

### Scripts

| File | Purpose |
|------|---------|
| `scripts/run_newscope_faithful.py` | Main pipeline: 6 stages covering embeddings, reranking, NEWSCOPE reproduction, diversity methods, info-gain methods, aggressive novelty variants |
| `scripts/reranker_bias_analysis.py` | Formalized reranker bias analysis + reranker-independent methods (FacLoc-Pure, LexCov, etc.) |
| `scripts/cheap_coverage_methods.py` | TF-IDF-only and embedding methods that beat NEWSCOPE with less compute |
| `scripts/pipeline_variants.py` | SCS-NoRerank and Div2K→SoftRerank: pipeline variant experiments |
| `scripts/qbias_analysis.py` | Question-level perspective bias analysis: majority vs minority, question types, embedding geometry |

### Pipeline Stages (in `run_newscope_faithful.py`)

| Stage | Description | HPC Job |
|-------|-------------|---------|
| 1 | Paragraph embeddings (bilingual-embedding-large) + dense retrieval scores | `hpc/job_stage1.sh` |
| 2 | BGE cross-encoder reranker scores on relevant paragraphs | `hpc/job_stage2.sh` |
| 3 | NEWSCOPE faithful reproduction: GreedySCS + GreedyPlus with OPTICS | `hpc/job_stage3.sh` |
| 4 | Standard diversity: MMR, KL-Div, DPP, FacLoc, LogDet | `hpc/job_stage4.sh` |
| 5 | Information-gain methods: SentNovelty, SatCoverage, InfoGain | `hpc/job_stage5.sh` |
| 6 | Aggressive novelty variants | `hpc/job_stage6.sh` |

### HPC Deployment

| File | Purpose |
|------|---------|
| `hpc/go.sh` | Deploy/submit/download script (usage: `bash hpc/go.sh upload\|submit\|download\|status\|logs`) |
| `hpc/job_stage[1-6].sh` | SLURM job scripts for DelftBlue (gpu partition, V100) |
| `hpc/fix_env.sh` | Environment setup: torch 2.4.1+cu118, numpy<2, model downloads |

### Data Files

| File | Description |
|------|-------------|
| `data/processed/coverage_data.json` | Combined event data (paragraphs, claims, headlines) |
| `data/processed/llm_coverage_labels.json` | LLM-generated coverage labels: claim_id → [paragraph_ids] |
| `data/processed/newscope_dense_scores.json` | Dense retrieval cosine similarities |
| `data/processed/newscope_reranker_scores.json` | BGE cross-encoder scores |
| `data/processed/newscope_faithful_results.json` | Stage 3 results (GreedySCS, GreedyPlus, baselines) |
| `data/processed/rq2_diversity_results.json` | Stage 4 results (MMR, KL, DPP, FacLoc, LogDet) |
| `data/processed/rq2_infogain_results.json` | Stage 5 results (SentNovelty, SatCoverage, InfoGain) |
| `data/processed/rq2_aggressive_results.json` | Stage 6 results (aggressive novelty variants) |
| `data/processed/rq2_reranker_independent_results.json` | Reranker-independent method results |
| `data/processed/rq2_cheap_methods_results.json` | TF-IDF and cheap method results |
| `data/processed/pipeline_variants_results.json` | SCS-NoRerank and Div2K→SoftRerank results |
| `data/processed/qbias_diversity_results.json` | QBias political balance results for all methods |
| `data/processed/retrieval_results.json` | BM25 DiverseSumm results |
| `data/processed/maxdiv_diversesumm_results.json` | MaxDiv on DiverseSumm (reranker pool) |
| `data/processed/emb_*.npz` | Per-event paragraph embeddings (147 files) |

---

## Paper Narrative

### The Argument (in order)

1. **NEWSCOPE proposes geometric diversity metrics** (APD, PCC, IDR) to evaluate multi-perspective news retrieval but never validates them against actual perspective coverage.

2. **We propose Coverage@K** grounded in DiverseSumm's human-annotated perspective claims as a ground-truth evaluation metric.

3. **RQ1 — APD fails as a proxy**: APD is negatively correlated with Coverage@K (ρ = -0.291, p = 0.0003 for GreedySCS at K=10). Maximizing geometric distance does not maximize perspective coverage.

4. **RQ2 — Standard diversity methods underperform**: Off-the-shelf methods (MMR, DPP, FacLoc, LogDet) all underperform GreedySCS when they rely on the reranker.

5. **Root cause — reranker bias**: The cross-encoder reranker systematically suppresses minority perspectives (p < 1e-10). It is trained to predict relevance, but relevance correlates with majority framing.

6. **Fix — remove or downweight the reranker**:
   - Within NEWSCOPE's own pipeline, SCS-NoRerank beats GreedySCS at every K
   - Our Div2K→SoftRerank statistically significantly beats GreedySCS at K=10 (p = 0.017)
   - Pure TF-IDF methods match GreedySCS with zero GPU compute

7. **The pipeline should be reversed**: retrieve → diversify → soft-rerank, not retrieve → rerank → diversify. The diversification step must happen before the reranker can suppress minority perspectives.

8. **QBias political balance analysis**: Two sources of bias identified:
   - **Representation bias**: FacLoc(emb) is biased (χ²=40.6***) but TF-FacLoc is balanced (χ²=0.8, ns) — same algorithm, different vectors
   - **Reranker bias**: GreedySCS is biased (χ²=40.1***) but NoRerank is balanced (χ²=8.4, ns) — same clustering, no reranker tiebreaker
   - Methods achieving political balance (MaxDiv, TF-FacLoc) fail on DiverseSumm — geometric diversification is encoding-dependent

---

## Remaining Work

- [ ] Human annotation validation (225 paragraph-claim pairs, 4 annotators)
- [x] Formalize diversify-first and NEWSCOPE variant experiments into proper scripts
- [x] Generate paper figures (reranker_bias.pdf, apd_vs_coverage.pdf)
- [x] QBias political balance experiments for all methods
- [x] Write paper (IR_PAPER.tex — full draft complete)
- [ ] Merge paper versions (our analytical version + groupmates' detailed methods version)
- [ ] Confound analysis with correct faithful NEWSCOPE numbers
