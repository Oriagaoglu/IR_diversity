# Evaluating Diversification for Multi-Perspective News Retrieval

Companion repository for the paper *Evaluating Diversification for Multi-Perspective News Retrieval* (Group 19, DSAIT4050/Q3, TU Delft 2026).

## Research Questions

- **RQ1**: Do NEWSCOPE's geometric diversity metrics (APD, PCC) predict actual perspective coverage?
- **RQ2**: Which diversification mechanism and pipeline design best maximizes Coverage@K?
- **RQ3** (transfer): Does geometric diversification achieve political balance on QBias?

## Datasets

| Dataset | Events | Purpose |
|---------|--------|---------|
| DiverseSumm + DSGlobal | 147 events, 1,275 claims, 7,385 paragraphs | Main evaluation: perspective coverage |
| QBias (AllSides) | 3,976 events, 3 articles each (L/C/R) | Transfer: political balance |

## Repository Structure

```
scripts/
  run_newscope_faithful.py       # Main pipeline (stages 1-6)
  reranker_bias_analysis.py      # Reranker bias analysis
  cheap_coverage_methods.py      # TF-IDF and embedding methods
  pipeline_variants.py           # NoRerank and Div-First ablations
  qbias_transfer.py              # QBias political balance experiments
  qbias_analysis.py              # Perspective bias analysis (majority/minority)
  confound_analysis.py           # Confound checks
  llm_scorer.py                  # LLM coverage label generation
  generate_paper_figure.py       # Reproduce paper figures
  annotator_app.py               # Human annotation tool (PyQt6)
  compute_annotation_agreement.py # Inter-annotator agreement

src/
  data_utils.py                  # Data loading utilities
  evaluate.py                    # Evaluation functions
  metrics.py                     # Coverage@K and other metrics
  rerankers.py                   # Reranker wrappers
  diversity/                     # Diversity method implementations

data/
  annotations/                   # Human annotations (4 annotators, 100 pairs)
  processed/                     # Precomputed results (see below)

figures/                         # Paper figures (PDF + PNG)
hpc/                             # SLURM job scripts for DelftBlue
docs/                            # Supplementary analysis documents
```

## Reproducing Results

### Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Models required (downloaded automatically on first run):
- `Lajavaness/bilingual-embedding-large` (paragraph embeddings)
- `BAAI/bge-reranker-large` (cross-encoder reranker)

### Pipeline Stages

The main pipeline runs in 6 stages via `scripts/run_newscope_faithful.py`. On HPC (DelftBlue), use the job scripts in `hpc/`.

| Stage | Script | Description |
|-------|--------|-------------|
| 1 | `run_newscope_faithful.py` | Paragraph embeddings + dense retrieval |
| 2 | `run_newscope_faithful.py` | BGE cross-encoder reranker scores |
| 3 | `run_newscope_faithful.py` | NEWSCOPE reproduction (GreedySCS, GreedyPlus) |
| 4 | `run_newscope_faithful.py` | Standard diversity (MMR, DPP, FacLoc, LogDet) |
| 5 | `run_newscope_faithful.py` | Information-gain methods (SentNovelty, InfoGain) |
| 6 | `cheap_coverage_methods.py` | TF-IDF and compute-efficient methods |

Additional experiments:
```bash
python scripts/pipeline_variants.py      # NoRerank + Div-First ablations
python scripts/qbias_transfer.py         # QBias political balance
python scripts/reranker_bias_analysis.py  # Reranker bias statistics
python scripts/generate_paper_figure.py   # Reproduce figures 1 and 2
```

### Human Annotation Validation

```bash
python scripts/annotator_app.py               # Annotation tool
python scripts/compute_annotation_agreement.py # Compute agreement
```

See `docs/LLM_JUDGE_VALIDATION.md` for full validation methodology and results.

## Key Results

| Method | Coverage@5 | Coverage@10 | Coverage@20 |
|--------|-----------|------------|------------|
| Reranker (baseline) | 27.9 | 41.7 | 67.5 |
| GreedySCS (NEWSCOPE) | 35.8 | 53.6 | 72.2 |
| TF-FacLoc (no GPU) | 36.1 | 53.2 | 73.1 |
| NoRerank (ablation) | 36.4 | 54.5 | 74.6 |
| **Div-First** | **38.2** | **57.4** | 73.9 |
| Oracle | 84.1 | 92.4 | 94.3 |

See `FINDINGS.md` for complete results across all 16 methods, QBias transfer analysis, and detailed discussion.

## Authors

Orhan Agaoglu, Melani Evangelou, Constantinos Kaniklides, Zhiyong Zhu

TU Delft, 2026

## Declaration of Generative AI Usage

This repository was developed with the assistance of Claude (Anthropic).
Claude was used for code scaffolding, documentation structuring, and
debugging support. Claude Haiku was additionally used as the LLM judge
for Coverage@K labeling (`scripts/llm_scorer.py`). All research
decisions, experimental design, interpretation of results, and final
claims were made and verified by the authors.
