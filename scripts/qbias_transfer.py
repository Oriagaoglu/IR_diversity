"""
QBias Transfer Experiment
=========================
Phase 5: Test whether our reranker bias findings from DiverseSumm
generalize to political media bias in the QBias (AllSides) dataset.

Research Question:
  Does the BGE cross-encoder reranker systematically favor certain
  political leanings (left/center/right) in news retrieval?

Dataset: AllSides balanced news (21,754 articles, 7,263 events)
  - Each event has articles from left/center/right sources
  - We use the 3,979 perfectly balanced events (1 article per bias)

Experiment:
  1. Embed articles using the same model as our main pipeline
  2. Score articles with BGE reranker (query = event headline/title)
  3. Analyze whether reranker scores differ by political leaning
  4. Test if our diversify-first approach improves political diversity
  5. Compare with DiverseSumm findings (majority/minority bias)

Usage:
  python scripts/qbias_transfer.py [stage]
  stage 1: Embed articles + compute reranker scores (GPU needed)
  stage 2: Analyze reranker bias by political leaning (CPU only)
  stage 3: Diversity methods comparison (CPU only)
"""

import json
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path("data/processed")
RAW_DIR = Path("data/raw/qbias")


def load_balanced_events():
    """Load events with exactly one article per left/center/right."""
    df = pd.read_csv(RAW_DIR / "allsides_balanced_news_headlines-texts.csv")
    df = df.dropna(subset=["text"])

    # Filter to perfectly balanced events (1 per bias)
    balanced_titles = []
    for title, group in df.groupby("title"):
        if len(group) == 3 and group["bias_rating"].nunique() == 3:
            balanced_titles.append(title)

    df_balanced = df[df["title"].isin(balanced_titles)].copy()
    df_balanced = df_balanced.sort_values(["title", "bias_rating"])

    print(f"Balanced events: {len(balanced_titles)}")
    print(f"Articles: {len(df_balanced)}")
    print(f"Bias distribution: {df_balanced['bias_rating'].value_counts().to_dict()}")

    return df_balanced, balanced_titles


def stage1_embed_and_score():
    """Stage 1: Embed articles and compute reranker scores."""
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch

    print("=" * 60)
    print("STAGE 1: Embed articles + compute reranker scores")
    print("=" * 60)

    df, titles = load_balanced_events()

    # --- Embeddings ---
    print("\n--- Computing article embeddings ---")
    emb_model = SentenceTransformer("Lajavaness/bilingual-embedding-large", trust_remote_code=True)

    texts = df["text"].tolist()
    # Truncate long texts to first 512 chars (paragraph-level, matching our pipeline)
    texts = [t[:2000] if isinstance(t, str) else "" for t in texts]

    embeddings = emb_model.encode(texts, batch_size=64, show_progress_bar=True,
                                   normalize_embeddings=True)
    print(f"Embeddings shape: {embeddings.shape}")

    # Also embed titles (queries)
    title_texts = df["title"].unique().tolist()
    title_embeddings = emb_model.encode(title_texts, batch_size=64, show_progress_bar=True,
                                         normalize_embeddings=True)
    title_emb_map = dict(zip(title_texts, title_embeddings))

    # --- Reranker scores ---
    print("\n--- Computing BGE reranker scores ---")
    reranker_name = "BAAI/bge-reranker-large"
    tokenizer = AutoTokenizer.from_pretrained(reranker_name)
    reranker = AutoModelForSequenceClassification.from_pretrained(reranker_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reranker = reranker.to(device).eval()

    reranker_scores = []
    batch_size = 32

    for i in range(0, len(df), batch_size):
        batch = df.iloc[i : i + batch_size]
        pairs = []
        for _, row in batch.iterrows():
            query = str(row["title"])
            doc = str(row["text"])[:1000]  # Truncate for reranker
            pairs.append((query, doc))

        inputs = tokenizer(
            [p[0] for p in pairs],
            [p[1] for p in pairs],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            logits = reranker(**inputs).logits.squeeze(-1)
            scores = torch.sigmoid(logits).cpu().numpy()

        reranker_scores.extend(scores.tolist())

        if (i // batch_size) % 50 == 0:
            print(f"  Processed {i}/{len(df)} articles...")

    print(f"Reranker scores computed: {len(reranker_scores)}")

    # --- Dense retrieval scores ---
    print("\n--- Computing dense retrieval scores ---")
    dense_scores = []
    for _, row in df.iterrows():
        idx = df.index.get_loc(row.name)
        t_emb = title_emb_map[row["title"]]
        a_emb = embeddings[idx]
        score = float(np.dot(t_emb, a_emb))
        dense_scores.append(score)

    # --- BM25 scores ---
    print("\n--- Computing BM25 scores ---")
    from rank_bm25 import BM25Okapi

    # Tokenize all documents
    tokenized_docs = [str(text).lower().split() for text in texts]
    bm25 = BM25Okapi(tokenized_docs)

    # Score each document against its corresponding query (title)
    bm25_scores = []
    for _, row in df.iterrows():
        query = str(row["title"]).lower().split()
        idx = df.index.get_loc(row.name)
        # Get score for this specific document
        doc_scores = bm25.get_scores(query)
        bm25_scores.append(float(doc_scores[idx]))

    print(f"BM25 scores computed: {len(bm25_scores)}")

    # --- Save results ---
    results = {
        "articles": [],
        "title_list": titles,
    }

    for i, (_, row) in enumerate(df.iterrows()):
        results["articles"].append({
            "title": row["title"],
            "heading": row["heading"],
            "source": row["source"],
            "bias_rating": row["bias_rating"],
            "reranker_score": reranker_scores[i],
            "dense_score": dense_scores[i],
            "bm25_score": bm25_scores[i],
            "text_len": len(str(row["text"])),
        })

    # Save embeddings separately (large)
    np.savez_compressed(
        DATA_DIR / "qbias_embeddings.npz",
        embeddings=embeddings,
        titles=np.array(titles),
    )

    with open(DATA_DIR / "qbias_scores.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to {DATA_DIR / 'qbias_scores.json'}")
    print(f"Embeddings saved to {DATA_DIR / 'qbias_embeddings.npz'}")


def stage1b_precompute_sentences():
    """Stage 1b: Pre-compute sentence embeddings and clusters for efficiency."""
    from sentence_transformers import SentenceTransformer
    from newscope_methods import (
        segment_sentences_stanza,
        segment_sentences_nltk,
        compute_sentence_representations,
        cluster_sentences_optics
    )

    print("=" * 60)
    print("STAGE 1b: Pre-compute sentence embeddings + clusters")
    print("=" * 60)

    # Load article data
    df_full = pd.read_csv(RAW_DIR / "allsides_balanced_news_headlines-texts.csv")
    df_full = df_full.dropna(subset=["text"])
    balanced_titles = []
    for title, group in df_full.groupby("title"):
        if len(group) == 3 and group["bias_rating"].nunique() == 3:
            balanced_titles.append(title)
    df_full = df_full[df_full["title"].isin(balanced_titles)].copy()
    df_full = df_full.sort_values(["title", "bias_rating"])

    print(f"Articles to process: {len(df_full)}")

    # Load embedding model
    print("Loading embedding model...")
    emb_model = SentenceTransformer("Lajavaness/bilingual-embedding-large", trust_remote_code=True)

    # Pre-compute sentence data for each article
    print("\n--- Processing articles (sentence segmentation + embeddings + clustering) ---")

    sentence_data = {
        "stanza": {},  # For GreedySCS
        "nltk": {}     # For GreedyPlus
    }

    for idx, (_, row) in enumerate(df_full.iterrows()):
        if idx % 500 == 0:
            print(f"  Processed {idx}/{len(df_full)} articles...")

        article_id = f"{row['title']}_{row['bias_rating']}"
        text = str(row["text"])[:2000]

        # Stanza segmentation (GreedySCS)
        sentences_stanza, para_to_sent, sent_to_para = segment_sentences_stanza([text])
        if len(sentences_stanza) > 0:
            sent_reps_stanza = compute_sentence_representations(
                sentences_stanza, emb_model, use_tfidf=True
            )
            clusters_stanza = cluster_sentences_optics(sent_reps_stanza)

            sentence_data["stanza"][article_id] = {
                "sentences": sentences_stanza,
                "embeddings": sent_reps_stanza,
                "clusters": clusters_stanza.tolist(),
                "para_to_sent": para_to_sent,
                "sent_to_para": sent_to_para
            }

        # NLTK segmentation (GreedyPlus)
        sentences_nltk, para_to_sent, sent_to_para = segment_sentences_nltk([text])
        if len(sentences_nltk) > 0:
            sent_reps_nltk = compute_sentence_representations(
                sentences_nltk, emb_model, use_tfidf=False
            )
            clusters_nltk = cluster_sentences_optics(sent_reps_nltk)

            sentence_data["nltk"][article_id] = {
                "sentences": sentences_nltk,
                "embeddings": sent_reps_nltk,
                "clusters": clusters_nltk.tolist(),
                "para_to_sent": para_to_sent,
                "sent_to_para": sent_to_para
            }

    print(f"\n  Processed {len(df_full)}/{len(df_full)} articles")

    # Save sentence data
    print("\n--- Saving sentence embeddings and clusters ---")

    # Save embeddings and metadata separately (embeddings are large)
    for method in ["stanza", "nltk"]:
        # Embeddings (npz)
        embeddings_dict = {}
        for art_id, data in sentence_data[method].items():
            embeddings_dict[art_id] = data["embeddings"]

        np.savez_compressed(
            DATA_DIR / f"qbias_sentence_embeddings_{method}.npz",
            **embeddings_dict
        )

        # Metadata (json) - sentences, clusters, mappings
        metadata = {}
        for art_id, data in sentence_data[method].items():
            metadata[art_id] = {
                "sentences": data["sentences"],
                "clusters": data["clusters"],
                "para_to_sent": data["para_to_sent"],
                "sent_to_para": data["sent_to_para"]
            }

        with open(DATA_DIR / f"qbias_sentence_metadata_{method}.json", "w") as f:
            json.dump(metadata, f)

        print(f"  Saved {method}: {len(embeddings_dict)} articles")

    print(f"\nSaved to {DATA_DIR / 'qbias_sentence_*.npz'} and {DATA_DIR / 'qbias_sentence_*.json'}")


def stage2_reranker_bias():
    """Stage 2: Analyze reranker bias by political leaning."""
    from scipy import stats

    print("=" * 60)
    print("STAGE 2: Reranker bias analysis by political leaning")
    print("=" * 60)

    with open(DATA_DIR / "qbias_scores.json") as f:
        data = json.load(f)

    articles = data["articles"]
    df = pd.DataFrame(articles)

    # --- Part 1: Reranker score distributions ---
    print("\n--- Part 1: Reranker score by political leaning ---")
    for bias in ["left", "center", "right"]:
        scores = df[df["bias_rating"] == bias]["reranker_score"]
        print(f"  {bias:>6}: mean={scores.mean():.4f}, median={scores.median():.4f}, "
              f"std={scores.std():.4f}, n={len(scores)}")

    # Pairwise Mann-Whitney U tests
    print("\n--- Pairwise Mann-Whitney U tests (reranker scores) ---")
    for b1, b2 in [("left", "center"), ("left", "right"), ("center", "right")]:
        s1 = df[df["bias_rating"] == b1]["reranker_score"]
        s2 = df[df["bias_rating"] == b2]["reranker_score"]
        u, p = stats.mannwhitneyu(s1, s2, alternative="two-sided")
        print(f"  {b1} vs {b2}: U={u:.0f}, p={p:.6e}")

    # --- Part 2: Dense retrieval score distributions ---
    print("\n--- Part 2: Dense retrieval score by political leaning ---")
    for bias in ["left", "center", "right"]:
        scores = df[df["bias_rating"] == bias]["dense_score"]
        print(f"  {bias:>6}: mean={scores.mean():.4f}, median={scores.median():.4f}, "
              f"std={scores.std():.4f}")

    # --- Part 3: Per-event analysis ---
    print("\n--- Part 3: Per-event reranker ranking ---")
    # For each balanced event, which bias gets the highest reranker score?
    rank_counts = defaultdict(lambda: {"rank1": 0, "rank2": 0, "rank3": 0})
    score_diffs = {"left_vs_right": [], "center_vs_avg": []}

    events = df.groupby("title")
    for title, group in events:
        if len(group) != 3:
            continue
        ranked = group.sort_values("reranker_score", ascending=False)
        biases = ranked["bias_rating"].tolist()
        scores = ranked["reranker_score"].tolist()

        for i, b in enumerate(biases):
            rank_counts[b][f"rank{i+1}"] += 1

        # Score differences
        left_score = group[group["bias_rating"] == "left"]["reranker_score"].values[0]
        right_score = group[group["bias_rating"] == "right"]["reranker_score"].values[0]
        center_score = group[group["bias_rating"] == "center"]["reranker_score"].values[0]
        avg_score = (left_score + right_score) / 2

        score_diffs["left_vs_right"].append(left_score - right_score)
        score_diffs["center_vs_avg"].append(center_score - avg_score)

    print("  How often each bias ranks #1, #2, #3 by reranker score:")
    print(f"  {'Bias':>8} {'Rank 1':>8} {'Rank 2':>8} {'Rank 3':>8}")
    n_events = sum(rank_counts["left"].values())
    for bias in ["left", "center", "right"]:
        r1 = rank_counts[bias]["rank1"]
        r2 = rank_counts[bias]["rank2"]
        r3 = rank_counts[bias]["rank3"]
        print(f"  {bias:>8} {r1:>5} ({r1/n_events*100:.1f}%) "
              f"{r2:>5} ({r2/n_events*100:.1f}%) "
              f"{r3:>5} ({r3/n_events*100:.1f}%)")

    # Statistical test: is center favored?
    print("\n--- Part 4: Is center systematically favored? ---")
    center_diffs = score_diffs["center_vs_avg"]
    t_stat, p_val = stats.ttest_1samp(center_diffs, 0)
    print(f"  Center vs avg(left,right) score diff: "
          f"mean={np.mean(center_diffs):.4f}, t={t_stat:.3f}, p={p_val:.6e}")

    lr_diffs = score_diffs["left_vs_right"]
    t_stat2, p_val2 = stats.ttest_1samp(lr_diffs, 0)
    print(f"  Left vs right score diff: "
          f"mean={np.mean(lr_diffs):.4f}, t={t_stat2:.3f}, p={p_val2:.6e}")

    # --- Part 5: Source-level analysis ---
    print("\n--- Part 5: Mean reranker score by source (top 15) ---")
    source_stats = df.groupby("source").agg(
        mean_score=("reranker_score", "mean"),
        n=("reranker_score", "count"),
        bias=("bias_rating", "first"),
    ).sort_values("mean_score", ascending=False)

    source_stats_filtered = source_stats[source_stats["n"] >= 50]
    print(f"  {'Source':<35} {'Bias':>6} {'Mean Score':>10} {'N':>5}")
    for source, row in source_stats_filtered.head(15).iterrows():
        print(f"  {source[:35]:<35} {row['bias']:>6} {row['mean_score']:>10.4f} {int(row['n']):>5}")

    print("\n  --- Bottom 15 ---")
    for source, row in source_stats_filtered.tail(15).iterrows():
        print(f"  {source[:35]:<35} {row['bias']:>6} {row['mean_score']:>10.4f} {int(row['n']):>5}")

    # --- Part 6: Text length as confound ---
    print("\n--- Part 6: Text length confound ---")
    for bias in ["left", "center", "right"]:
        lens = df[df["bias_rating"] == bias]["text_len"]
        print(f"  {bias:>6}: mean_len={lens.mean():.0f}, median_len={lens.median():.0f}")

    rho, p = stats.spearmanr(df["text_len"], df["reranker_score"])
    print(f"  Text length ↔ reranker score: ρ={rho:.4f}, p={p:.6e}")

    # --- Part 7: Comparison with DiverseSumm findings ---
    print("\n--- Part 7: Summary & comparison with DiverseSumm ---")
    print("  DiverseSumm finding: Reranker favors majority perspectives (p < 1e-10)")
    print("  DiverseSumm finding: Minority perspectives geometrically marginalized (p = 0.004)")
    print()
    print("  QBias findings:")
    print(f"    Center vs partisan diff: mean={np.mean(center_diffs):.4f}")
    print(f"    Left vs right diff: mean={np.mean(lr_diffs):.4f}")
    center_rank1_pct = rank_counts["center"]["rank1"] / n_events * 100
    print(f"    Center ranked #1: {center_rank1_pct:.1f}% (expected: 33.3%)")

    # Save analysis results
    analysis = {
        "n_events": n_events,
        "reranker_by_bias": {},
        "rank_distribution": {},
        "center_vs_avg_diff": float(np.mean(center_diffs)),
        "left_vs_right_diff": float(np.mean(lr_diffs)),
    }
    for bias in ["left", "center", "right"]:
        scores = df[df["bias_rating"] == bias]["reranker_score"]
        analysis["reranker_by_bias"][bias] = {
            "mean": float(scores.mean()),
            "median": float(scores.median()),
            "std": float(scores.std()),
        }
        analysis["rank_distribution"][bias] = rank_counts[bias]

    with open(DATA_DIR / "qbias_analysis_results.json", "w") as f:
        json.dump(analysis, f, indent=2)

    print(f"\nSaved analysis to {DATA_DIR / 'qbias_analysis_results.json'}")


def stage3_diversity_methods():
    """Stage 3: Test diversity methods on QBias events."""
    from scipy import stats
    from newscope_methods import greedy_scs, greedy_plus
    from sentence_transformers import SentenceTransformer

    print("=" * 60)
    print("STAGE 3: Diversity methods for political perspective coverage")
    print("=" * 60)

    # Set seed for reproducible shuffling
    np.random.seed(42)

    with open(DATA_DIR / "qbias_scores.json") as f:
        data = json.load(f)

    emb_data = np.load(DATA_DIR / "qbias_embeddings.npz", allow_pickle=True)
    embeddings = emb_data["embeddings"]

    # Load original QBias data to get article texts (needed for GreedySCS/GreedyPlus)
    df_full = pd.read_csv(RAW_DIR / "allsides_balanced_news_headlines-texts.csv")
    df_full = df_full.dropna(subset=["text"])
    # Filter to balanced events (same as stage1)
    balanced_titles = []
    for title, group in df_full.groupby("title"):
        if len(group) == 3 and group["bias_rating"].nunique() == 3:
            balanced_titles.append(title)
    df_full = df_full[df_full["title"].isin(balanced_titles)].copy()
    df_full = df_full.sort_values(["title", "bias_rating"])  # Same sort as stage1

    # Build text lookup by (title, bias)
    text_lookup = {}
    for _, row in df_full.iterrows():
        key = (row["title"], row["bias_rating"])
        text_lookup[key] = str(row["text"])[:2000]  # Same truncation as stage1

    df = pd.DataFrame(data["articles"])

    # Load embedding model for NEWSCOPE methods
    print("\n--- Loading embedding model for GreedySCS/GreedyPlus ---")
    emb_model = SentenceTransformer("Lajavaness/bilingual-embedding-large", trust_remote_code=True)

    # Build per-event index
    event_indices = defaultdict(list)
    for i, art in enumerate(data["articles"]):
        event_indices[art["title"]].append(i)

    # For balanced events (3 articles), the diversity task is:
    # Given 3 articles, rank them. Coverage = how many biases are in top-K.
    # But with only 3 articles, K=1 is the interesting case (which one do you pick first?)

    # For a more meaningful test, use events with MORE articles
    # Let's also look at events with 4-6 articles
    print("\n--- Event size distribution ---")
    event_sizes = {t: len(idxs) for t, idxs in event_indices.items()}
    for sz in sorted(set(event_sizes.values())):
        n = sum(1 for v in event_sizes.values() if v == sz)
        print(f"  {sz} articles: {n} events")

    # For balanced 3-article events: which bias does each method pick first?
    print("\n--- Method comparison: Which bias gets ranked #1? ---")

    # Helper to adapt NEWSCOPE methods to (scores, embs, texts, ids) interface
    def _newscope_ranking_helper(method_fn, scores, embs, texts, ids):
        """Adapter for GreedySCS/GreedyPlus to work with QBias event data."""
        # Create reranker_scores dict
        reranker_scores = {aid: float(scores[i]) for i, aid in enumerate(ids)}
        # Call NEWSCOPE method
        selected_ids = method_fn(ids, texts, reranker_scores, emb_model, max_k=len(ids))
        # Convert back to indices
        id_to_idx = {aid: i for i, aid in enumerate(ids)}
        ranking = [id_to_idx[aid] for aid in selected_ids]
        return ranking

    methods = {
        "Reranker": lambda scores, embs, texts, ids, bm25_scores: np.argsort(-scores).tolist(),
        "BM25": lambda scores, embs, texts, ids, bm25_scores: np.argsort(-bm25_scores).tolist(),
        "DenseRetrieval": lambda scores, embs, texts, ids, bm25_scores: np.argsort(
            -np.array([embs[i] for i in range(len(scores))]).dot(embs.mean(axis=0))
        ).tolist(),
        "MMR(λ=0.5)": lambda scores, embs, texts, ids, bm25_scores: _mmr_ranking(scores, embs, lambda_param=0.5),
        "MaxDiv": lambda scores, embs, texts, ids, bm25_scores: _maxdiv_ranking(embs),
        "FacLoc": lambda scores, embs, texts, ids, bm25_scores: _facloc_ranking(embs),
        "SoftRerank(0.3)": lambda scores, embs, texts, ids, bm25_scores: _soft_rerank(scores, embs, alpha=0.3),
        "GreedySCS(w=1)": lambda scores, embs, texts, ids, bm25_scores: _newscope_ranking_helper(
            greedy_scs, scores, embs, texts, ids
        ),
        "GreedyPlus(2,1)": lambda scores, embs, texts, ids, bm25_scores: _newscope_ranking_helper(
            greedy_plus, scores, embs, texts, ids
        ),
    }

    results = {m: {"first_pick": defaultdict(int), "covers_all_at_3": 0} for m in methods}

    balanced_events = [t for t, idxs in event_indices.items() if len(idxs) == 3]

    for title in balanced_events:
        idxs = event_indices[title]
        if len(idxs) != 3:
            continue

        arts = [data["articles"][i] for i in idxs]
        biases = [a["bias_rating"] for a in arts]
        if len(set(biases)) != 3:
            continue

        reranker_scores = np.array([a["reranker_score"] for a in arts])
        bm25_scores = np.array([a.get("bm25_score", 0.0) for a in arts])
        # Get article texts from lookup
        texts = [text_lookup.get((a["title"], a["bias_rating"]), "") for a in arts]
        article_ids = [f"{title}_{i}" for i in range(len(arts))]  # Create unique IDs
        embs = embeddings[idxs]

        # CRITICAL FIX: Shuffle order to avoid bias from alphabetical sorting
        # Articles are sorted by bias_rating (center, left, right), which biases
        # diversity methods that start with index 0 or depend on order
        shuffle_idx = np.random.permutation(3)
        biases = [biases[i] for i in shuffle_idx]
        reranker_scores = reranker_scores[shuffle_idx]
        bm25_scores = bm25_scores[shuffle_idx]
        texts = [texts[i] for i in shuffle_idx]
        article_ids = [article_ids[i] for i in shuffle_idx]
        embs = embs[shuffle_idx]

        for method_name, method_fn in methods.items():
            ranking = method_fn(reranker_scores, embs, texts, article_ids, bm25_scores)
            first_bias = biases[ranking[0]]
            results[method_name]["first_pick"][first_bias] += 1
            # With 3 articles and 3 biases, all methods cover all at K=3
            results[method_name]["covers_all_at_3"] += 1

    n = len(balanced_events)
    print(f"\n  Balanced events analyzed: {n}")
    print(f"\n  {'Method':<20} {'Left #1':>10} {'Center #1':>10} {'Right #1':>10} {'Expected':>10}")
    for method_name in methods:
        fp = results[method_name]["first_pick"]
        l = fp["left"]
        c = fp["center"]
        r = fp["right"]
        print(f"  {method_name:<20} {l:>5} ({l/n*100:.1f}%) {c:>5} ({c/n*100:.1f}%) "
              f"{r:>5} ({r/n*100:.1f}%) {n/3:.0f} (33.3%)")

    # Chi-squared test for each method
    print("\n--- Chi-squared test for uniform ranking ---")
    n_methods = len(methods)
    bonferroni_alpha = 0.05 / n_methods
    print(f"  Bonferroni-corrected α = {bonferroni_alpha:.4f} (testing {n_methods} methods)")
    print()
    expected = [n / 3] * 3
    for method_name in methods:
        fp = results[method_name]["first_pick"]
        observed = [fp["left"], fp["center"], fp["right"]]
        chi2, p = stats.chisquare(observed, expected)
        p_bonf = min(p * n_methods, 1.0)  # Bonferroni-adjusted p-value
        sig = '***' if p_bonf < 0.001 else '**' if p_bonf < 0.01 else '*' if p_bonf < 0.05 else 'ns'
        print(f"  {method_name:<20}: χ²={chi2:.2f}, p={p:.6e}, p_adj={p_bonf:.6e} {sig}")

    # --- Larger events analysis ---
    print("\n--- Larger events (4+ articles): Bias coverage at K ---")
    large_events = [t for t, idxs in event_indices.items() if len(idxs) >= 4]
    print(f"  Events with 4+ articles: {len(large_events)}")

    if large_events:
        for K in [1, 2, 3]:
            print(f"\n  K={K}:")
            for method_name, method_fn in methods.items():
                bias_coverages = []
                for title in large_events:
                    idxs = event_indices[title]
                    arts = [data["articles"][i] for i in idxs]
                    biases = [a["bias_rating"] for a in arts]
                    reranker_scores = np.array([a["reranker_score"] for a in arts])
                    bm25_scores = np.array([a.get("bm25_score", 0.0) for a in arts])
                    texts = [text_lookup.get((a["title"], a["bias_rating"]), "") for a in arts]
                    article_ids = [f"{title}_{i}" for i in range(len(arts))]
                    embs = embeddings[idxs]

                    ranking = method_fn(reranker_scores, embs, texts, article_ids, bm25_scores)
                    top_k_biases = set(biases[r] for r in ranking[:K])
                    bias_coverages.append(len(top_k_biases) / 3)

                mean_cov = np.mean(bias_coverages)
                print(f"    {method_name:<20}: {mean_cov:.3f}")

    # Save
    save_results = {}
    for method_name in methods:
        fp = results[method_name]["first_pick"]
        save_results[method_name] = {
            "first_pick": dict(fp),
            "n_events": n,
        }

    with open(DATA_DIR / "qbias_diversity_results.json", "w") as f:
        json.dump(save_results, f, indent=2)

    print(f"\nSaved to {DATA_DIR / 'qbias_diversity_results.json'}")


def _maxdiv_ranking(embs):
    """Greedy max-diversity: pick most distant from already selected."""
    n = len(embs)
    if n == 0:
        return []
    sim = embs @ embs.T
    selected = [0]  # Start with first
    remaining = list(range(1, n))

    while remaining:
        min_sims = []
        for r in remaining:
            max_sim_to_selected = max(sim[r][s] for s in selected)
            min_sims.append(max_sim_to_selected)
        # Pick the one most distant from selected
        best_idx = np.argmin(min_sims)
        selected.append(remaining[best_idx])
        remaining.pop(best_idx)

    return selected


def _facloc_ranking(embs):
    """Greedy facility location maximization."""
    n = len(embs)
    sim = embs @ embs.T
    sim = np.clip(sim, 0, None)

    selected = []
    remaining = list(range(n))
    current_max = np.zeros(n)

    for _ in range(n):
        gains = []
        for r in remaining:
            new_max = np.maximum(current_max, sim[:, r])
            gain = new_max.sum() - current_max.sum()
            gains.append(gain)
        best_idx = np.argmax(gains)
        best = remaining[best_idx]
        current_max = np.maximum(current_max, sim[:, best])
        selected.append(best)
        remaining.pop(best_idx)

    return selected


def _soft_rerank(reranker_scores, embs, alpha=0.3):
    """Diversify first (FacLoc), then soft rerank."""
    div_ranking = _facloc_ranking(embs)
    n = len(reranker_scores)

    # Normalize reranker scores to [0,1]
    rs = reranker_scores.copy()
    if rs.max() > rs.min():
        rs = (rs - rs.min()) / (rs.max() - rs.min())

    # Diversity rank score: 1.0 for first, 0.0 for last
    div_scores = np.zeros(n)
    for rank, idx in enumerate(div_ranking):
        div_scores[idx] = 1.0 - rank / max(n - 1, 1)

    combined = alpha * rs + (1 - alpha) * div_scores
    return list(np.argsort(-combined))


def _mmr_ranking(reranker_scores, embs, lambda_param=0.5):
    """
    Maximal Marginal Relevance (MMR) - classic IR diversity baseline.

    MMR = λ * Relevance(d) - (1-λ) * max_similarity(d, Selected)

    Args:
        reranker_scores: relevance scores (higher = more relevant)
        embs: embeddings for computing similarity
        lambda_param: trade-off between relevance and diversity (default 0.5)

    Returns:
        ranking: list of indices in selection order
    """
    n = len(reranker_scores)
    if n == 0:
        return []

    # Normalize relevance scores to [0,1]
    rel = reranker_scores.copy()
    if rel.max() > rel.min():
        rel = (rel - rel.min()) / (rel.max() - rel.min())

    # Compute similarity matrix
    sim = embs @ embs.T

    # Greedy selection
    selected = []
    remaining = list(range(n))

    for _ in range(n):
        best_score = -np.inf
        best_idx = None

        for i, idx in enumerate(remaining):
            relevance = rel[idx]

            if len(selected) == 0:
                diversity_penalty = 0.0
            else:
                # Max similarity to already selected
                max_sim = max(sim[idx][s] for s in selected)
                diversity_penalty = max_sim

            mmr_score = lambda_param * relevance - (1 - lambda_param) * diversity_penalty

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i

        if best_idx is None:
            break

        selected.append(remaining[best_idx])
        remaining.pop(best_idx)

    return selected


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "2"

    if stage == "1":
        stage1_embed_and_score()
    elif stage == "1b":
        stage1b_precompute_sentences()
    elif stage == "2":
        stage2_reranker_bias()
    elif stage == "3":
        stage3_diversity_methods()
    else:
        print(f"Unknown stage: {stage}")
        print("Usage: python scripts/qbias_transfer.py [1|1b|2|3]")
