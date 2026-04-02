"""
Run retrieval methods and evaluate Coverage@K.

Methods:
1. Random baseline
2. BM25 (relevance-only)
3. Dense retrieval (embedding similarity to headline)
4. MMR (Maximal Marginal Relevance)
5. NEWSCOPE-style GreedySCS (sentence cluster coverage + relevance)

Usage:
    python scripts/run_retrieval_and_evaluate.py
"""

import json
import os
import random
import time
import warnings
import numpy as np
from pathlib import Path

# Suppress harmless overflow warnings from sklearn matmul with high-dim embeddings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = PROJECT_ROOT / "data" / "processed"

random.seed(42)
np.random.seed(42)


# ──────────────────────────────────────────────
# Coverage@K computation
# ──────────────────────────────────────────────

def coverage_at_k(event, coverage_labels, selected_pids):
    """Fraction of claims covered by at least one paragraph in selected_pids."""
    selected_set = set(selected_pids)
    n_claims = event["n_claims"]
    if n_claims == 0:
        return 0.0
    covered = 0
    for c in event["claims"]:
        cid = c["claim_id"]
        covering_pids = set(coverage_labels.get(cid, []))
        if covering_pids & selected_set:
            covered += 1
    return covered / n_claims


# ──────────────────────────────────────────────
# Retrieval methods
# ──────────────────────────────────────────────

def random_baseline(event, K, n_trials=50):
    """Random selection from all paragraphs."""
    all_pids = [p["paragraph_id"] for p in event["paragraphs"]]
    results = []
    for _ in range(n_trials):
        sample = random.sample(all_pids, min(K, len(all_pids)))
        results.append(sample)
    return results  # returns list of lists (multiple trials)


def bm25_retrieval(event, headline, K, bm25_index, pid_list):
    """BM25 retrieval using headline as query."""
    from rank_bm25 import BM25Okapi
    scores = bm25_index.get_scores(headline.lower().split())
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [pid_list[i] for i in ranked[:K]]


def dense_retrieval(event, headline_emb, para_embeddings, pid_list, K):
    """Dense retrieval: cosine similarity between headline and paragraphs."""
    from sklearn.metrics.pairwise import cosine_similarity
    sims = cosine_similarity([headline_emb], para_embeddings)[0]
    ranked = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)
    return [pid_list[i] for i in ranked[:K]]


def mmr_retrieval(headline_emb, para_embeddings, pid_list, K, lambda_=0.5):
    """Maximal Marginal Relevance."""
    from sklearn.metrics.pairwise import cosine_similarity
    relevance = cosine_similarity([headline_emb], para_embeddings)[0]
    selected = []
    selected_idx = []
    remaining = list(range(len(pid_list)))

    for _ in range(K):
        if not remaining:
            break
        if not selected_idx:
            # First pick: most relevant
            best = max(remaining, key=lambda i: relevance[i])
        else:
            # MMR: lambda * relevance - (1-lambda) * max_sim_to_selected
            sel_embs = para_embeddings[selected_idx]
            best_score = -float("inf")
            best = None
            for i in remaining:
                rel = relevance[i]
                max_sim = cosine_similarity([para_embeddings[i]], sel_embs).max()
                score = lambda_ * rel - (1 - lambda_) * max_sim
                if score > best_score:
                    best_score = score
                    best = i
        selected.append(pid_list[best])
        selected_idx.append(best)
        remaining.remove(best)

    return selected


def newscope_greedy_scs(para_embeddings, pid_list, para_texts, relevance_scores,
                        K, w_diversity=1.0, w_relevance=1.0):
    """
    NEWSCOPE GreedySCS-style: sentence-level clustering + greedy selection.
    Simplified: we cluster at paragraph level (not sentence) using OPTICS-like approach,
    then greedily pick paragraphs to cover new clusters weighted by relevance.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity

    # Cluster paragraphs (NEWSCOPE uses sentence-level OPTICS, we approximate with paragraph-level KMeans)
    n_clusters = min(20, len(pid_list) // 2)
    if n_clusters < 2:
        # Too few paragraphs, just return by relevance
        ranked = sorted(range(len(pid_list)), key=lambda i: relevance_scores[i], reverse=True)
        return [pid_list[i] for i in ranked[:K]]

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(para_embeddings)

    # Greedy selection: pick paragraphs to maximize cluster coverage + relevance
    selected = []
    selected_idx = set()
    covered_clusters = set()

    for _ in range(K):
        if len(selected_idx) == len(pid_list):
            break
        best_idx = None
        best_score = -float("inf")

        for i in range(len(pid_list)):
            if i in selected_idx:
                continue
            cluster = cluster_labels[i]
            new_cluster = 1.0 if cluster not in covered_clusters else 0.0
            score = w_diversity * new_cluster + w_relevance * relevance_scores[i]
            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx is None:
            break
        selected.append(pid_list[best_idx])
        selected_idx.add(best_idx)
        covered_clusters.add(cluster_labels[best_idx])

    return selected


def newscope_greedy_plus(para_embeddings, pid_list, para_texts, relevance_scores,
                         K, w_cluster=2.0, w_sim=1.0):
    """
    NEWSCOPE GreedyPlus: cluster scores weighted by mean relevance of cluster members.
    """
    from sklearn.cluster import KMeans

    n_clusters = min(20, len(pid_list) // 2)
    if n_clusters < 2:
        ranked = sorted(range(len(pid_list)), key=lambda i: relevance_scores[i], reverse=True)
        return [pid_list[i] for i in ranked[:K]]

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(para_embeddings)

    # Compute cluster scores (mean relevance of members)
    cluster_scores = defaultdict(list)
    for i, cl in enumerate(cluster_labels):
        cluster_scores[cl].append(relevance_scores[i])
    cluster_scores = {cl: np.mean(scores) for cl, scores in cluster_scores.items()}

    # Greedy selection
    selected = []
    selected_idx = set()
    covered_clusters = set()

    for _ in range(K):
        if len(selected_idx) == len(pid_list):
            break
        best_idx = None
        best_score = -float("inf")

        for i in range(len(pid_list)):
            if i in selected_idx:
                continue
            cluster = cluster_labels[i]
            if cluster not in covered_clusters:
                diversity_score = cluster_scores[cluster]
            else:
                diversity_score = 0.0
            score = w_cluster * diversity_score + w_sim * relevance_scores[i]
            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx is None:
            break
        selected.append(pid_list[best_idx])
        selected_idx.add(best_idx)
        covered_clusters.add(cluster_labels[best_idx])

    return selected


# ──────────────────────────────────────────────
# Geometric diversity metrics
# ──────────────────────────────────────────────

def compute_apd(embeddings):
    """Average Pairwise Distance (cosine)."""
    from sklearn.metrics.pairwise import cosine_similarity
    if len(embeddings) < 2:
        return 0.0
    sim = cosine_similarity(embeddings)
    n = len(embeddings)
    total = 0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1 - sim[i][j]
            count += 1
    return total / count if count > 0 else 0.0


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    from rank_bm25 import BM25Okapi
    import statistics

    print("Loading data...")
    with open(DATA / "coverage_data.json") as f:
        events = json.load(f)
    with open(DATA / "llm_coverage_labels.json") as f:
        labels = json.load(f)

    print("Loading embedding model (NEWSCOPE's bilingual-embedding-large)...")
    model = SentenceTransformer("Lajavaness/bilingual-embedding-large", trust_remote_code=True)

    Ks = [5, 10, 20]
    methods = ["Random", "BM25", "Dense", "MMR_0.5", "MMR_0.7", "GreedySCS", "GreedyPlus"]

    # Store all results
    all_results = []  # list of dicts

    print(f"\nRunning {len(methods)} methods on {len(events)} events...")
    t_start = time.time()

    for ei, event in enumerate(events):
        eid = event["dsglobal_id"]
        cov = labels.get(eid, {})

        if not cov:
            continue

        # Prepare data for this event
        pid_list = [p["paragraph_id"] for p in event["paragraphs"]]
        para_texts = [p["text"] for p in event["paragraphs"]]
        headline = event["headline"]

        # Compute embeddings
        para_embeddings = model.encode(para_texts, show_progress_bar=False)
        headline_emb = model.encode(headline, show_progress_bar=False)

        # BM25 index
        tokenized = [t.lower().split() for t in para_texts]
        bm25 = BM25Okapi(tokenized)

        # Relevance scores (dense similarity to headline — used by NEWSCOPE methods)
        relevance_scores = cosine_similarity([headline_emb], para_embeddings)[0]
        # Normalize to [0, 1]
        rel_min, rel_max = relevance_scores.min(), relevance_scores.max()
        if rel_max > rel_min:
            relevance_scores_norm = (relevance_scores - rel_min) / (rel_max - rel_min)
        else:
            relevance_scores_norm = np.ones_like(relevance_scores)

        for K in Ks:
            # Random
            random_trials = random_baseline(event, K)
            random_covs = [coverage_at_k(event, cov, trial) for trial in random_trials]
            random_avg = statistics.mean(random_covs)
            # APD for one random trial
            random_pids_sample = random_trials[0]
            random_idx = [pid_list.index(p) for p in random_pids_sample if p in pid_list]
            random_apd = compute_apd(para_embeddings[random_idx]) if len(random_idx) >= 2 else 0

            all_results.append({
                "event_id": eid, "method": "Random", "K": K,
                "coverage": float(random_avg), "apd": float(random_apd)
            })

            # BM25
            bm25_pids = bm25_retrieval(event, headline, K, bm25, pid_list)
            bm25_cov = coverage_at_k(event, cov, bm25_pids)
            bm25_idx = [pid_list.index(p) for p in bm25_pids]
            bm25_apd = compute_apd(para_embeddings[bm25_idx]) if len(bm25_idx) >= 2 else 0
            all_results.append({
                "event_id": eid, "method": "BM25", "K": K,
                "coverage": float(bm25_cov), "apd": float(bm25_apd)
            })

            # Dense
            dense_pids = dense_retrieval(event, headline_emb, para_embeddings, pid_list, K)
            dense_cov = coverage_at_k(event, cov, dense_pids)
            dense_idx = [pid_list.index(p) for p in dense_pids]
            dense_apd = compute_apd(para_embeddings[dense_idx]) if len(dense_idx) >= 2 else 0
            all_results.append({
                "event_id": eid, "method": "Dense", "K": K,
                "coverage": float(dense_cov), "apd": float(dense_apd)
            })

            # MMR lambda=0.5
            mmr5_pids = mmr_retrieval(headline_emb, para_embeddings, pid_list, K, lambda_=0.5)
            mmr5_cov = coverage_at_k(event, cov, mmr5_pids)
            mmr5_idx = [pid_list.index(p) for p in mmr5_pids]
            mmr5_apd = compute_apd(para_embeddings[mmr5_idx]) if len(mmr5_idx) >= 2 else 0
            all_results.append({
                "event_id": eid, "method": "MMR_0.5", "K": K,
                "coverage": float(mmr5_cov), "apd": float(mmr5_apd)
            })

            # MMR lambda=0.7
            mmr7_pids = mmr_retrieval(headline_emb, para_embeddings, pid_list, K, lambda_=0.7)
            mmr7_cov = coverage_at_k(event, cov, mmr7_pids)
            mmr7_idx = [pid_list.index(p) for p in mmr7_pids]
            mmr7_apd = compute_apd(para_embeddings[mmr7_idx]) if len(mmr7_idx) >= 2 else 0
            all_results.append({
                "event_id": eid, "method": "MMR_0.7", "K": K,
                "coverage": float(mmr7_cov), "apd": float(mmr7_apd)
            })

            # GreedySCS
            scs_pids = newscope_greedy_scs(
                para_embeddings, pid_list, para_texts, relevance_scores_norm, K
            )
            scs_cov = coverage_at_k(event, cov, scs_pids)
            scs_idx = [pid_list.index(p) for p in scs_pids]
            scs_apd = compute_apd(para_embeddings[scs_idx]) if len(scs_idx) >= 2 else 0
            all_results.append({
                "event_id": eid, "method": "GreedySCS", "K": K,
                "coverage": float(scs_cov), "apd": float(scs_apd)
            })

            # GreedyPlus
            gp_pids = newscope_greedy_plus(
                para_embeddings, pid_list, para_texts, relevance_scores_norm, K
            )
            gp_cov = coverage_at_k(event, cov, gp_pids)
            gp_idx = [pid_list.index(p) for p in gp_pids]
            gp_apd = compute_apd(para_embeddings[gp_idx]) if len(gp_idx) >= 2 else 0
            all_results.append({
                "event_id": eid, "method": "GreedyPlus", "K": K,
                "coverage": float(gp_cov), "apd": float(gp_apd)
            })

        if (ei + 1) % 20 == 0:
            elapsed = time.time() - t_start
            print(f"  [{ei+1}/{len(events)}] {elapsed:.0f}s elapsed")

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.0f}s")

    # Save raw results
    with open(DATA / "retrieval_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # ── Summary table ──
    print("\n" + "=" * 80)
    print("RESULTS: Mean Coverage@K across 147 events")
    print("=" * 80)

    for K in Ks:
        print(f"\n  Coverage@{K}:")
        print(f"  {'Method':<15s} {'Coverage':>10s} {'APD':>8s}")
        print(f"  {'-'*15} {'-'*10} {'-'*8}")
        for method in methods:
            rows = [r for r in all_results if r["method"] == method and r["K"] == K]
            if not rows:
                continue
            mean_cov = statistics.mean(r["coverage"] for r in rows)
            mean_apd = statistics.mean(r["apd"] for r in rows)
            print(f"  {method:<15s} {mean_cov*100:>9.1f}% {mean_apd:>8.3f}")

    # ── RQ1: Correlation between APD and Coverage@K ──
    from scipy.stats import spearmanr
    print("\n" + "=" * 80)
    print("RQ1: Spearman correlation between APD and Coverage@K")
    print("=" * 80)
    for K in Ks:
        rows = [r for r in all_results if r["K"] == K]
        apds = [r["apd"] for r in rows]
        covs = [r["coverage"] for r in rows]
        rho, p = spearmanr(apds, covs)
        print(f"  K={K}: ρ = {rho:.3f} (p = {p:.4f})")

    print(f"\nResults saved to data/processed/retrieval_results.json")


if __name__ == "__main__":
    main()
