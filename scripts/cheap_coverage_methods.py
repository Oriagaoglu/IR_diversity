"""
Cheap coverage methods: maximize perspective coverage with minimal compute.

Goal: beat GreedySCS (35.8/53.6/72.2 at K=5/10/20) using methods that
do NOT require:
- Neural sentence embeddings
- Sentence segmentation (stanza/nltk)
- Cross-encoder reranker scores
- GPU at all

Only allowed inputs: raw paragraph text + headline.
We test with/without paragraph embeddings to see what's needed.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.stats import spearmanr, wilcoxon
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
from sklearn.cluster import KMeans, AgglomerativeClustering, OPTICS
import statistics
import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = PROJECT_ROOT / "data" / "processed"


def coverage_at_k(event, coverage_labels, selected_pids):
    selected_set = set(selected_pids)
    n_claims = event["n_claims"]
    if n_claims == 0:
        return 0.0
    covered = sum(1 for c in event["claims"]
                  if set(coverage_labels.get(c["claim_id"], [])) & selected_set)
    return covered / n_claims


def compute_apd(embeddings):
    if len(embeddings) < 2:
        return 0.0
    sim = sk_cosine(embeddings)
    n = len(embeddings)
    total = sum(1 - sim[i][j] for i in range(n) for j in range(i + 1, n))
    count = n * (n - 1) / 2
    return total / count if count > 0 else 0.0


def main():
    print("=" * 80)
    print("CHEAP COVERAGE METHODS")
    print("=" * 80)

    with open(DATA / "coverage_data.json") as f:
        events = json.load(f)
    with open(DATA / "llm_coverage_labels.json") as f:
        labels = json.load(f)
    with open(DATA / "newscope_reranker_scores.json") as f:
        reranker_store = json.load(f)

    Ks = [5, 10, 20]
    all_results = []

    for ei, event in enumerate(events):
        eid = event["dsglobal_id"]
        cov = labels.get(eid, {})
        if not cov or eid not in reranker_store:
            continue

        rel_paras = [p for p in event["paragraphs"] if p["relevant"] == 1]
        rel_pids = [p["paragraph_id"] for p in rel_paras]
        rel_texts = [p["text"] for p in rel_paras]

        if len(rel_pids) < 2:
            continue

        rscores = reranker_store[eid]

        # Load paragraph embeddings (for embedding-based methods)
        npz_path = DATA / f"emb_{eid}.npz"
        if not npz_path.exists():
            continue
        npz = np.load(npz_path, allow_pickle=True)
        emb_pids = list(npz["pids"])
        emb_matrix = npz["embeddings"]
        para_emb_dict = {pid: emb_matrix[i] for i, pid in enumerate(emb_pids)}

        valid_pids = [pid for pid in rel_pids if pid in para_emb_dict]
        if len(valid_pids) < 2:
            continue

        embs = np.array([para_emb_dict[pid] for pid in valid_pids])
        scores = np.array([rscores.get(pid, 0.0) for pid in valid_pids])
        texts = [rel_texts[rel_pids.index(pid)] for pid in valid_pids]
        n = len(valid_pids)

        # ── Precompute TF-IDF (cheap, CPU-only) ──
        tfidf = TfidfVectorizer(stop_words="english", max_df=0.9,
                                ngram_range=(1, 2), max_features=5000)
        try:
            tfidf_mat = tfidf.fit_transform(texts)
            tfidf_dense = tfidf_mat.toarray()
            tfidf_sim = (tfidf_mat @ tfidf_mat.T).toarray()
        except:
            continue

        # Embedding similarity
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1e-10)
        emb_sim = (embs / norms) @ (embs / norms).T

        # ════════════════════════════════════════════
        # ZERO-COMPUTE METHODS (TF-IDF only, no embeddings, no reranker)
        # ════════════════════════════════════════════

        # ── M1: TF-IDF Facility Location (pure representativeness) ──
        def tfidf_facloc(K):
            selected = []
            remaining = list(range(n))
            current_max = np.full(n, -np.inf)
            for _ in range(min(K, n)):
                best_idx, best_gain = None, -1e9
                for idx in remaining:
                    gain = np.sum(np.maximum(0, tfidf_sim[:, idx] - current_max))
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = idx
                if best_idx is None:
                    break
                selected.append(best_idx)
                remaining.remove(best_idx)
                current_max = np.maximum(current_max, tfidf_sim[:, best_idx])
            return [valid_pids[i] for i in selected]

        # ── M2: TF-IDF K-Means + Greedy Cluster Coverage ──
        # Mimics GreedySCS's discrete topic discovery but with TF-IDF
        def tfidf_kmeans_greedy(K, n_clusters=None):
            if n_clusters is None:
                n_clusters = min(max(K * 2, 10), n)
            actual_k = min(n_clusters, n)
            km = KMeans(n_clusters=actual_k, random_state=42, n_init=10)
            cluster_labels = km.fit_predict(tfidf_dense)

            # Greedy: pick paragraph covering most new clusters
            # (like GreedySCS but with KMeans clusters on TF-IDF)
            selected = []
            remaining = list(range(n))
            covered_clusters = set()
            for _ in range(min(K, n)):
                best_idx, best_score = None, -1e9
                for idx in remaining:
                    cl = cluster_labels[idx]
                    new_clusters = 1 if cl not in covered_clusters else 0
                    # Tiebreak by TF-IDF centrality (distance to centroid)
                    centroid_dist = np.linalg.norm(tfidf_dense[idx] - km.cluster_centers_[cl])
                    centrality = 1.0 / (1.0 + centroid_dist)
                    score = new_clusters + 0.1 * centrality
                    if score > best_score:
                        best_score = score
                        best_idx = idx
                if best_idx is None:
                    break
                selected.append(best_idx)
                remaining.remove(best_idx)
                covered_clusters.add(cluster_labels[best_idx])
            return [valid_pids[i] for i in selected]

        # ── M3: TF-IDF Agglomerative + Round Robin ──
        def tfidf_agglo_roundrobin(K, n_clusters=None):
            if n_clusters is None:
                n_clusters = min(max(K, 5), n)
            actual_k = min(n_clusters, n)
            agglo = AgglomerativeClustering(n_clusters=actual_k, metric='cosine',
                                            linkage='average')
            cluster_labels = agglo.fit_predict(tfidf_dense)

            # Build cluster -> paragraphs (sorted by TF-IDF norm = "informativeness")
            clusters = defaultdict(list)
            for idx in range(n):
                info = float(np.linalg.norm(tfidf_dense[idx]))
                clusters[cluster_labels[idx]].append((idx, info))
            for cl in clusters:
                clusters[cl].sort(key=lambda x: -x[1])

            # Round-robin: pick from each cluster in order of cluster size
            selected = []
            sorted_clusters = sorted(clusters.keys(),
                                     key=lambda c: -len(clusters[c]))
            pointers = {cl: 0 for cl in sorted_clusters}
            while len(selected) < min(K, n):
                picked_any = False
                for cl in sorted_clusters:
                    if len(selected) >= K:
                        break
                    while pointers[cl] < len(clusters[cl]):
                        idx, _ = clusters[cl][pointers[cl]]
                        pointers[cl] += 1
                        if idx not in selected:
                            selected.append(idx)
                            picked_any = True
                            break
                if not picked_any:
                    break
            return [valid_pids[i] for i in selected]

        # ── M4: TF-IDF OPTICS + Greedy (direct GreedySCS replica but cheap) ──
        def tfidf_optics_greedy(K):
            if n < 3:
                return [valid_pids[i] for i in range(min(K, n))]
            dist = 1.0 - np.clip(tfidf_sim, 0, 1)
            dist = np.where(dist > 0, dist, 0)
            try:
                cluster_labels = OPTICS(min_samples=2, metric='precomputed',
                                        n_jobs=-1).fit_predict(dist)
            except:
                return tfidf_kmeans_greedy(K)

            # Greedy cluster coverage (same logic as GreedySCS)
            selected = []
            remaining = list(range(n))
            covered = set()
            for _ in range(min(K, n)):
                best_idx, best_score = None, -1e9
                for idx in remaining:
                    new_cl = {cluster_labels[idx]} - covered
                    div = len(new_cl)
                    # No reranker — use TF-IDF norm as quality proxy
                    quality = float(np.linalg.norm(tfidf_dense[idx]))
                    score = 1.0 * div + 0.1 * quality
                    if score > best_score:
                        best_score = score
                        best_idx = idx
                if best_idx is None:
                    break
                selected.append(best_idx)
                remaining.remove(best_idx)
                covered.add(cluster_labels[best_idx])
            return [valid_pids[i] for i in selected]

        # ── M5: Lexical Coverage + TF-IDF FacLoc hybrid ──
        def lexcov_facloc_hybrid(K):
            """First half by FacLoc (representativeness), second half by
            lexical coverage (novelty). Best of both worlds."""
            half = max(K // 2, 1)
            # Phase 1: FacLoc for representatives
            selected = []
            remaining = list(range(n))
            current_max = np.full(n, -np.inf)
            for _ in range(min(half, n)):
                best_idx, best_gain = None, -1e9
                for idx in remaining:
                    gain = np.sum(np.maximum(0, tfidf_sim[:, idx] - current_max))
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = idx
                if best_idx is None:
                    break
                selected.append(best_idx)
                remaining.remove(best_idx)
                current_max = np.maximum(current_max, tfidf_sim[:, best_idx])

            # Phase 2: Lexical novelty for remaining slots
            covered_terms = np.zeros(tfidf_mat.shape[1])
            for idx in selected:
                covered_terms = np.maximum(covered_terms, tfidf_dense[idx])
            for _ in range(min(K - len(selected), len(remaining))):
                best_idx, best_score = None, -1e9
                for idx in remaining:
                    new_cov = np.sum(np.maximum(0, tfidf_dense[idx] - covered_terms))
                    if new_cov > best_score:
                        best_score = new_cov
                        best_idx = idx
                if best_idx is None:
                    break
                selected.append(best_idx)
                remaining.remove(best_idx)
                covered_terms = np.maximum(covered_terms, tfidf_dense[best_idx])
            return [valid_pids[i] for i in selected]

        # ════════════════════════════════════════════
        # EMBEDDING METHODS (paragraph embeddings, no reranker)
        # ════════════════════════════════════════════

        # ── M6: Embedding FacLoc (no reranker) — our previous best ──
        def emb_facloc_pure(K):
            selected = []
            remaining = list(range(n))
            current_max = np.full(n, -np.inf)
            for _ in range(min(K, n)):
                best_idx, best_gain = None, -1e9
                for idx in remaining:
                    gain = np.sum(np.maximum(0, emb_sim[:, idx] - current_max))
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = idx
                if best_idx is None:
                    break
                selected.append(best_idx)
                remaining.remove(best_idx)
                current_max = np.maximum(current_max, emb_sim[:, best_idx])
            return [valid_pids[i] for i in selected]

        # ── M7: Embedding OPTICS + Greedy (like GreedySCS but paragraph-level) ──
        def emb_optics_greedy(K):
            if n < 3:
                return [valid_pids[i] for i in range(min(K, n))]
            dist = 1.0 - np.clip(emb_sim, 0, 1)
            dist = np.where(dist > 0, dist, 0)
            try:
                cluster_labels = OPTICS(min_samples=2, metric='precomputed',
                                        n_jobs=-1).fit_predict(dist)
            except:
                return emb_facloc_pure(K)

            selected = []
            remaining = list(range(n))
            covered = set()
            for _ in range(min(K, n)):
                best_idx, best_score = None, -1e9
                for idx in remaining:
                    new_cl = {cluster_labels[idx]} - covered
                    div = len(new_cl)
                    # No reranker — use embedding norm as proxy
                    score = 1.0 * div
                    if score > best_score or (score == best_score and best_idx is not None):
                        best_score = score
                        best_idx = idx
                if best_idx is None:
                    break
                selected.append(best_idx)
                remaining.remove(best_idx)
                covered.add(cluster_labels[best_idx])
            return [valid_pids[i] for i in selected]

        # ── M8: Embedding + TF-IDF combined FacLoc ──
        def combined_facloc(K):
            """Use both embedding and TF-IDF similarity (averaged) for FacLoc.
            Captures both semantic and lexical representativeness."""
            combined_sim = 0.5 * emb_sim + 0.5 * tfidf_sim
            selected = []
            remaining = list(range(n))
            current_max = np.full(n, -np.inf)
            for _ in range(min(K, n)):
                best_idx, best_gain = None, -1e9
                for idx in remaining:
                    gain = np.sum(np.maximum(0, combined_sim[:, idx] - current_max))
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = idx
                if best_idx is None:
                    break
                selected.append(best_idx)
                remaining.remove(best_idx)
                current_max = np.maximum(current_max, combined_sim[:, best_idx])
            return [valid_pids[i] for i in selected]

        # ── M9: Embedding KMeans + FacLoc within clusters ──
        def emb_kmeans_facloc(K, n_clusters=None):
            """Cluster with KMeans, then pick most representative from each cluster."""
            if n_clusters is None:
                n_clusters = min(K, n)
            actual_k = min(n_clusters, n)
            km = KMeans(n_clusters=actual_k, random_state=42, n_init=10)
            cluster_labels = km.fit_predict(embs)

            # For each cluster, find the paragraph closest to centroid
            clusters = defaultdict(list)
            for idx in range(n):
                cl = cluster_labels[idx]
                dist = np.linalg.norm(embs[idx] - km.cluster_centers_[cl])
                clusters[cl].append((idx, dist))
            for cl in clusters:
                clusters[cl].sort(key=lambda x: x[1])

            # Round-robin from largest clusters, picking centroids first
            selected = []
            sorted_clusters = sorted(clusters.keys(), key=lambda c: -len(clusters[c]))
            pointers = {cl: 0 for cl in sorted_clusters}
            while len(selected) < min(K, n):
                picked_any = False
                for cl in sorted_clusters:
                    if len(selected) >= K:
                        break
                    while pointers[cl] < len(clusters[cl]):
                        idx, _ = clusters[cl][pointers[cl]]
                        pointers[cl] += 1
                        if idx not in selected:
                            selected.append(idx)
                            picked_any = True
                            break
                if not picked_any:
                    break
            return [valid_pids[i] for i in selected]

        # ════════════════════════════════════════════
        # EVALUATE
        # ════════════════════════════════════════════
        methods = {
            # Zero-compute (TF-IDF only)
            "TF-FacLoc": tfidf_facloc,
            "TF-KMeans": tfidf_kmeans_greedy,
            "TF-Agglo-RR": tfidf_agglo_roundrobin,
            "TF-OPTICS": tfidf_optics_greedy,
            "TF-Hybrid": lexcov_facloc_hybrid,
            # Embedding-based (no reranker)
            "Emb-FacLoc": emb_facloc_pure,
            "Emb-OPTICS": emb_optics_greedy,
            "Emb+TF-FacLoc": combined_facloc,
            "Emb-KM-FacLoc": emb_kmeans_facloc,
        }

        for K in Ks:
            for method_name, method_fn in methods.items():
                try:
                    selected_pids = method_fn(K)
                except Exception as e:
                    selected_pids = valid_pids[:K]

                cov_score = coverage_at_k(event, cov, selected_pids)
                sel_embs = np.array([para_emb_dict[p] for p in selected_pids
                                     if p in para_emb_dict])
                apd = compute_apd(sel_embs) if len(sel_embs) >= 2 else 0

                all_results.append({
                    "event_id": eid, "method": method_name, "K": K,
                    "coverage": float(cov_score), "apd": float(apd)
                })

        if (ei + 1) % 20 == 0:
            print(f"  [{ei+1}/{len(events)}] events processed", flush=True)

    # Save
    with open(DATA / "rq2_cheap_methods_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # ── Load baselines for comparison ──
    with open(DATA / "newscope_faithful_results.json") as f:
        prev = json.load(f)
    baseline_lookup = defaultdict(list)
    for r in prev:
        baseline_lookup[(r["method"], r["K"])].append(r)

    new_lookup = defaultdict(list)
    for r in all_results:
        new_lookup[(r["method"], r["K"])].append(r)

    # ── Results ──
    print("\n" + "=" * 80)
    print("RESULTS: Cheap Coverage Methods vs GreedySCS")
    print("=" * 80)

    print(f"\n  {'Method':<18s} {'K=5':>8s} {'K=10':>8s} {'K=20':>8s}  {'Compute':>20s}")
    print(f"  {'-'*18} {'-'*8} {'-'*8} {'-'*8}  {'-'*20}")

    # Baselines
    for method in ["DenseRetrieval", "Reranker", "GreedySCS", "GreedyPlus"]:
        vals = []
        for K in Ks:
            rows = baseline_lookup.get((method, K), [])
            v = statistics.mean(r["coverage"] for r in rows) * 100 if rows else 0
            vals.append(v)
        compute = {"DenseRetrieval": "Emb (all paras)",
                    "Reranker": "Emb + CrossEnc",
                    "GreedySCS": "Emb+TF+Stanza+OPTICS",
                    "GreedyPlus": "Emb+OPTICS"}[method]
        print(f"  {method:<18s} {vals[0]:>7.1f}% {vals[1]:>7.1f}% {vals[2]:>7.1f}%  {compute:>20s}")

    print(f"  {'--- ZERO COMPUTE (TF-IDF only, no GPU) ---'}")
    for method in ["TF-FacLoc", "TF-KMeans", "TF-Agglo-RR", "TF-OPTICS", "TF-Hybrid"]:
        vals = []
        for K in Ks:
            rows = new_lookup.get((method, K), [])
            v = statistics.mean(r["coverage"] for r in rows) * 100 if rows else 0
            vals.append(v)
        print(f"  {method:<18s} {vals[0]:>7.1f}% {vals[1]:>7.1f}% {vals[2]:>7.1f}%  {'TF-IDF only':>20s}")

    print(f"  {'--- EMBEDDING (no reranker, no sentences) ---'}")
    for method in ["Emb-FacLoc", "Emb-OPTICS", "Emb+TF-FacLoc", "Emb-KM-FacLoc"]:
        vals = []
        for K in Ks:
            rows = new_lookup.get((method, K), [])
            v = statistics.mean(r["coverage"] for r in rows) * 100 if rows else 0
            vals.append(v)
        print(f"  {method:<18s} {vals[0]:>7.1f}% {vals[1]:>7.1f}% {vals[2]:>7.1f}%  {'Para emb only':>20s}")

    # ── Statistical tests vs GreedySCS ──
    print("\n" + "=" * 80)
    print("STATISTICAL TESTS vs GreedySCS (paired Wilcoxon)")
    print("=" * 80)

    scs_by_event = defaultdict(dict)
    for r in prev:
        if r["method"] == "GreedySCS":
            scs_by_event[r["K"]][r["event_id"]] = r["coverage"]

    all_methods = ["TF-FacLoc", "TF-KMeans", "TF-Agglo-RR", "TF-OPTICS", "TF-Hybrid",
                   "Emb-FacLoc", "Emb-OPTICS", "Emb+TF-FacLoc", "Emb-KM-FacLoc"]

    for K in Ks:
        print(f"\n  K={K}:")
        scs_k = scs_by_event[K]
        for method in all_methods:
            method_by_event = {}
            for r in all_results:
                if r["method"] == method and r["K"] == K:
                    method_by_event[r["event_id"]] = r["coverage"]

            common = sorted(set(scs_k.keys()) & set(method_by_event.keys()))
            if len(common) < 10:
                continue

            scs_arr = np.array([scs_k[eid] for eid in common])
            new_arr = np.array([method_by_event[eid] for eid in common])
            diff = new_arr - scs_arr
            wins = np.sum(diff > 0.01)
            losses = np.sum(diff < -0.01)
            ties = len(diff) - wins - losses

            try:
                stat, p = wilcoxon(new_arr, scs_arr, alternative='two-sided')
                sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            except:
                p, sig = 1.0, 'ns'

            mean_diff = np.mean(diff) * 100
            marker = " <-- BEATS" if mean_diff > 0 and p < 0.05 else ""
            marker = " <-- MATCHES" if abs(mean_diff) < 2 and p > 0.05 else marker
            print(f"    {method:<16s} Δ={mean_diff:>+5.1f}pp  W/L/T={wins}/{losses}/{ties}  p={p:.4f} {sig}{marker}")

    print(f"\n\nSaved to data/processed/rq2_cheap_methods_results.json")
    print("=" * 80)


if __name__ == "__main__":
    main()
