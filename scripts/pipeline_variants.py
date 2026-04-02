"""
Pipeline variant experiments: SCS-NoRerank and Div2K→SoftRerank.

These were originally tested as inline experiments during development.
This script formalizes them into reproducible, standalone evaluations.

SCS-NoRerank:
  Same as GreedySCS (stanza + emb+TF-IDF + OPTICS clustering) but the
  reranker tiebreaker is removed from the greedy selection score:
    Original:  score = 1.0 * div + reranker_score
    Variant:   score = 1.0 * div  (no reranker influence)

Div2K→SoftRerank:
  Diversify-first pipeline — reverse the standard retrieve→rerank→diversify
  order to retrieve→diversify→soft-rerank:
    1. Start from all relevant paragraphs (paragraph embeddings)
    2. Diversify via FacLoc on combined emb+TF-IDF similarity
    3. Soft rerank: 0.3 × reranker_score + 0.7 × diversity_rank_score

Requires pre-computed data from run_newscope_faithful.py stages 1-2:
  - data/processed/coverage_data.json
  - data/processed/llm_coverage_labels.json
  - data/processed/newscope_reranker_scores.json
  - data/processed/emb_*.npz (per-event paragraph embeddings)
  - data/processed/newscope_faithful_results.json (baselines for comparison)
"""

import json
import time
import warnings
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.stats import wilcoxon
from sklearn.cluster import OPTICS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import statistics

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
    sim = cosine_similarity(embeddings)
    n = len(embeddings)
    total = sum(1 - sim[i][j] for i in range(n) for j in range(i + 1, n))
    count = n * (n - 1) / 2
    return total / count if count > 0 else 0.0


def scs_norerank_select(rel_texts, scs_p2s, scs_clusters, rel_pids, K):
    """GreedySCS selection without reranker tiebreaker.

    Same OPTICS clustering as GreedySCS, but the greedy score is purely
    the number of new clusters covered — no reranker influence at all.

    Original GreedySCS:  score = 1.0 * div + reranker_score
    This variant:        score = 1.0 * div
    """
    actual_k = min(K, len(rel_pids))
    selected = set()
    covered = set()
    ranking = []

    for _ in range(actual_k):
        best = None
        best_score = -1
        for pid in range(len(rel_texts)):
            if pid in selected:
                continue
            sids = scs_p2s[pid]
            new_cl = set(scs_clusters[s] for s in sids) - covered if sids else set()
            div = len(new_cl)
            # No reranker tiebreaker — this is the key difference
            score = 1.0 * div
            if score > best_score:
                best_score = score
                best = pid
            elif score == best_score and best is not None:
                # Tiebreak by fewer sentences (prefer concise paragraphs)
                if len(sids) < len(scs_p2s[best]):
                    best = pid
        if best is None:
            break
        selected.add(best)
        covered.update(scs_clusters[s] for s in scs_p2s[best])
        ranking.append(best)

    return [rel_pids[i] for i in ranking]


def div2k_soft_rerank_select(valid_pids, embs, tfidf_sim, reranker_scores, K):
    """Diversify-first pipeline: FacLoc on emb+TF-IDF, then soft rerank.

    Pipeline: retrieve → diversify (FacLoc) → soft rerank
    1. Compute combined similarity (0.5 * emb + 0.5 * TF-IDF)
    2. Run FacLoc to get a diversity ranking of all paragraphs
    3. Soft rerank: 0.3 × reranker + 0.7 × diversity_rank_score

    This reverses the standard retrieve→rerank→diversify order, ensuring
    diversification happens before the reranker can suppress minority views.
    """
    n = len(valid_pids)
    if n <= K:
        return valid_pids[:K]

    # Embedding similarity
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1e-10)
    emb_sim = (embs / norms) @ (embs / norms).T

    # Combined similarity for FacLoc
    combined_sim = 0.5 * emb_sim + 0.5 * tfidf_sim

    # Step 1: FacLoc diversity ranking (on all paragraphs)
    div_ranking = []
    remaining = list(range(n))
    current_max = np.full(n, -np.inf)
    for _ in range(n):
        best_idx, best_gain = None, -1e9
        for idx in remaining:
            gain = np.sum(np.maximum(0, combined_sim[:, idx] - current_max))
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
        if best_idx is None:
            break
        div_ranking.append(best_idx)
        remaining.remove(best_idx)
        current_max = np.maximum(current_max, combined_sim[:, best_idx])

    # Step 2: Soft rerank (α=0.3 reranker, 0.7 diversity rank)
    alpha = 0.3
    scores = np.array([reranker_scores.get(pid, 0.0) for pid in valid_pids])

    # Normalize reranker scores to [0, 1]
    if scores.max() > scores.min():
        rs = (scores - scores.min()) / (scores.max() - scores.min())
    else:
        rs = np.zeros(n)

    # Diversity rank score: 1.0 for first (most diverse), 0.0 for last
    div_scores = np.zeros(n)
    for rank, idx in enumerate(div_ranking):
        div_scores[idx] = 1.0 - rank / max(n - 1, 1)

    combined = alpha * rs + (1 - alpha) * div_scores
    final_ranking = np.argsort(-combined)

    return [valid_pids[i] for i in final_ranking[:K]]


def main():
    print("=" * 80)
    print("PIPELINE VARIANTS: SCS-NoRerank and Div2K→SoftRerank")
    print("=" * 80)

    # Lazy import — only needed for SCS-NoRerank
    import stanza
    stanza_nlp = stanza.Pipeline(lang='en', processors='tokenize', verbose=False,
                                  download_method=None)

    with open(DATA / "coverage_data.json") as f:
        events = json.load(f)
    with open(DATA / "llm_coverage_labels.json") as f:
        labels = json.load(f)
    with open(DATA / "newscope_reranker_scores.json") as f:
        reranker_store = json.load(f)

    # We need sentence embeddings for SCS-NoRerank OPTICS clustering
    from sentence_transformers import SentenceTransformer
    print("Loading bilingual-embedding-large for sentence clustering...")
    emb_model = SentenceTransformer('Lajavaness/bilingual-embedding-large',
                                     trust_remote_code=True)
    print("Model loaded.")

    Ks = [5, 10, 20]
    all_results = []
    t_start = time.time()

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

        # Load paragraph embeddings
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
        texts = [rel_texts[rel_pids.index(pid)] for pid in valid_pids]
        n = len(valid_pids)

        # ── Precompute TF-IDF (for both methods) ──
        tfidf = TfidfVectorizer(stop_words="english", max_df=0.9,
                                ngram_range=(1, 2), max_features=5000)
        try:
            tfidf_mat = tfidf.fit_transform(texts)
            tfidf_sim = (tfidf_mat @ tfidf_mat.T).toarray()
        except Exception:
            continue

        # ════════════════════════════════════════════
        # SCS-NoRerank: stanza + emb+TF-IDF + OPTICS (no reranker tiebreaker)
        # ════════════════════════════════════════════

        # Sentence segmentation with stanza (identical to GreedySCS in stage 3)
        scs_sentences = []
        scs_p2s = defaultdict(list)
        scs_s2p = {}
        for para_id, para in enumerate(texts):
            doc = stanza_nlp(para)
            para_sents = [sentence.text for sentence in doc.sentences]
            for sent in para_sents:
                if '\n' in sent:
                    for s in sent.split("\n"):
                        if len(s.split(" ")) > 5:
                            sid = len(scs_sentences)
                            scs_sentences.append(s)
                            scs_p2s[para_id].append(sid)
                            scs_s2p[sid] = para_id
                elif len(sent.split(" ")) > 5:
                    if sent.strip():
                        sid = len(scs_sentences)
                        scs_sentences.append(sent)
                        scs_p2s[para_id].append(sid)
                        scs_s2p[sid] = para_id

        # OPTICS clustering (identical to GreedySCS: emb+TF-IDF representation)
        if len(scs_sentences) >= 2:
            scs_sent_embs = emb_model.encode(scs_sentences, show_progress_bar=False,
                                              batch_size=64)
            scs_tfidf = TfidfVectorizer(stop_words="english", max_df=0.9,
                                         max_features=len(scs_sentences),
                                         ngram_range=(1, 2))
            scs_tfidf_mat = scs_tfidf.fit_transform(scs_sentences).toarray()
            scs_repr = np.concatenate((scs_sent_embs, scs_tfidf_mat), axis=1)
            scs_sim = cosine_similarity(scs_repr)
            scs_dist = 1.0 - scs_sim
            scs_dist = np.where(scs_dist > 0, scs_dist, 0)
            scs_clusters = OPTICS(min_samples=2, metric='precomputed',
                                   n_jobs=-1).fit_predict(scs_dist)
        else:
            scs_clusters = np.zeros(len(scs_sentences), dtype=int)

        # ════════════════════════════════════════════
        # Evaluate both methods at each K
        # ════════════════════════════════════════════

        for K in Ks:
            # SCS-NoRerank
            if len(scs_sentences) >= 2:
                norerank_pids = scs_norerank_select(
                    texts, scs_p2s, scs_clusters, valid_pids, K)
            else:
                norerank_pids = valid_pids[:K]

            norerank_cov = coverage_at_k(event, cov, norerank_pids)
            norerank_embs = np.array([para_emb_dict[p] for p in norerank_pids
                                       if p in para_emb_dict])
            norerank_apd = compute_apd(norerank_embs) if len(norerank_embs) >= 2 else 0

            all_results.append({
                "event_id": eid, "method": "SCS-NoRerank", "K": K,
                "coverage": float(norerank_cov), "apd": float(norerank_apd)
            })

            # Div2K→SoftRerank
            softrerank_pids = div2k_soft_rerank_select(
                valid_pids, embs, tfidf_sim, rscores, K)

            softrerank_cov = coverage_at_k(event, cov, softrerank_pids)
            softrerank_embs = np.array([para_emb_dict[p] for p in softrerank_pids
                                         if p in para_emb_dict])
            softrerank_apd = compute_apd(softrerank_embs) if len(softrerank_embs) >= 2 else 0

            all_results.append({
                "event_id": eid, "method": "Div2K-SoftRerank", "K": K,
                "coverage": float(softrerank_cov), "apd": float(softrerank_apd)
            })

        if (ei + 1) % 10 == 0:
            elapsed = time.time() - t_start
            print(f"  [{ei+1}/{len(events)}] {elapsed:.0f}s elapsed", flush=True)
            # Checkpoint
            with open(DATA / "pipeline_variants_results.json", "w") as f:
                json.dump(all_results, f, indent=2)

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.0f}s")

    with open(DATA / "pipeline_variants_results.json", "w") as f:
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
    print("RESULTS: Pipeline Variants vs Baselines")
    print("=" * 80)

    print(f"\n  {'Method':<20s} {'K=5':>8s} {'K=10':>8s} {'K=20':>8s}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8}")

    for method in ["DenseRetrieval", "Reranker", "GreedySCS", "GreedyPlus"]:
        vals = []
        for K in Ks:
            rows = baseline_lookup.get((method, K), [])
            v = statistics.mean(r["coverage"] for r in rows) * 100 if rows else 0
            vals.append(v)
        print(f"  {method:<20s} {vals[0]:>7.1f}% {vals[1]:>7.1f}% {vals[2]:>7.1f}%")

    print()
    for method in ["SCS-NoRerank", "Div2K-SoftRerank"]:
        vals = []
        for K in Ks:
            rows = new_lookup.get((method, K), [])
            v = statistics.mean(r["coverage"] for r in rows) * 100 if rows else 0
            vals.append(v)
        marker = " ← NEW" if method == "Div2K-SoftRerank" else ""
        print(f"  {method:<20s} {vals[0]:>7.1f}% {vals[1]:>7.1f}% {vals[2]:>7.1f}%{marker}")

    # ── Statistical tests vs GreedySCS ──
    print("\n" + "=" * 80)
    print("STATISTICAL TESTS vs GreedySCS (paired Wilcoxon)")
    print("=" * 80)

    scs_by_event = defaultdict(dict)
    for r in prev:
        if r["method"] == "GreedySCS":
            scs_by_event[r["K"]][r["event_id"]] = r["coverage"]

    for K in Ks:
        print(f"\n  K={K}:")
        scs_k = scs_by_event[K]
        for method in ["SCS-NoRerank", "Div2K-SoftRerank"]:
            method_by_event = {}
            for r in all_results:
                if r["method"] == method and r["K"] == K:
                    method_by_event[r["event_id"]] = r["coverage"]

            common = sorted(set(scs_k.keys()) & set(method_by_event.keys()))
            if len(common) < 10:
                print(f"    {method:<20s} Too few common events ({len(common)})")
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
            except Exception:
                p, sig = 1.0, 'ns'

            mean_diff = np.mean(diff) * 100
            marker = " <-- BEATS" if mean_diff > 0 and p < 0.05 else ""
            marker = " <-- MATCHES" if abs(mean_diff) < 2 and p > 0.05 else marker
            print(f"    {method:<20s} Δ={mean_diff:>+5.1f}pp  "
                  f"W/L/T={wins}/{losses}/{ties}  p={p:.4f} {sig}{marker}")

    print(f"\n\nSaved to data/processed/pipeline_variants_results.json")
    print("=" * 80)


if __name__ == "__main__":
    main()
