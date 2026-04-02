"""
Faithful NEWSCOPE reproduction + evaluation against Coverage@K.

Runs in 3 stages to manage memory:
  Stage 1: Compute paragraph embeddings + dense retrieval scores
  Stage 2: Compute BGE reranker scores on relevant paragraphs
  Stage 3: Sentence segmentation + OPTICS clustering + greedy selection + evaluation

References: NEWSCOPE_IMPLEMENTATION.md for line-by-line source mapping.
"""

import json
import sys
import os
import time
import warnings
import numpy as np
from pathlib import Path
from collections import defaultdict
import gc

warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = PROJECT_ROOT / "data" / "processed"

def flush_print(*args, **kwargs):
    print(*args, **kwargs, flush=True)


# ──────────────────────────────────────────────
# Coverage@K
# ──────────────────────────────────────────────
def coverage_at_k(event, coverage_labels, selected_pids):
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


def compute_apd(embeddings):
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
# Stage 1: Embeddings + Dense scores
# ──────────────────────────────────────────────
def stage1_embeddings():
    flush_print("=== STAGE 1: Paragraph embeddings + dense retrieval scores ===")
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    with open(DATA / "coverage_data.json") as f:
        events = json.load(f)

    flush_print("Loading bilingual-embedding-large...")
    model = SentenceTransformer('Lajavaness/bilingual-embedding-large', trust_remote_code=True)
    flush_print("Model loaded.")

    # Load existing dense scores if resuming
    dense_path = DATA / "newscope_dense_scores.json"
    if dense_path.exists():
        with open(dense_path) as f:
            dense_scores_store = json.load(f)
    else:
        dense_scores_store = {}

    for ei, event in enumerate(events):
        eid = event["dsglobal_id"]

        # Skip if already done
        if eid in dense_scores_store and (DATA / f"emb_{eid}.npz").exists():
            continue

        all_pids = [p["paragraph_id"] for p in event["paragraphs"]]
        all_texts = [p["text"] for p in event["paragraphs"]]
        headline = event["headline"]

        para_embs = model.encode(all_texts, show_progress_bar=False, batch_size=64)
        headline_emb = model.encode(headline, show_progress_bar=False)

        sims = cosine_similarity([headline_emb], para_embs)[0]
        dense_scores_store[eid] = {pid: float(sims[i]) for i, pid in enumerate(all_pids)}

        np.savez_compressed(DATA / f"emb_{eid}.npz", pids=all_pids, embeddings=para_embs)

        # Free memory
        del para_embs, headline_emb, sims
        gc.collect()

        if (ei + 1) % 20 == 0:
            flush_print(f"  [{ei+1}/{len(events)}] embeddings computed")
            # Checkpoint dense scores
            with open(dense_path, "w") as f:
                json.dump(dense_scores_store, f)

    with open(dense_path, "w") as f:
        json.dump(dense_scores_store, f)
    flush_print(f"Stage 1 done. Saved embeddings (.npz per event) and dense scores.")


# ──────────────────────────────────────────────
# Stage 2: BGE reranker scores
# ──────────────────────────────────────────────
def stage2_reranker():
    flush_print("=== STAGE 2: BGE reranker scores on relevant paragraphs ===")
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    with open(DATA / "coverage_data.json") as f:
        events = json.load(f)

    flush_print("Loading BGE reranker...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-large')
    model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-large').to(device)
    model.eval()
    flush_print("Reranker loaded.")

    reranker_store = {}  # eid -> {pid: score}

    for ei, event in enumerate(events):
        eid = event["dsglobal_id"]
        headline = event["headline"]
        rel_paras = [p for p in event["paragraphs"] if p["relevant"] == 1]
        if not rel_paras:
            continue

        rel_pids = [p["paragraph_id"] for p in rel_paras]
        rel_texts = [p["text"] for p in rel_paras]

        # Score in batches to avoid OOM
        # Ref: rerank_positive_samples.py lines 58-72
        batch_size = 32
        all_scores = []
        for b in range(0, len(rel_texts), batch_size):
            pairs = [[headline, t] for t in rel_texts[b:b+batch_size]]
            with torch.no_grad():
                inputs = tokenizer(pairs, padding=True, truncation=True,
                                   return_tensors='pt', max_length=512).to(device)
                logits = model(**inputs, return_dict=True).logits.view(-1,).float()
                scores = torch.sigmoid(logits).cpu().numpy()
            all_scores.extend(scores.tolist())

        reranker_store[eid] = {pid: all_scores[i] for i, pid in enumerate(rel_pids)}

        if (ei + 1) % 20 == 0:
            flush_print(f"  [{ei+1}/{len(events)}] reranker scores computed")

    with open(DATA / "newscope_reranker_scores.json", "w") as f:
        json.dump(reranker_store, f)
    flush_print(f"Stage 2 done. Saved reranker scores.")


# ──────────────────────────────────────────────
# Stage 3: Clustering + greedy selection + evaluation
# ──────────────────────────────────────────────
def stage3_cluster_and_evaluate():
    flush_print("=== STAGE 3: Sentence clustering + greedy selection + evaluation ===")
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import OPTICS
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from scipy.stats import spearmanr
    from nltk.tokenize import sent_tokenize
    import stanza
    import statistics

    with open(DATA / "coverage_data.json") as f:
        events = json.load(f)
    with open(DATA / "llm_coverage_labels.json") as f:
        labels = json.load(f)
    with open(DATA / "newscope_dense_scores.json") as f:
        dense_store = json.load(f)
    with open(DATA / "newscope_reranker_scores.json") as f:
        reranker_store = json.load(f)

    flush_print("Loading sentence embedding model...")
    emb_model = SentenceTransformer('Lajavaness/bilingual-embedding-large', trust_remote_code=True)
    flush_print("Loading stanza...")
    nlp = stanza.Pipeline(lang='en', processors='tokenize', verbose=False,
                          download_method=None)
    flush_print("Models loaded.")

    Ks = [5, 10, 20]
    all_results = []
    t_start = time.time()

    for ei, event in enumerate(events):
        eid = event["dsglobal_id"]
        cov = labels.get(eid, {})
        if not cov or eid not in reranker_store:
            continue

        all_pids = [p["paragraph_id"] for p in event["paragraphs"]]
        rel_paras = [p for p in event["paragraphs"] if p["relevant"] == 1]
        rel_pids = [p["paragraph_id"] for p in rel_paras]
        rel_texts = [p["text"] for p in rel_paras]
        headline = event["headline"]

        if not rel_pids:
            continue

        # Get precomputed data
        dense_scores = dense_store[eid]
        reranker_scores = reranker_store[eid]

        # Paragraph embeddings for APD computation (load from npz)
        npz = np.load(DATA / f"emb_{eid}.npz", allow_pickle=True)
        emb_pids = list(npz["pids"])
        emb_matrix = npz["embeddings"]
        para_emb_dict = {pid: emb_matrix[i] for i, pid in enumerate(emb_pids)}

        # Reranker scores as list indexed by local position
        reranker_by_idx = {i: reranker_scores.get(pid, 0.0) for i, pid in enumerate(rel_pids)}

        # ── GreedySCS: stanza + embedding+TF-IDF + OPTICS ──
        # Sentence segmentation with stanza (ref: greedy_select_weighted_sum.py lines 100-120)
        scs_sentences = []
        scs_p2s = defaultdict(list)
        scs_s2p = {}
        for para_id, para in enumerate(rel_texts):
            doc = nlp(para)
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

        # Cluster SCS sentences
        if len(scs_sentences) >= 2:
            scs_sent_embs = emb_model.encode(scs_sentences, show_progress_bar=False, batch_size=64)
            tfidf = TfidfVectorizer(stop_words="english", max_df=0.9,
                                    max_features=len(scs_sentences), ngram_range=(1, 2))
            tfidf_mat = tfidf.fit_transform(scs_sentences).toarray()
            scs_repr = np.concatenate((scs_sent_embs, tfidf_mat), axis=1)
            scs_sim = cosine_similarity(scs_repr)
            scs_dist = 1.0 - scs_sim
            scs_dist = np.where(scs_dist > 0, scs_dist, 0)
            scs_clusters = OPTICS(min_samples=2, metric='precomputed', n_jobs=-1).fit_predict(scs_dist)
        else:
            scs_clusters = np.zeros(len(scs_sentences), dtype=int)

        # Build PCC cluster structure for SCS
        scs_c2s = defaultdict(list)
        for sid, cid in enumerate(scs_clusters):
            scs_c2s[str(cid)].append({"text": scs_sentences[sid], "para_id": scs_s2p[sid]})

        # ── GreedyPlus: nltk + embedding-only + OPTICS ──
        # Sentence segmentation with nltk (ref: cluster_score_enhanced_greedy_select.py lines 110-130)
        gp_sentences = []
        gp_p2s = defaultdict(list)
        gp_s2p = {}
        for para_id, para in enumerate(rel_texts):
            para_sents = sent_tokenize(para)
            for sent in para_sents:
                if '\n' in sent:
                    for s in sent.split("\n"):
                        if len(s.split(" ")) > 5:
                            sid = len(gp_sentences)
                            gp_sentences.append(s)
                            gp_p2s[para_id].append(sid)
                            gp_s2p[sid] = para_id
                elif len(sent.split(" ")) > 5:
                    if sent.strip():
                        sid = len(gp_sentences)
                        gp_sentences.append(sent)
                        gp_p2s[para_id].append(sid)
                        gp_s2p[sid] = para_id

        # Cluster GP sentences (embedding only, no TF-IDF)
        if len(gp_sentences) >= 2:
            gp_sent_embs = emb_model.encode(gp_sentences, show_progress_bar=False, batch_size=64)
            gp_sim = cosine_similarity(gp_sent_embs)
            gp_dist = 1.0 - gp_sim
            gp_dist = np.where(gp_dist > 0, gp_dist, 0)
            gp_clusters = OPTICS(min_samples=2, metric='precomputed', n_jobs=-1).fit_predict(gp_dist)
        else:
            gp_clusters = np.zeros(len(gp_sentences), dtype=int)

        gp_c2s = defaultdict(list)
        for sid, cid in enumerate(gp_clusters):
            gp_c2s[str(cid)].append({"text": gp_sentences[sid], "para_id": gp_s2p[sid]})

        # Cluster scores for GreedyPlus
        gp_cluster_scores = {}
        for cid_str in gp_c2s:
            scores = [reranker_by_idx.get(s["para_id"], 0) for s in gp_c2s[cid_str]]
            gp_cluster_scores[cid_str] = np.mean(scores) if scores else 0

        # ── Evaluate at each K ──
        for K in Ks:
            actual_k = min(K, len(rel_pids))

            # Dense retrieval (all paragraphs, sorted by cosine sim)
            dense_ranked = sorted(all_pids, key=lambda pid: dense_scores.get(pid, 0), reverse=True)[:K]
            dense_cov = coverage_at_k(event, cov, dense_ranked)
            dense_embs = np.array([para_emb_dict[p] for p in dense_ranked if p in para_emb_dict])
            dense_apd = compute_apd(dense_embs) if len(dense_embs) >= 2 else 0
            all_results.append({
                "event_id": eid, "method": "DenseRetrieval", "K": K,
                "coverage": float(dense_cov), "apd": float(dense_apd)
            })

            # Reranker only (relevant paragraphs, sorted by BGE score)
            reranker_ranked = sorted(rel_pids, key=lambda pid: reranker_scores.get(pid, 0), reverse=True)[:actual_k]
            reranker_cov = coverage_at_k(event, cov, reranker_ranked)
            reranker_embs = np.array([para_emb_dict[p] for p in reranker_ranked if p in para_emb_dict])
            reranker_apd = compute_apd(reranker_embs) if len(reranker_embs) >= 2 else 0
            all_results.append({
                "event_id": eid, "method": "Reranker", "K": K,
                "coverage": float(reranker_cov), "apd": float(reranker_apd)
            })

            # GreedySCS (w=1)
            if len(scs_sentences) >= 2:
                scs_selected = set()
                scs_covered = set()
                scs_ranking = []
                for _ in range(actual_k):
                    best = None
                    best_score = 0
                    for pid in range(len(rel_texts)):
                        if pid in scs_selected:
                            continue
                        sids = scs_p2s[pid]
                        new_cl = set(scs_clusters[s] for s in sids) - scs_covered if sids else set()
                        div = len(new_cl)
                        sim = reranker_by_idx.get(pid, 0)
                        score = 1.0 * div + sim
                        if score > best_score:
                            best_score = score
                            best = pid
                        elif score == best_score and best is not None:
                            if len(sids) < len(scs_p2s[best]):
                                best = pid
                    if best is None:
                        break
                    scs_selected.add(best)
                    scs_covered.update(scs_clusters[s] for s in scs_p2s[best])
                    scs_ranking.append(best)
                scs_pids = [rel_pids[i] for i in scs_ranking]
            else:
                scs_pids = reranker_ranked

            scs_cov = coverage_at_k(event, cov, scs_pids)
            scs_embs = np.array([para_emb_dict[p] for p in scs_pids if p in para_emb_dict])
            scs_apd = compute_apd(scs_embs) if len(scs_embs) >= 2 else 0

            # PCC for SCS — NEWSCOPE evaluate uses nltk sent_tokenize for PCC
            # (even though GreedySCS uses stanza for clustering), so we match that
            scs_eval_sents = set()
            for pid_local in [rel_pids.index(p) for p in scs_pids if p in rel_pids]:
                para_text = rel_texts[pid_local]
                for sent in sent_tokenize(para_text):
                    if '\n' in sent:
                        for s in sent.split("\n"):
                            if len(s.split(" ")) > 5:
                                scs_eval_sents.add(s)
                    elif len(sent.split(" ")) > 5 and sent.strip():
                        scs_eval_sents.add(sent)
            scs_pcc = 0
            if scs_c2s:
                covered_cl = sum(1 for cid, sents in scs_c2s.items()
                                 if any(s["text"] in scs_eval_sents for s in sents))
                scs_pcc = covered_cl / len(scs_c2s)

            all_results.append({
                "event_id": eid, "method": "GreedySCS", "K": K,
                "coverage": float(scs_cov), "apd": float(scs_apd), "pcc": float(scs_pcc)
            })

            # GreedyPlus (w_cluster=2, w_sim=1)
            if len(gp_sentences) >= 2:
                gp_selected = set()
                gp_covered = set()
                gp_ranking = []
                for _ in range(actual_k):
                    best = None
                    best_score = 0
                    for pid in range(len(rel_texts)):
                        if pid in gp_selected:
                            continue
                        sids = gp_p2s[pid]
                        new_cl = set(str(gp_clusters[s]) for s in sids) - gp_covered if sids else set()
                        div = sum(gp_cluster_scores.get(c, 0) for c in new_cl)
                        sim = reranker_by_idx.get(pid, 0)
                        score = 2.0 * div + 1.0 * sim
                        if score > best_score:
                            best_score = score
                            best = pid
                        elif score == best_score and best is not None:
                            if len(sids) < len(gp_p2s[best]):
                                best = pid
                    if best is None:
                        break
                    gp_selected.add(best)
                    gp_covered.update(str(gp_clusters[s]) for s in gp_p2s[best])
                    gp_ranking.append(best)
                gp_pids = [rel_pids[i] for i in gp_ranking]
            else:
                gp_pids = reranker_ranked

            gp_cov = coverage_at_k(event, cov, gp_pids)
            gp_embs = np.array([para_emb_dict[p] for p in gp_pids if p in para_emb_dict])
            gp_apd = compute_apd(gp_embs) if len(gp_embs) >= 2 else 0

            # PCC for GreedyPlus — also use nltk for evaluation (matching NEWSCOPE)
            gp_eval_sents = set()
            for pid_local in [rel_pids.index(p) for p in gp_pids if p in rel_pids]:
                para_text = rel_texts[pid_local]
                for sent in sent_tokenize(para_text):
                    if '\n' in sent:
                        for s in sent.split("\n"):
                            if len(s.split(" ")) > 5:
                                gp_eval_sents.add(s)
                    elif len(sent.split(" ")) > 5 and sent.strip():
                        gp_eval_sents.add(sent)
            gp_pcc = 0
            if gp_c2s:
                covered_cl = sum(1 for cid, sents in gp_c2s.items()
                                 if any(s["text"] in gp_eval_sents for s in sents))
                gp_pcc = covered_cl / len(gp_c2s)

            all_results.append({
                "event_id": eid, "method": "GreedyPlus", "K": K,
                "coverage": float(gp_cov), "apd": float(gp_apd), "pcc": float(gp_pcc)
            })

        if (ei + 1) % 10 == 0:
            elapsed = time.time() - t_start
            flush_print(f"  [{ei+1}/{len(events)}] {elapsed:.0f}s elapsed")
            # Checkpoint
            with open(DATA / "newscope_faithful_results.json", "w") as f:
                json.dump(all_results, f, indent=2)

    elapsed = time.time() - t_start
    flush_print(f"\nDone in {elapsed:.0f}s")

    with open(DATA / "newscope_faithful_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # ── Summary ──
    methods = ["DenseRetrieval", "Reranker", "GreedySCS", "GreedyPlus"]
    flush_print("\n" + "=" * 80)
    flush_print("RESULTS: Faithful NEWSCOPE reproduction")
    flush_print("=" * 80)

    for K in Ks:
        flush_print(f"\n  K={K}:")
        flush_print(f"  {'Method':<18s} {'Coverage':>10s} {'APD':>8s} {'PCC':>8s}")
        flush_print(f"  {'-'*18} {'-'*10} {'-'*8} {'-'*8}")
        for method in methods:
            rows = [r for r in all_results if r["method"] == method and r["K"] == K]
            if not rows:
                continue
            mean_cov = statistics.mean(r["coverage"] for r in rows)
            mean_apd = statistics.mean(r["apd"] for r in rows)
            pcc_str = "N/A"
            if "pcc" in rows[0]:
                mean_pcc = statistics.mean(r.get("pcc", 0) for r in rows)
                pcc_str = f"{mean_pcc*100:>7.1f}%"
            flush_print(f"  {method:<18s} {mean_cov*100:>9.1f}% {mean_apd:>8.3f} {pcc_str:>8s}")

    # Correlations
    flush_print("\n" + "=" * 80)
    flush_print("CORRELATIONS: NEWSCOPE metrics vs Coverage@K")
    flush_print("=" * 80)
    for K in Ks:
        flush_print(f"\n  K={K}:")
        for method in methods:
            rows = [r for r in all_results if r["method"] == method and r["K"] == K]
            if not rows:
                continue
            covs = [r["coverage"] for r in rows]
            apds = [r["apd"] for r in rows]
            rho, p = spearmanr(apds, covs)
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            line = f"  {method:<18s} APD↔Cov: ρ={rho:+.3f} p={p:.4f} {sig}"
            if "pcc" in rows[0]:
                pccs = [r.get("pcc", 0) for r in rows]
                rho2, p2 = spearmanr(pccs, covs)
                sig2 = '***' if p2 < 0.001 else '**' if p2 < 0.01 else '*' if p2 < 0.05 else 'ns'
                line += f"  |  PCC↔Cov: ρ={rho2:+.3f} p={p2:.4f} {sig2}"
            flush_print(line)

    flush_print(f"\nSaved to data/processed/newscope_faithful_results.json")


# ──────────────────────────────────────────────
# Stage 4: RQ2 Diversity Mechanisms
# KL-divergence, DPP, Facility Location, Log-Det (Entropy)
# All use pre-computed embeddings + reranker scores from stages 1-2
# ──────────────────────────────────────────────
def stage4_diversity_mechanisms():
    flush_print("=== STAGE 4: RQ2 diversity mechanisms (KL, DPP, FacLoc, Entropy) ===")
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans
    from scipy.stats import spearmanr
    import statistics

    with open(DATA / "coverage_data.json") as f:
        events = json.load(f)
    with open(DATA / "llm_coverage_labels.json") as f:
        labels = json.load(f)
    with open(DATA / "newscope_reranker_scores.json") as f:
        reranker_store = json.load(f)

    Ks = [5, 10, 20]
    all_results = []
    t_start = time.time()

    # ── Helper: MMR baseline (for comparison) ──
    def mmr_select(emb_matrix, rel_scores, pids, K, lam=0.5):
        """Maximal Marginal Relevance. Standard IR diversity baseline."""
        selected = []
        remaining = list(range(len(pids)))
        for _ in range(min(K, len(pids))):
            best_idx, best_score = None, -1e9
            for idx in remaining:
                rel = rel_scores[idx]
                if not selected:
                    div_penalty = 0.0
                else:
                    sel_embs = emb_matrix[selected]
                    sims = cosine_similarity([emb_matrix[idx]], sel_embs)[0]
                    div_penalty = float(np.max(sims))
                score = lam * rel - (1 - lam) * div_penalty
                if score > best_score:
                    best_score = score
                    best_idx = idx
            if best_idx is None:
                break
            selected.append(best_idx)
            remaining.remove(best_idx)
        return [pids[i] for i in selected]

    # ── Helper: KL-divergence reranking ──
    def kl_select(emb_matrix, rel_scores, pids, K, n_topics=15, lam=0.5):
        """KL-divergence reranking: select paragraphs to match corpus topic distribution.
        Uses KMeans soft assignments as topic distribution proxy.
        """
        n = len(pids)
        if n <= K:
            return pids[:K]

        # Fit topic model via KMeans on embeddings
        actual_topics = min(n_topics, n)
        km = KMeans(n_clusters=actual_topics, random_state=42, n_init=10)
        km.fit(emb_matrix)

        # Soft topic assignments: use distance to centroids → softmax
        dists = km.transform(emb_matrix)  # (n, n_topics)
        # Convert to similarities (negative distance) and softmax
        neg_dists = -dists
        exp_d = np.exp(neg_dists - neg_dists.max(axis=1, keepdims=True))
        topic_probs = exp_d / exp_d.sum(axis=1, keepdims=True)  # (n, n_topics)

        # Corpus distribution: mean topic assignment
        corpus_dist = topic_probs.mean(axis=0)
        corpus_dist = corpus_dist / corpus_dist.sum()

        selected = []
        remaining = list(range(n))
        for _ in range(min(K, n)):
            best_idx, best_score = None, -1e9
            for idx in remaining:
                # Compute subset distribution if we add this document
                if selected:
                    subset_probs = topic_probs[selected + [idx]].mean(axis=0)
                else:
                    subset_probs = topic_probs[idx]
                subset_dist = subset_probs / (subset_probs.sum() + 1e-10)

                # KL(corpus || subset) - lower is better
                kl = np.sum(corpus_dist * np.log((corpus_dist + 1e-10) / (subset_dist + 1e-10)))

                # Combined: maximize relevance, minimize KL
                score = lam * rel_scores[idx] - (1 - lam) * kl
                if score > best_score:
                    best_score = score
                    best_idx = idx
            if best_idx is None:
                break
            selected.append(best_idx)
            remaining.remove(best_idx)
        return [pids[i] for i in selected]

    # ── Helper: DPP greedy MAP inference ──
    def dpp_select(emb_matrix, rel_scores, pids, K, alpha=1.0):
        """Greedy MAP inference for DPP with L-ensemble kernel.
        L[i,j] = q_i * q_j * sim(i,j) where q = alpha * relevance.
        Uses direct Schur complement for marginal gains.
        """
        n = len(pids)
        if n <= K:
            return pids[:K]

        # Construct L-ensemble kernel (PSD: clip sim to [0,1])
        q = np.array(rel_scores) * alpha
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1e-10)
        normed = emb_matrix / norms
        sim = np.clip(normed @ normed.T, 0, 1)  # ensure PSD
        L = np.outer(q, q) * sim

        selected = []
        remaining = list(range(n))

        for k_idx in range(min(K, n)):
            best_idx, best_gain = None, -1e9
            for idx in remaining:
                if k_idx == 0:
                    gain = np.log(L[idx, idx] + 1e-10)
                else:
                    sel = np.array(selected)
                    L_S = L[np.ix_(sel, sel)]
                    L_si = L[sel, idx]
                    schur = L[idx, idx] - float(L_si @ np.linalg.solve(
                        L_S + 1e-10 * np.eye(len(sel)), L_si))
                    gain = np.log(max(schur, 1e-10))
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
            if best_idx is None:
                break
            selected.append(best_idx)
            remaining.remove(best_idx)
        return [pids[i] for i in selected]

    # ── Helper: Facility Location ──
    def facloc_select(emb_matrix, rel_scores, pids, K):
        """Facility Location submodular maximization.
        f(S) = sum_i max_{j in S} sim(i,j) * rel(j)
        Greedy gives (1 - 1/e) approximation.
        """
        n = len(pids)
        if n <= K:
            return pids[:K]

        # Pairwise cosine similarity
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1e-10)
        normed = emb_matrix / norms
        sim = normed @ normed.T

        # Weighted similarity: sim(i,j) * rel(j)
        rel_arr = np.array(rel_scores)
        weighted_sim = sim * rel_arr[np.newaxis, :]  # (n, n) — column j weighted by rel(j)

        selected = []
        remaining = list(range(n))
        # Track current max coverage per element
        current_max = np.full(n, -np.inf)

        for _ in range(min(K, n)):
            best_idx, best_gain = None, -1e9
            for idx in remaining:
                # Marginal gain: sum_i max(0, weighted_sim[i, idx] - current_max[i])
                gain = np.sum(np.maximum(0, weighted_sim[:, idx] - current_max))
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
            if best_idx is None:
                break
            selected.append(best_idx)
            remaining.remove(best_idx)
            # Update current_max
            current_max = np.maximum(current_max, weighted_sim[:, best_idx])
        return [pids[i] for i in selected]

    # ── Helper: Log-Det (Entropy) ──
    def logdet_select(emb_matrix, rel_scores, pids, K, lam=1.0, gamma=0.01):
        """Log-determinant diversity: maximize log det(G_S + gamma*I) + lambda * sum(rel).
        G_S is Gram matrix of selected embeddings.
        Greedy with incremental Cholesky.
        """
        n = len(pids)
        if n <= K:
            return pids[:K]

        # Normalize embeddings
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1e-10)
        normed = emb_matrix / norms

        selected = []
        remaining = list(range(n))
        # Track Gram matrix incrementally
        G_inv = None  # Will be (k x k) inverse of (G_S + gamma*I)

        for k_idx in range(min(K, n)):
            best_idx, best_score = None, -1e9
            for idx in remaining:
                if k_idx == 0:
                    # First element: log(g_ii + gamma) + lambda * rel
                    g_ii = float(normed[idx] @ normed[idx])
                    det_gain = np.log(g_ii + gamma)
                else:
                    # Marginal gain: log(g_ii + gamma - g_s^T (G_S+gamma*I)^-1 g_s)
                    g_s = normed[selected] @ normed[idx]  # (k,)
                    g_ii = float(normed[idx] @ normed[idx])
                    schur = g_ii + gamma - float(g_s @ G_inv @ g_s)
                    det_gain = np.log(max(schur, 1e-10))
                score = det_gain + lam * rel_scores[idx]
                if score > best_score:
                    best_score = score
                    best_idx = idx
            if best_idx is None:
                break
            selected.append(best_idx)
            remaining.remove(best_idx)
            # Update G_inv using Woodbury/block matrix inverse
            sel = np.array(selected)
            G_S = normed[sel] @ normed[sel].T + gamma * np.eye(len(sel))
            G_inv = np.linalg.inv(G_S)
        return [pids[i] for i in selected]

    # ── Main loop over events ──
    methods_rq2 = {
        "MMR": lambda emb, rel, pids, K: mmr_select(emb, rel, pids, K, lam=0.5),
        "KL-Div": lambda emb, rel, pids, K: kl_select(emb, rel, pids, K, n_topics=15, lam=0.5),
        "DPP": lambda emb, rel, pids, K: dpp_select(emb, rel, pids, K, alpha=1.0),
        "FacLoc": lambda emb, rel, pids, K: facloc_select(emb, rel, pids, K),
        "LogDet": lambda emb, rel, pids, K: logdet_select(emb, rel, pids, K, lam=1.0, gamma=0.01),
    }

    for ei, event in enumerate(events):
        eid = event["dsglobal_id"]
        cov = labels.get(eid, {})
        if not cov or eid not in reranker_store:
            continue

        all_pids = [p["paragraph_id"] for p in event["paragraphs"]]
        rel_paras = [p for p in event["paragraphs"] if p["relevant"] == 1]
        rel_pids = [p["paragraph_id"] for p in rel_paras]

        if not rel_pids:
            continue

        reranker_scores = reranker_store[eid]

        # Load paragraph embeddings
        npz_path = DATA / f"emb_{eid}.npz"
        if not npz_path.exists():
            flush_print(f"  WARNING: no embeddings for {eid}, skipping")
            continue
        npz = np.load(npz_path, allow_pickle=True)
        emb_pids = list(npz["pids"])
        emb_matrix = npz["embeddings"]
        para_emb_dict = {pid: emb_matrix[i] for i, pid in enumerate(emb_pids)}

        # Build arrays for relevant paragraphs only (matching NEWSCOPE's approach)
        # Filter consistently so embeddings, scores, and pids are aligned
        valid_rel_pids = [pid for pid in rel_pids if pid in para_emb_dict]
        rel_embs = np.array([para_emb_dict[pid] for pid in valid_rel_pids])
        rel_scores_list = [reranker_scores.get(pid, 0.0) for pid in valid_rel_pids]

        if len(rel_embs) < 2:
            continue

        for K in Ks:
            for method_name, method_fn in methods_rq2.items():
                try:
                    selected_pids = method_fn(rel_embs, rel_scores_list, valid_rel_pids, K)
                except Exception as e:
                    flush_print(f"  ERROR {method_name} on {eid} K={K}: {e}")
                    selected_pids = sorted(valid_rel_pids,
                                           key=lambda p: reranker_scores.get(p, 0), reverse=True)[:K]

                cov_score = coverage_at_k(event, cov, selected_pids)
                sel_embs = np.array([para_emb_dict[p] for p in selected_pids if p in para_emb_dict])
                apd = compute_apd(sel_embs) if len(sel_embs) >= 2 else 0

                all_results.append({
                    "event_id": eid, "method": method_name, "K": K,
                    "coverage": float(cov_score), "apd": float(apd)
                })

        if (ei + 1) % 10 == 0:
            elapsed = time.time() - t_start
            flush_print(f"  [{ei+1}/{len(events)}] {elapsed:.0f}s elapsed")
            with open(DATA / "rq2_diversity_results.json", "w") as f:
                json.dump(all_results, f, indent=2)

    elapsed = time.time() - t_start
    flush_print(f"\nStage 4 done in {elapsed:.0f}s")

    with open(DATA / "rq2_diversity_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # ── Summary ──
    methods = list(methods_rq2.keys())
    flush_print("\n" + "=" * 80)
    flush_print("RESULTS: RQ2 Diversity Mechanisms")
    flush_print("=" * 80)

    for K in Ks:
        flush_print(f"\n  K={K}:")
        flush_print(f"  {'Method':<12s} {'Coverage':>10s} {'APD':>8s}")
        flush_print(f"  {'-'*12} {'-'*10} {'-'*8}")
        for method in methods:
            rows = [r for r in all_results if r["method"] == method and r["K"] == K]
            if not rows:
                continue
            mean_cov = statistics.mean(r["coverage"] for r in rows)
            mean_apd = statistics.mean(r["apd"] for r in rows)
            flush_print(f"  {method:<12s} {mean_cov*100:>9.1f}% {mean_apd:>8.3f}")

    # Correlations
    flush_print("\n" + "=" * 80)
    flush_print("CORRELATIONS: APD vs Coverage@K (RQ2 methods)")
    flush_print("=" * 80)
    for K in Ks:
        flush_print(f"\n  K={K}:")
        for method in methods:
            rows = [r for r in all_results if r["method"] == method and r["K"] == K]
            if len(rows) < 5:
                continue
            covs = [r["coverage"] for r in rows]
            apds = [r["apd"] for r in rows]
            rho, p = spearmanr(apds, covs)
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            flush_print(f"  {method:<12s} APD↔Cov: ρ={rho:+.3f} p={p:.4f} {sig}")

    flush_print(f"\nSaved to data/processed/rq2_diversity_results.json")


# ──────────────────────────────────────────────
# Stage 5: Information-gain methods
# Sentence-level Novelty, Saturated Coverage, Cross-Encoder Novelty
# Needs sentence embeddings (computed here) + reranker scores from stage 2
# ──────────────────────────────────────────────
def stage5_information_gain():
    flush_print("=== STAGE 5: Information-gain diversity methods ===")
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    from nltk.tokenize import sent_tokenize
    from scipy.stats import spearmanr
    import statistics

    with open(DATA / "coverage_data.json") as f:
        events = json.load(f)
    with open(DATA / "llm_coverage_labels.json") as f:
        labels = json.load(f)
    with open(DATA / "newscope_reranker_scores.json") as f:
        reranker_store = json.load(f)

    flush_print("Loading embedding model...")
    emb_model = SentenceTransformer('Lajavaness/bilingual-embedding-large', trust_remote_code=True)
    flush_print("Model loaded.")

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

        reranker_scores = reranker_store[eid]

        # Load paragraph embeddings for APD
        npz_path = DATA / f"emb_{eid}.npz"
        if not npz_path.exists():
            continue
        npz = np.load(npz_path, allow_pickle=True)
        emb_pids = list(npz["pids"])
        emb_matrix = npz["embeddings"]
        para_emb_dict = {pid: emb_matrix[i] for i, pid in enumerate(emb_pids)}

        # ── Sentence segmentation (nltk, matching NEWSCOPE evaluate) ──
        para_sentences = []  # list of (para_local_idx, list_of_sentences)
        all_sentences = []
        sent_to_para = []  # sent_idx -> para_local_idx
        para_to_sents = defaultdict(list)  # para_local_idx -> [sent_idx, ...]

        for pid_local, text in enumerate(rel_texts):
            sents = []
            for sent in sent_tokenize(text):
                if '\n' in sent:
                    for s in sent.split("\n"):
                        if len(s.split(" ")) > 5:
                            sents.append(s)
                elif len(sent.split(" ")) > 5 and sent.strip():
                    sents.append(sent)
            for s in sents:
                sid = len(all_sentences)
                all_sentences.append(s)
                sent_to_para.append(pid_local)
                para_to_sents[pid_local].append(sid)

        if len(all_sentences) < 2:
            continue

        # Compute sentence embeddings
        sent_embs = emb_model.encode(all_sentences, show_progress_bar=False, batch_size=64)
        # Pairwise sentence similarity matrix
        sent_sim = cosine_similarity(sent_embs)

        rel_scores = [reranker_scores.get(pid, 0.0) for pid in rel_pids]

        # ── Method 1: Sentence Novelty Score ──
        # Score = λ * (novel_sents / total_sents) + reranker_score
        # A sentence is "novel" if max similarity to any selected sentence < threshold
        def sent_novelty_select(K, lam=1.0, threshold=0.7):
            selected_paras = []
            selected_sents = set()  # indices of sentences in selected paragraphs
            remaining = list(range(len(rel_texts)))

            for _ in range(min(K, len(rel_texts))):
                best_pid, best_score = None, -1e9
                for pid in remaining:
                    sids = para_to_sents[pid]
                    if not sids:
                        novelty = 0.0
                    elif not selected_sents:
                        novelty = 1.0  # all sentences are novel
                    else:
                        sel_list = list(selected_sents)
                        novel_count = 0
                        for sid in sids:
                            max_sim = max(sent_sim[sid][ss] for ss in sel_list)
                            if max_sim < threshold:
                                novel_count += 1
                        novelty = novel_count / len(sids)

                    score = lam * novelty + rel_scores[pid]
                    if score > best_score:
                        best_score = score
                        best_pid = pid
                if best_pid is None:
                    break
                selected_paras.append(best_pid)
                remaining.remove(best_pid)
                selected_sents.update(para_to_sents[best_pid])
            return [rel_pids[i] for i in selected_paras]

        # ── Method 2: Saturated Coverage ──
        # f(S) = Σ_i min(Σ_{j∈S} sim(i,j), α) + λ * Σ_{j∈S} rel(j)
        # Concave-over-modular: diminishing returns per region
        def saturated_coverage_select(K, alpha=2.0, lam=0.5):
            n = len(rel_texts)
            # Use sentence-level similarities for finer granularity
            n_sents = len(all_sentences)
            selected_paras = []
            remaining = list(range(n))
            # Track current coverage per sentence
            current_cov = np.zeros(n_sents)

            for _ in range(min(K, n)):
                best_pid, best_gain = None, -1e9
                for pid in remaining:
                    sids = para_to_sents[pid]
                    if not sids:
                        marginal = 0.0
                    else:
                        # Adding this paragraph's sentences: how much new coverage?
                        marginal = 0.0
                        for sid in sids:
                            # Each sentence in this paragraph covers other sentences
                            new_cov = current_cov + sent_sim[sid]
                            # Marginal = Σ min(new_cov, α) - Σ min(current_cov, α)
                            marginal += np.sum(np.minimum(new_cov, alpha) -
                                              np.minimum(current_cov, alpha))
                        marginal /= len(sids)  # normalize by paragraph size

                    score = marginal + lam * rel_scores[pid]
                    if score > best_gain:
                        best_gain = score
                        best_pid = pid
                if best_pid is None:
                    break
                selected_paras.append(best_pid)
                remaining.remove(best_pid)
                for sid in para_to_sents[best_pid]:
                    current_cov += sent_sim[sid]
            return [rel_pids[i] for i in selected_paras]

        # ── Method 3: Max Coverage with Information Gain ──
        # Greedily select paragraph whose sentences cover the most
        # "uncovered" sentence embedding space.
        # Coverage of sentence i by set S = max_{j∈S_sents} sim(i,j)
        # Marginal gain = Σ_i max(0, max_{s∈new_sents} sim(i,s) - current_max_i)
        # Combined with relevance: gain + λ * rel
        def info_gain_select(K, lam=0.5):
            n_sents = len(all_sentences)
            selected_paras = []
            remaining = list(range(len(rel_texts)))
            current_max = np.full(n_sents, -np.inf)

            for _ in range(min(K, len(rel_texts))):
                best_pid, best_score = None, -1e9
                for pid in remaining:
                    sids = para_to_sents[pid]
                    if not sids:
                        gain = 0.0
                    else:
                        # Max similarity from each sentence to this paragraph's sentences
                        para_max = np.max(sent_sim[:, sids], axis=1)
                        # Marginal gain: how much does this improve coverage?
                        gain = float(np.sum(np.maximum(0, para_max - current_max)))

                    score = gain + lam * rel_scores[pid]
                    if score > best_score:
                        best_score = score
                        best_pid = pid
                if best_pid is None:
                    break
                selected_paras.append(best_pid)
                remaining.remove(best_pid)
                sids = para_to_sents[best_pid]
                if sids:
                    para_max = np.max(sent_sim[:, sids], axis=1)
                    current_max = np.maximum(current_max, para_max)
            return [rel_pids[i] for i in selected_paras]

        # ── Evaluate all methods at each K ──
        methods = {
            "SentNovelty": lambda K: sent_novelty_select(K, lam=1.0, threshold=0.7),
            "SatCoverage": lambda K: saturated_coverage_select(K, alpha=2.0, lam=0.5),
            "InfoGain": lambda K: info_gain_select(K, lam=0.5),
        }

        for K in Ks:
            for method_name, method_fn in methods.items():
                try:
                    selected_pids = method_fn(K)
                except Exception as e:
                    flush_print(f"  ERROR {method_name} on {eid} K={K}: {e}")
                    selected_pids = sorted(rel_pids,
                                           key=lambda p: reranker_scores.get(p, 0),
                                           reverse=True)[:K]

                cov_score = coverage_at_k(event, cov, selected_pids)
                sel_embs = np.array([para_emb_dict[p] for p in selected_pids
                                     if p in para_emb_dict])
                apd = compute_apd(sel_embs) if len(sel_embs) >= 2 else 0

                all_results.append({
                    "event_id": eid, "method": method_name, "K": K,
                    "coverage": float(cov_score), "apd": float(apd)
                })

        if (ei + 1) % 10 == 0:
            elapsed = time.time() - t_start
            flush_print(f"  [{ei+1}/{len(events)}] {elapsed:.0f}s elapsed")
            with open(DATA / "rq2_infogain_results.json", "w") as f:
                json.dump(all_results, f, indent=2)

    elapsed = time.time() - t_start
    flush_print(f"\nStage 5 done in {elapsed:.0f}s")

    with open(DATA / "rq2_infogain_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # ── Summary ──
    method_names = list(methods.keys())
    flush_print("\n" + "=" * 80)
    flush_print("RESULTS: Information-Gain Methods")
    flush_print("=" * 80)

    for K in Ks:
        flush_print(f"\n  K={K}:")
        flush_print(f"  {'Method':<14s} {'Coverage':>10s} {'APD':>8s}")
        flush_print(f"  {'-'*14} {'-'*10} {'-'*8}")
        for method in method_names:
            rows = [r for r in all_results if r["method"] == method and r["K"] == K]
            if not rows:
                continue
            mean_cov = statistics.mean(r["coverage"] for r in rows)
            mean_apd = statistics.mean(r["apd"] for r in rows)
            flush_print(f"  {method:<14s} {mean_cov*100:>9.1f}% {mean_apd:>8.3f}")

    # Correlations
    flush_print("\n" + "=" * 80)
    flush_print("CORRELATIONS: APD vs Coverage@K (Info-Gain methods)")
    flush_print("=" * 80)
    for K in Ks:
        flush_print(f"\n  K={K}:")
        for method in method_names:
            rows = [r for r in all_results if r["method"] == method and r["K"] == K]
            if len(rows) < 5:
                continue
            covs = [r["coverage"] for r in rows]
            apds = [r["apd"] for r in rows]
            rho, p = spearmanr(apds, covs)
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            flush_print(f"  {method:<14s} APD↔Cov: ρ={rho:+.3f} p={p:.4f} {sig}")

    flush_print(f"\nSaved to data/processed/rq2_infogain_results.json")


# ──────────────────────────────────────────────
# Stage 6: Aggressive novelty variants
# Mimics GreedySCS's key advantage: integer-valued diversity scores
# that dominate the sigmoid reranker scores (0.5-0.9 range).
# ──────────────────────────────────────────────
def stage6_aggressive_novelty():
    flush_print("=== STAGE 6: Aggressive novelty variants ===")
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    from nltk.tokenize import sent_tokenize
    from scipy.stats import spearmanr
    import statistics

    with open(DATA / "coverage_data.json") as f:
        events = json.load(f)
    with open(DATA / "llm_coverage_labels.json") as f:
        labels = json.load(f)
    with open(DATA / "newscope_reranker_scores.json") as f:
        reranker_store = json.load(f)

    flush_print("Loading embedding model...")
    emb_model = SentenceTransformer('Lajavaness/bilingual-embedding-large', trust_remote_code=True)
    flush_print("Model loaded.")

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

        reranker_scores = reranker_store[eid]

        # Load paragraph embeddings for APD
        npz_path = DATA / f"emb_{eid}.npz"
        if not npz_path.exists():
            continue
        npz = np.load(npz_path, allow_pickle=True)
        emb_pids = list(npz["pids"])
        emb_matrix = npz["embeddings"]
        para_emb_dict = {pid: emb_matrix[i] for i, pid in enumerate(emb_pids)}

        # ── Sentence segmentation (nltk) ──
        all_sentences = []
        sent_to_para = []
        para_to_sents = defaultdict(list)

        for pid_local, text in enumerate(rel_texts):
            for sent in sent_tokenize(text):
                if '\n' in sent:
                    for s in sent.split("\n"):
                        if len(s.split(" ")) > 5:
                            sid = len(all_sentences)
                            all_sentences.append(s)
                            sent_to_para.append(pid_local)
                            para_to_sents[pid_local].append(sid)
                elif len(sent.split(" ")) > 5 and sent.strip():
                    sid = len(all_sentences)
                    all_sentences.append(sent)
                    sent_to_para.append(pid_local)
                    para_to_sents[pid_local].append(sid)

        if len(all_sentences) < 2:
            continue

        # Compute sentence embeddings + similarity
        sent_embs = emb_model.encode(all_sentences, show_progress_bar=False, batch_size=64)
        sent_sim = cosine_similarity(sent_embs)

        rel_scores = [reranker_scores.get(pid, 0.0) for pid in rel_pids]

        # ── Method A: SentNovelty with high λ, low threshold ──
        # λ=3.0 means each novel sentence fraction is worth 3x a perfect reranker score
        # threshold=0.5 means more sentences count as "novel"
        def sent_novelty_aggressive(K, lam=3.0, threshold=0.5):
            selected_paras = []
            selected_sents = set()
            remaining = list(range(len(rel_texts)))
            for _ in range(min(K, len(rel_texts))):
                best_pid, best_score = None, -1e9
                for pid in remaining:
                    sids = para_to_sents[pid]
                    if not sids:
                        novelty = 0.0
                    elif not selected_sents:
                        novelty = 1.0
                    else:
                        sel_list = list(selected_sents)
                        novel_count = 0
                        for sid in sids:
                            max_sim = max(sent_sim[sid][ss] for ss in sel_list)
                            if max_sim < threshold:
                                novel_count += 1
                        novelty = novel_count / len(sids)
                    score = lam * novelty + rel_scores[pid]
                    if score > best_score:
                        best_score = score
                        best_pid = pid
                if best_pid is None:
                    break
                selected_paras.append(best_pid)
                remaining.remove(best_pid)
                selected_sents.update(para_to_sents[best_pid])
            return [rel_pids[i] for i in selected_paras]

        # ── Method B: Binary novelty count (mimics GreedySCS integer scoring) ──
        # Score = w * (number of novel sentences) + reranker_score
        # This directly parallels GreedySCS: w * n_new_clusters + reranker_score
        # Novel = max sim to any selected sentence < threshold
        def binary_novelty_count(K, w=1.0, threshold=0.7):
            selected_paras = []
            selected_sents = set()
            remaining = list(range(len(rel_texts)))
            for _ in range(min(K, len(rel_texts))):
                best_pid, best_score = None, -1e9
                for pid in remaining:
                    sids = para_to_sents[pid]
                    if not sids:
                        novel_count = 0
                    elif not selected_sents:
                        novel_count = len(sids)
                    else:
                        sel_list = list(selected_sents)
                        novel_count = 0
                        for sid in sids:
                            max_sim = max(sent_sim[sid][ss] for ss in sel_list)
                            if max_sim < threshold:
                                novel_count += 1
                    # Integer count + sigmoid score (like GreedySCS)
                    score = w * novel_count + rel_scores[pid]
                    if score > best_score:
                        best_score = score
                        best_pid = pid
                if best_pid is None:
                    break
                selected_paras.append(best_pid)
                remaining.remove(best_pid)
                selected_sents.update(para_to_sents[best_pid])
            return [rel_pids[i] for i in selected_paras]

        # ── Evaluate ──
        methods = {
            "SentNov-Agg": lambda K: sent_novelty_aggressive(K, lam=3.0, threshold=0.5),
            "BinNovelty": lambda K: binary_novelty_count(K, w=1.0, threshold=0.7),
            "BinNov-Low": lambda K: binary_novelty_count(K, w=1.0, threshold=0.5),
            "BinNov-W2": lambda K: binary_novelty_count(K, w=2.0, threshold=0.7),
        }

        for K in Ks:
            for method_name, method_fn in methods.items():
                try:
                    selected_pids = method_fn(K)
                except Exception as e:
                    flush_print(f"  ERROR {method_name} on {eid} K={K}: {e}")
                    selected_pids = sorted(rel_pids,
                                           key=lambda p: reranker_scores.get(p, 0),
                                           reverse=True)[:K]

                cov_score = coverage_at_k(event, cov, selected_pids)
                sel_embs = np.array([para_emb_dict[p] for p in selected_pids
                                     if p in para_emb_dict])
                apd = compute_apd(sel_embs) if len(sel_embs) >= 2 else 0

                all_results.append({
                    "event_id": eid, "method": method_name, "K": K,
                    "coverage": float(cov_score), "apd": float(apd)
                })

        if (ei + 1) % 10 == 0:
            elapsed = time.time() - t_start
            flush_print(f"  [{ei+1}/{len(events)}] {elapsed:.0f}s elapsed")
            with open(DATA / "rq2_aggressive_results.json", "w") as f:
                json.dump(all_results, f, indent=2)

    elapsed = time.time() - t_start
    flush_print(f"\nStage 6 done in {elapsed:.0f}s")

    with open(DATA / "rq2_aggressive_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # ── Summary ──
    method_names = list(methods.keys())
    flush_print("\n" + "=" * 80)
    flush_print("RESULTS: Aggressive Novelty Variants")
    flush_print("=" * 80)

    for K in Ks:
        flush_print(f"\n  K={K}:")
        flush_print(f"  {'Method':<14s} {'Coverage':>10s} {'APD':>8s}")
        flush_print(f"  {'-'*14} {'-'*10} {'-'*8}")
        for method in method_names:
            rows = [r for r in all_results if r["method"] == method and r["K"] == K]
            if not rows:
                continue
            mean_cov = statistics.mean(r["coverage"] for r in rows)
            mean_apd = statistics.mean(r["apd"] for r in rows)
            flush_print(f"  {method:<14s} {mean_cov*100:>9.1f}% {mean_apd:>8.3f}")

    # Correlations
    flush_print("\n" + "=" * 80)
    flush_print("CORRELATIONS: APD vs Coverage@K (Aggressive variants)")
    flush_print("=" * 80)
    for K in Ks:
        flush_print(f"\n  K={K}:")
        for method in method_names:
            rows = [r for r in all_results if r["method"] == method and r["K"] == K]
            if len(rows) < 5:
                continue
            covs = [r["coverage"] for r in rows]
            apds = [r["apd"] for r in rows]
            rho, p = spearmanr(apds, covs)
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            flush_print(f"  {method:<14s} APD↔Cov: ρ={rho:+.3f} p={p:.4f} {sig}")

    flush_print(f"\nSaved to data/processed/rq2_aggressive_results.json")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        stage = sys.argv[1]
        if stage == "1":
            stage1_embeddings()
        elif stage == "2":
            stage2_reranker()
        elif stage == "3":
            stage3_cluster_and_evaluate()
        elif stage == "4":
            stage4_diversity_mechanisms()
        elif stage == "5":
            stage5_information_gain()
        elif stage == "6":
            stage6_aggressive_novelty()
        else:
            flush_print(f"Unknown stage: {stage}. Use 1-6.")
    else:
        stage1_embeddings()
        gc.collect()
        stage2_reranker()
        gc.collect()
        stage3_cluster_and_evaluate()
        gc.collect()
        stage4_diversity_mechanisms()
        gc.collect()
        stage5_information_gain()
        gc.collect()
        stage6_aggressive_novelty()
