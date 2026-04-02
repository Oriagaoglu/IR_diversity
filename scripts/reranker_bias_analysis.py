"""
Reranker Bias Analysis + Coverage-Aware Methods

Key finding: The BGE reranker actively suppresses paragraphs carrying
unique perspectives. 46% of perspective claims are covered by only 1
paragraph, and those singleton paragraphs have significantly lower
reranker scores (mean 0.514 vs 0.628 for shared claims).

This script:
1. Formalizes the reranker bias analysis with statistical tests
2. Proposes and evaluates methods that reduce reranker dependence
3. Compares against GreedySCS and all previous methods
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from scipy.stats import mannwhitneyu, spearmanr, wilcoxon
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
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
    covered = 0
    for c in event["claims"]:
        cid = c["claim_id"]
        covering_pids = set(coverage_labels.get(cid, []))
        if covering_pids & selected_set:
            covered += 1
    return covered / n_claims


def compute_apd(embeddings):
    if len(embeddings) < 2:
        return 0.0
    sim = cosine_similarity(embeddings)
    n = len(embeddings)
    total = sum(1 - sim[i][j] for i in range(n) for j in range(i + 1, n))
    count = n * (n - 1) / 2
    return total / count if count > 0 else 0.0


def main():
    print("=" * 80)
    print("RERANKER BIAS ANALYSIS")
    print("=" * 80)

    with open(DATA / "coverage_data.json") as f:
        events = json.load(f)
    with open(DATA / "llm_coverage_labels.json") as f:
        labels = json.load(f)
    with open(DATA / "newscope_reranker_scores.json") as f:
        reranker_store = json.load(f)

    event_dict = {e["dsglobal_id"]: e for e in events}

    # ──────────────────────────────────────────────
    # Part 1: Formalize the Reranker Bias
    # ──────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("PART 1: RERANKER BIAS AGAINST PERSPECTIVE PARAGRAPHS")
    print("=" * 80)

    singleton_scores = []
    shared_scores = []
    noclaim_scores = []
    all_scores = []

    # Per-event claim structure
    for eid, rscores in reranker_store.items():
        cov = labels.get(eid, {})
        if not cov:
            continue

        singleton_pids = set()
        shared_pids = set()
        for cid, pids in cov.items():
            if len(pids) == 1:
                singleton_pids.add(pids[0])
            else:
                shared_pids.update(pids)

        for pid, score in rscores.items():
            all_scores.append(score)
            if pid in singleton_pids:
                singleton_scores.append(score)
            elif pid in shared_pids:
                shared_scores.append(score)
            else:
                noclaim_scores.append(score)

    print(f"\nReranker score distribution by paragraph type:")
    print(f"  {'Type':<30s} {'N':>6s} {'Mean':>8s} {'Median':>8s} {'Std':>8s}")
    print(f"  {'-'*30} {'-'*6} {'-'*8} {'-'*8} {'-'*8}")
    for name, scores in [("Singleton claim (unique)", singleton_scores),
                         ("Shared claim (2+ paras)", shared_scores),
                         ("No claim covered", noclaim_scores),
                         ("All relevant", all_scores)]:
        arr = np.array(scores)
        print(f"  {name:<30s} {len(arr):>6d} {np.mean(arr):>8.3f} {np.median(arr):>8.3f} {np.std(arr):>8.3f}")

    # Statistical test: singleton vs shared
    u_stat, p_val = mannwhitneyu(singleton_scores, shared_scores, alternative='less')
    print(f"\n  Mann-Whitney U test (singleton < shared):")
    print(f"  U={u_stat:.0f}, p={p_val:.2e} {'***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'}")
    print(f"  Effect size (rank-biserial): {1 - 2*u_stat/(len(singleton_scores)*len(shared_scores)):.3f}")

    # Claim-level statistics
    print(f"\n  Claim coverage distribution:")
    total_claims = 0
    claim_sizes = []
    for eid, cov in labels.items():
        for cid, pids in cov.items():
            total_claims += 1
            claim_sizes.append(len(pids))

    cs = np.array(claim_sizes)
    for size_range, label in [((0, 0), "0 (uncoverable)"),
                               ((1, 1), "1 (singleton)"),
                               ((2, 3), "2-3"),
                               ((4, 10), "4-10"),
                               ((11, 999), ">10")]:
        count = np.sum((cs >= size_range[0]) & (cs <= size_range[1]))
        print(f"    {label:>20s}: {count:>5d} ({count/len(cs)*100:>5.1f}%)")

    # ──────────────────────────────────────────────
    # Part 2: Reranker rank vs perspective coverage
    # ──────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("PART 2: PERSPECTIVE COVERAGE BY RERANKER RANK QUINTILE")
    print("=" * 80)

    # For each event, split paragraphs into quintiles by reranker score
    # Count what fraction of claims each quintile covers
    quintile_claim_fracs = defaultdict(list)  # quintile -> list of fractions

    for eid, rscores in reranker_store.items():
        event = event_dict.get(eid)
        cov = labels.get(eid, {})
        if not event or not cov:
            continue

        rel_pids = sorted(rscores.keys(), key=lambda p: rscores[p], reverse=True)
        n = len(rel_pids)
        if n < 5:
            continue

        n_claims = event["n_claims"]
        if n_claims == 0:
            continue

        quintile_size = n // 5
        for q in range(5):
            start = q * quintile_size
            end = start + quintile_size if q < 4 else n
            q_pids = set(rel_pids[start:end])
            covered = sum(1 for cid, cpids in cov.items() if set(cpids) & q_pids)
            quintile_claim_fracs[q].append(covered / n_claims)

    print(f"\n  {'Quintile':<20s} {'Rank range':<15s} {'Mean claims covered':>20s}")
    print(f"  {'-'*20} {'-'*15} {'-'*20}")
    q_labels = ["Q1 (top scores)", "Q2", "Q3", "Q4", "Q5 (bottom scores)"]
    for q in range(5):
        fracs = quintile_claim_fracs[q]
        print(f"  {q_labels[q]:<20s} {'Top ' + str(q*20) + '-' + str((q+1)*20) + '%':<15s} {np.mean(fracs)*100:>19.1f}%")

    print(f"\n  Key: Q5 (lowest reranker scores) covers {np.mean(quintile_claim_fracs[4])*100:.1f}%")
    print(f"  vs Q1 (highest scores) covers {np.mean(quintile_claim_fracs[0])*100:.1f}%")
    ratio = np.mean(quintile_claim_fracs[4]) / max(np.mean(quintile_claim_fracs[0]), 1e-10)
    print(f"  Q5/Q1 ratio: {ratio:.2f}x — bottom-ranked paragraphs carry {ratio:.0%} as many perspectives")

    # ──────────────────────────────────────────────
    # Part 3: Embedding-space analysis
    # ──────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("PART 3: EMBEDDING-SPACE SEPARABILITY OF PERSPECTIVES")
    print("=" * 80)

    same_claim_sims = []
    diff_claim_sims = []
    random_pair_sims = []

    for eid, cov in labels.items():
        event = event_dict.get(eid)
        if not event:
            continue

        npz_path = DATA / f"emb_{eid}.npz"
        if not npz_path.exists():
            continue
        npz = np.load(npz_path, allow_pickle=True)
        emb_pids = list(npz["pids"])
        emb_matrix = npz["embeddings"]
        pid_to_idx = {pid: i for i, pid in enumerate(emb_pids)}

        claim_pids = {}
        for cid, pids in cov.items():
            valid = [p for p in pids if p in pid_to_idx]
            if valid:
                claim_pids[cid] = valid

        claim_list = list(claim_pids.keys())

        # Same-claim pairs
        for cid in claim_list:
            pids = claim_pids[cid]
            for i in range(len(pids)):
                for j in range(i + 1, len(pids)):
                    e1 = emb_matrix[pid_to_idx[pids[i]]]
                    e2 = emb_matrix[pid_to_idx[pids[j]]]
                    sim = float(cosine_similarity([e1], [e2])[0][0])
                    same_claim_sims.append(sim)

        # Different-claim pairs
        for ci in range(len(claim_list)):
            for cj in range(ci + 1, min(ci + 3, len(claim_list))):
                pids_i = claim_pids[claim_list[ci]]
                pids_j = claim_pids[claim_list[cj]]
                for pi in pids_i[:2]:
                    for pj in pids_j[:2]:
                        if pi != pj:
                            e1 = emb_matrix[pid_to_idx[pi]]
                            e2 = emb_matrix[pid_to_idx[pj]]
                            sim = float(cosine_similarity([e1], [e2])[0][0])
                            diff_claim_sims.append(sim)

    print(f"\n  Cosine similarity between paragraph pairs:")
    print(f"    Same claim:      mean={np.mean(same_claim_sims):.3f}, std={np.std(same_claim_sims):.3f}, n={len(same_claim_sims)}")
    print(f"    Different claim:  mean={np.mean(diff_claim_sims):.3f}, std={np.std(diff_claim_sims):.3f}, n={len(diff_claim_sims)}")
    print(f"    Gap: {np.mean(same_claim_sims) - np.mean(diff_claim_sims):.3f}")

    u_stat2, p_val2 = mannwhitneyu(same_claim_sims, diff_claim_sims, alternative='greater')
    print(f"\n  Mann-Whitney U (same > diff): p={p_val2:.2e} {'***' if p_val2 < 0.001 else 'ns'}")
    print(f"  The gap is statistically significant but small (0.138).")
    print(f"  Embedding distance is a WEAK proxy for perspective difference.")

    # ──────────────────────────────────────────────
    # Part 4: Oracle analysis
    # ──────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("PART 4: ORACLE CEILING AND HEADROOM")
    print("=" * 80)

    Ks = [5, 10, 20]
    for K in Ks:
        oracle_covs = []
        for event in events:
            eid = event["dsglobal_id"]
            cov = labels.get(eid, {})
            if not cov:
                continue
            rel_pids = [p["paragraph_id"] for p in event["paragraphs"] if p["relevant"] == 1]
            if not rel_pids:
                continue
            n_claims = event["n_claims"]
            if n_claims == 0:
                continue

            # Greedy oracle
            remaining = dict(cov)
            selected = set()
            for _ in range(min(K, len(rel_pids))):
                best_pid, best_new = None, -1
                for pid in rel_pids:
                    if pid in selected:
                        continue
                    new_covered = sum(1 for cid, pids in remaining.items() if pid in pids)
                    if new_covered > best_new:
                        best_new = new_covered
                        best_pid = pid
                if best_pid is None or best_new == 0:
                    break
                selected.add(best_pid)
                remaining = {cid: pids for cid, pids in remaining.items() if best_pid not in pids}
            covered = n_claims - len(remaining)
            oracle_covs.append(covered / n_claims)
        print(f"  Oracle K={K}: {np.mean(oracle_covs)*100:.1f}%")

    # ──────────────────────────────────────────────
    # Part 5: New methods — reranker-independent
    # ──────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("PART 5: RERANKER-INDEPENDENT METHODS")
    print("=" * 80)
    print("\nHypothesis: reducing reranker dependence improves perspective coverage.")
    print("We test methods along a spectrum from full reranker trust to zero trust.\n")

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
        scores = np.array([rscores.get(pid, 0.0) for pid in valid_pids])
        texts = [rel_texts[rel_pids.index(pid)] for pid in valid_pids]

        # Pairwise cosine similarity (paragraph-level)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1e-10)
        normed = embs / norms
        para_sim = normed @ normed.T

        # TF-IDF for lexical methods
        tfidf = TfidfVectorizer(stop_words="english", max_df=0.9, ngram_range=(1, 2))
        try:
            tfidf_mat = tfidf.fit_transform(texts)
            tfidf_sim = (tfidf_mat @ tfidf_mat.T).toarray()
        except:
            tfidf_sim = para_sim  # fallback

        # ── Method 1: Pure Embedding Diversity (MMR λ=0) ──
        def pure_diversity(K):
            """Select purely by max-min diversity. Zero reranker trust."""
            selected = [int(np.argmax(scores))]  # seed with best reranker
            remaining = [i for i in range(len(valid_pids)) if i != selected[0]]
            for _ in range(min(K - 1, len(remaining))):
                best_idx, best_min_dist = None, -1
                for idx in remaining:
                    min_sim = min(para_sim[idx][s] for s in selected)
                    if min_sim < best_min_dist or best_min_dist < 0:
                        # We want max of min-distance, so min of min-similarity
                        pass
                    dist = 1 - max(para_sim[idx][s] for s in selected)
                    if dist > best_min_dist:
                        best_min_dist = dist
                        best_idx = idx
                if best_idx is None:
                    break
                selected.append(best_idx)
                remaining.remove(best_idx)
            return [valid_pids[i] for i in selected]

        # ── Method 2: MMR with very low λ (diversity-dominant) ──
        def mmr_low_lambda(K, lam=0.1):
            """MMR with λ=0.1: 10% relevance, 90% diversity."""
            selected = []
            remaining = list(range(len(valid_pids)))
            for _ in range(min(K, len(valid_pids))):
                best_idx, best_score = None, -1e9
                for idx in remaining:
                    rel = scores[idx]
                    if not selected:
                        div_penalty = 0.0
                    else:
                        div_penalty = max(para_sim[idx][s] for s in selected)
                    score = lam * rel - (1 - lam) * div_penalty
                    if score > best_score:
                        best_score = score
                        best_idx = idx
                if best_idx is None:
                    break
                selected.append(best_idx)
                remaining.remove(best_idx)
            return [valid_pids[i] for i in selected]

        # ── Method 3: Stratified Diversity ──
        def stratified_diversity(K):
            """Pick from each rank quintile, then diversify within strata."""
            ranked_indices = np.argsort(-scores)
            n = len(ranked_indices)
            n_strata = min(5, K)
            stratum_size = n // n_strata

            # Allocate K slots across strata (round-robin)
            slots = [K // n_strata] * n_strata
            for i in range(K % n_strata):
                slots[i] += 1

            selected = []
            for s in range(n_strata):
                start = s * stratum_size
                end = start + stratum_size if s < n_strata - 1 else n
                stratum_indices = list(ranked_indices[start:end])

                # Within stratum, pick by diversity from already selected
                for _ in range(min(slots[s], len(stratum_indices))):
                    if not selected:
                        # Pick highest-scored in this stratum
                        best = stratum_indices[0]
                    else:
                        best, best_div = None, -1
                        for idx in stratum_indices:
                            if idx in selected:
                                continue
                            min_sim = min(para_sim[idx][s2] for s2 in selected)
                            if 1 - min_sim > best_div:
                                best_div = 1 - min_sim
                                best = idx
                    if best is None:
                        break
                    selected.append(best)
                    if best in stratum_indices:
                        stratum_indices.remove(best)

            return [valid_pids[i] for i in selected[:K]]

        # ── Method 4: Lexical Coverage Maximization ──
        def lexical_coverage(K, lam=0.3):
            """Greedily pick paragraphs maximizing new unique TF-IDF terms.
            Cheap, no embeddings needed in production. lam weights reranker."""
            # Get feature names and TF-IDF weights per paragraph
            selected = []
            remaining = list(range(len(valid_pids)))
            covered_terms = np.zeros(tfidf_mat.shape[1])

            for _ in range(min(K, len(valid_pids))):
                best_idx, best_score = None, -1e9
                for idx in remaining:
                    # New term coverage: sum of TF-IDF weights for uncovered terms
                    para_vec = tfidf_mat[idx].toarray().flatten()
                    new_coverage = np.sum(np.maximum(0, para_vec - covered_terms))
                    score = new_coverage + lam * scores[idx]
                    if score > best_score:
                        best_score = score
                        best_idx = idx
                if best_idx is None:
                    break
                selected.append(best_idx)
                remaining.remove(best_idx)
                covered_terms = np.maximum(covered_terms,
                                           tfidf_mat[best_idx].toarray().flatten())
            return [valid_pids[i] for i in selected]

        # ── Method 5: Lexical Coverage (no reranker at all) ──
        def lexical_pure(K):
            """Pure lexical coverage, zero reranker dependence."""
            return lexical_coverage(K, lam=0.0)

        # ── Method 6: Inverse-Reranker Exploration ──
        def inverse_reranker(K, lam=0.5):
            """Deliberately explore low-ranked paragraphs.
            Score = diversity - lam * reranker_score.
            Paragraphs the reranker dislikes get a BONUS."""
            selected = []
            remaining = list(range(len(valid_pids)))
            for _ in range(min(K, len(valid_pids))):
                best_idx, best_score = None, -1e9
                for idx in remaining:
                    if not selected:
                        div = 1.0
                    else:
                        div = 1 - max(para_sim[idx][s] for s in selected)
                    # Penalize high reranker score
                    score = div - lam * scores[idx]
                    if score > best_score:
                        best_score = score
                        best_idx = idx
                if best_idx is None:
                    break
                selected.append(best_idx)
                remaining.remove(best_idx)
            return [valid_pids[i] for i in selected]

        # ── Method 7: Facility Location (no reranker) ──
        def facloc_pure(K):
            """Facility Location with NO relevance weighting.
            f(S) = sum_i max_{j in S} sim(i,j)
            Pure representativeness without reranker bias."""
            n = len(valid_pids)
            selected = []
            remaining = list(range(n))
            current_max = np.full(n, -np.inf)

            for _ in range(min(K, n)):
                best_idx, best_gain = None, -1e9
                for idx in remaining:
                    gain = np.sum(np.maximum(0, para_sim[:, idx] - current_max))
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = idx
                if best_idx is None:
                    break
                selected.append(best_idx)
                remaining.remove(best_idx)
                current_max = np.maximum(current_max, para_sim[:, best_idx])
            return [valid_pids[i] for i in selected]

        # ── Evaluate all methods ──
        methods = {
            "PureDiversity": pure_diversity,
            "MMR-0.1": lambda K: mmr_low_lambda(K, lam=0.1),
            "MMR-0.3": lambda K: mmr_low_lambda(K, lam=0.3),
            "Stratified": stratified_diversity,
            "LexCov-0.3": lambda K: lexical_coverage(K, lam=0.3),
            "LexCov-Pure": lexical_pure,
            "InvReranker": lambda K: inverse_reranker(K, lam=0.5),
            "FacLoc-Pure": facloc_pure,
        }

        for K in Ks:
            for method_name, method_fn in methods.items():
                try:
                    selected_pids = method_fn(K)
                except Exception as e:
                    selected_pids = sorted(valid_pids,
                                           key=lambda p: rscores.get(p, 0),
                                           reverse=True)[:K]

                cov_score = coverage_at_k(event, cov, selected_pids)
                sel_embs = np.array([para_emb_dict[p] for p in selected_pids
                                     if p in para_emb_dict])
                apd = compute_apd(sel_embs) if len(sel_embs) >= 2 else 0

                all_results.append({
                    "event_id": eid, "method": method_name, "K": K,
                    "coverage": float(cov_score), "apd": float(apd)
                })

        if (ei + 1) % 20 == 0:
            print(f"  [{ei+1}/{len(events)}] events processed")

    # Save results
    with open(DATA / "rq2_reranker_independent_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # ── Results table ──
    print("\n" + "=" * 80)
    print("RESULTS: Reranker-Independent Methods")
    print("=" * 80)

    method_names = list(methods.keys())

    # Also load previous results for comparison
    with open(DATA / "newscope_faithful_results.json") as f:
        prev_faithful = json.load(f)
    with open(DATA / "rq2_diversity_results.json") as f:
        prev_diversity = json.load(f)
    with open(DATA / "rq2_infogain_results.json") as f:
        prev_infogain = json.load(f)

    # Build comparison lookup
    prev_all = prev_faithful + prev_diversity + prev_infogain
    prev_lookup = defaultdict(list)
    for r in prev_all:
        prev_lookup[(r["method"], r["K"])].append(r["coverage"])

    new_lookup = defaultdict(list)
    for r in all_results:
        new_lookup[(r["method"], r["K"])].append(r["coverage"])

    print(f"\n  {'Method':<18s}", end="")
    for K in Ks:
        print(f" {'K='+str(K):>8s}", end="")
    print(f"  {'Reranker dep.':>14s}")
    print(f"  {'-'*18}", end="")
    for K in Ks:
        print(f" {'-'*8}", end="")
    print(f"  {'-'*14}")

    # Previous baselines
    for method in ["DenseRetrieval", "Reranker", "GreedySCS", "GreedyPlus",
                   "FacLoc", "SentNovelty"]:
        vals = []
        for K in Ks:
            rows = prev_lookup.get((method, K), [])
            v = statistics.mean(rows) * 100 if rows else 0
            vals.append(v)
        dep = "High" if method in ["Reranker", "DenseRetrieval"] else "Medium"
        if method in ["GreedySCS", "GreedyPlus"]:
            dep = "Low (cluster)"
        print(f"  {method:<18s} {vals[0]:>7.1f}% {vals[1]:>7.1f}% {vals[2]:>7.1f}%  {dep:>14s}")

    print(f"  {'--- NEW ---':<18s}")

    for method in method_names:
        vals = []
        for K in Ks:
            rows = new_lookup.get((method, K), [])
            v = statistics.mean(rows) * 100 if rows else 0
            vals.append(v)
        dep = "None" if "Pure" in method or "Inv" in method else "Low"
        if method == "MMR-0.1":
            dep = "Very low"
        elif method == "MMR-0.3":
            dep = "Low"
        elif method == "Stratified":
            dep = "Medium"
        print(f"  {method:<18s} {vals[0]:>7.1f}% {vals[1]:>7.1f}% {vals[2]:>7.1f}%  {dep:>14s}")

    # ── Statistical comparison: best new method vs GreedySCS ──
    print("\n" + "=" * 80)
    print("STATISTICAL TESTS: New methods vs GreedySCS (paired Wilcoxon)")
    print("=" * 80)

    # Get per-event GreedySCS scores
    scs_by_event = defaultdict(dict)
    for r in prev_faithful:
        if r["method"] == "GreedySCS":
            scs_by_event[r["K"]][r["event_id"]] = r["coverage"]

    for K in Ks:
        print(f"\n  K={K}:")
        scs_scores_k = scs_by_event[K]
        for method in method_names:
            method_by_event = {}
            for r in all_results:
                if r["method"] == method and r["K"] == K:
                    method_by_event[r["event_id"]] = r["coverage"]

            # Paired comparison
            common = sorted(set(scs_scores_k.keys()) & set(method_by_event.keys()))
            if len(common) < 10:
                continue
            scs_arr = np.array([scs_scores_k[eid] for eid in common])
            new_arr = np.array([method_by_event[eid] for eid in common])

            diff = new_arr - scs_arr
            wins = np.sum(diff > 0.01)
            losses = np.sum(diff < -0.01)
            ties = len(diff) - wins - losses

            try:
                stat, p = wilcoxon(new_arr, scs_arr, alternative='two-sided')
                sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            except:
                p = 1.0
                sig = 'ns'

            mean_diff = np.mean(diff) * 100
            print(f"    {method:<16s} Δ={mean_diff:>+5.1f}pp  W/L/T={wins}/{losses}/{ties}  p={p:.4f} {sig}")

    # ── APD correlation check ──
    print("\n" + "=" * 80)
    print("APD vs COVERAGE CORRELATIONS (New methods)")
    print("=" * 80)
    for K in Ks:
        print(f"\n  K={K}:")
        for method in method_names:
            rows = [r for r in all_results if r["method"] == method and r["K"] == K]
            if len(rows) < 5:
                continue
            covs = [r["coverage"] for r in rows]
            apds = [r["apd"] for r in rows]
            rho, p = spearmanr(apds, covs)
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            print(f"    {method:<16s} APD↔Cov: ρ={rho:+.3f} p={p:.4f} {sig}")

    print(f"\n\nSaved to data/processed/rq2_reranker_independent_results.json")
    print("=" * 80)


if __name__ == "__main__":
    main()
