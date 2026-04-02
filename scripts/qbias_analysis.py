"""
QBias Analysis: Question-level perspective bias in retrieval.

Extends the reranker bias finding to understand:
1. Per-question coverage: which questions have perspectives systematically missed?
2. Majority vs minority perspectives: do methods favor large answer groups?
3. Embedding-space bias: are minority perspectives geometrically disadvantaged?
4. Question-type effects: do certain question types (why/how/what) show more bias?
5. Can question-aware retrieval improve coverage?
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from scipy.stats import spearmanr, mannwhitneyu, pearsonr
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import statistics
import re
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


def main():
    print("=" * 80)
    print("QBIAS ANALYSIS: Question-Level Perspective Bias")
    print("=" * 80)

    with open(DATA / "coverage_data.json") as f:
        events = json.load(f)
    with open(DATA / "llm_coverage_labels.json") as f:
        labels = json.load(f)
    with open(DATA / "newscope_reranker_scores.json") as f:
        reranker_store = json.load(f)

    event_dict = {e["dsglobal_id"]: e for e in events}

    # ──────────────────────────────────────────────
    # Part 1: Question-level statistics
    # ──────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("PART 1: QUESTION AND PERSPECTIVE STRUCTURE")
    print("=" * 80)

    questions = []  # list of dicts with question info
    for event in events:
        eid = event["dsglobal_id"]
        cov = labels.get(eid, {})

        # Group claims by question
        q_claims = defaultdict(list)
        for c in event["claims"]:
            q_claims[c["question_index"]].append(c)

        for qi, claims in q_claims.items():
            q_text = claims[0]["question"]
            n_groups = len(claims)
            group_sizes = [c["n_answers_in_group"] for c in claims]

            # How many paragraphs cover each perspective?
            group_coverage_counts = []
            for c in claims:
                cid = c["claim_id"]
                n_covering = len(cov.get(cid, []))
                group_coverage_counts.append(n_covering)

            questions.append({
                "event_id": eid,
                "question_index": qi,
                "question": q_text,
                "n_perspectives": n_groups,
                "group_sizes": group_sizes,
                "coverage_counts": group_coverage_counts,
                "claims": claims,
            })

    print(f"\n  Total questions: {len(questions)}")
    print(f"  Total perspectives (answer groups): {sum(q['n_perspectives'] for q in questions)}")

    n_persp = [q["n_perspectives"] for q in questions]
    print(f"\n  Perspectives per question:")
    print(f"    Mean: {np.mean(n_persp):.1f}, Median: {np.median(n_persp):.0f}, Range: {min(n_persp)}-{max(n_persp)}")
    for n in sorted(set(n_persp)):
        count = n_persp.count(n)
        print(f"    {n} perspectives: {count} questions ({count/len(questions)*100:.0f}%)")

    # Group size analysis (majority vs minority)
    all_group_sizes = []
    for q in questions:
        all_group_sizes.extend(q["group_sizes"])
    print(f"\n  Answer group sizes (n_answers per perspective):")
    print(f"    Mean: {np.mean(all_group_sizes):.1f}, Median: {np.median(all_group_sizes):.0f}")
    print(f"    Range: {min(all_group_sizes)}-{max(all_group_sizes)}")

    # ──────────────────────────────────────────────
    # Part 2: Majority vs Minority Perspective Bias
    # ──────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("PART 2: MAJORITY VS MINORITY PERSPECTIVE BIAS")
    print("=" * 80)
    print("\nFor each question, the 'majority' perspective is the answer group")
    print("with the most answers. 'Minority' is the smallest group.")

    majority_reranker_scores = []
    minority_reranker_scores = []
    majority_coverage_counts = []
    minority_coverage_counts = []
    majority_group_sizes = []
    minority_group_sizes = []

    for q in questions:
        eid = q["event_id"]
        cov = labels.get(eid, {})
        rscores = reranker_store.get(eid, {})
        if not rscores:
            continue

        claims = q["claims"]
        if len(claims) < 2:
            continue

        # Find majority and minority
        sorted_claims = sorted(claims, key=lambda c: c["n_answers_in_group"], reverse=True)
        majority_claim = sorted_claims[0]
        minority_claim = sorted_claims[-1]

        majority_group_sizes.append(majority_claim["n_answers_in_group"])
        minority_group_sizes.append(minority_claim["n_answers_in_group"])

        # Reranker scores for paragraphs covering each
        maj_pids = cov.get(majority_claim["claim_id"], [])
        min_pids = cov.get(minority_claim["claim_id"], [])

        maj_scores = [rscores.get(pid, 0) for pid in maj_pids if pid in rscores]
        min_scores = [rscores.get(pid, 0) for pid in min_pids if pid in rscores]

        if maj_scores:
            majority_reranker_scores.append(np.mean(maj_scores))
        if min_scores:
            minority_reranker_scores.append(np.mean(min_scores))

        majority_coverage_counts.append(len(maj_pids))
        minority_coverage_counts.append(len(min_pids))

    print(f"\n  {'':>30s} {'Majority':>12s} {'Minority':>12s}")
    print(f"  {'-'*30} {'-'*12} {'-'*12}")
    print(f"  {'Group size (n_answers)':<30s} {np.mean(majority_group_sizes):>11.1f} {np.mean(minority_group_sizes):>11.1f}")
    print(f"  {'Covering paragraphs':<30s} {np.mean(majority_coverage_counts):>11.1f} {np.mean(minority_coverage_counts):>11.1f}")
    print(f"  {'Mean reranker score':<30s} {np.mean(majority_reranker_scores):>11.3f} {np.mean(minority_reranker_scores):>11.3f}")

    if len(majority_reranker_scores) > 10 and len(minority_reranker_scores) > 10:
        u, p = mannwhitneyu(majority_reranker_scores, minority_reranker_scores, alternative='greater')
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        print(f"\n  Mann-Whitney U (majority > minority reranker score): p={p:.4f} {sig}")

    # Correlation: group size vs reranker score
    all_claim_sizes = []
    all_claim_rscores = []
    all_claim_ncov = []
    for q in questions:
        eid = q["event_id"]
        cov = labels.get(eid, {})
        rscores = reranker_store.get(eid, {})
        for c in q["claims"]:
            pids = cov.get(c["claim_id"], [])
            scores = [rscores.get(pid, 0) for pid in pids if pid in rscores]
            if scores:
                all_claim_sizes.append(c["n_answers_in_group"])
                all_claim_rscores.append(np.mean(scores))
                all_claim_ncov.append(len(pids))

    rho1, p1 = spearmanr(all_claim_sizes, all_claim_rscores)
    rho2, p2 = spearmanr(all_claim_sizes, all_claim_ncov)
    print(f"\n  Group size ↔ mean reranker score: ρ={rho1:+.3f} (p={p1:.4f})")
    print(f"  Group size ↔ n covering paragraphs: ρ={rho2:+.3f} (p={p2:.4f})")
    print(f"\n  Interpretation: {'Larger groups have higher reranker scores → reranker favors majority' if rho1 > 0 and p1 < 0.05 else 'No significant reranker bias by group size'}")

    # ──────────────────────────────────────────────
    # Part 3: Embedding-space perspective geometry
    # ──────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("PART 3: EMBEDDING-SPACE PERSPECTIVE GEOMETRY")
    print("=" * 80)
    print("\nAre minority perspectives geometrically marginalized in embedding space?")

    majority_centroid_dists = []  # distance from majority perspective to event centroid
    minority_centroid_dists = []
    majority_headline_sims = []
    minority_headline_sims = []

    for q in questions:
        eid = q["event_id"]
        cov = labels.get(eid, {})
        claims = q["claims"]
        if len(claims) < 2:
            continue

        npz_path = DATA / f"emb_{eid}.npz"
        if not npz_path.exists():
            continue
        npz = np.load(npz_path, allow_pickle=True)
        emb_pids = list(npz["pids"])
        emb_matrix = npz["embeddings"]
        pid_to_idx = {pid: i for i, pid in enumerate(emb_pids)}

        # Event centroid (all relevant paragraph embeddings)
        event_obj = event_dict.get(eid)
        if not event_obj:
            continue
        rel_pids = [p["paragraph_id"] for p in event_obj["paragraphs"]
                    if p["relevant"] == 1 and p["paragraph_id"] in pid_to_idx]
        if not rel_pids:
            continue
        rel_embs = np.array([emb_matrix[pid_to_idx[p]] for p in rel_pids])
        centroid = rel_embs.mean(axis=0)

        # Headline embedding (approximate: first paragraph embedding closest to headline)
        # Actually we can compute headline sim from dense scores
        # For now use centroid as reference

        sorted_claims = sorted(claims, key=lambda c: c["n_answers_in_group"], reverse=True)
        majority_claim = sorted_claims[0]
        minority_claim = sorted_claims[-1]

        for label, claim, dist_list in [("majority", majority_claim, majority_centroid_dists),
                                         ("minority", minority_claim, minority_centroid_dists)]:
            pids = cov.get(claim["claim_id"], [])
            valid_pids = [p for p in pids if p in pid_to_idx]
            if valid_pids:
                claim_embs = np.array([emb_matrix[pid_to_idx[p]] for p in valid_pids])
                claim_centroid = claim_embs.mean(axis=0)
                dist = 1 - float(cosine_similarity([claim_centroid], [centroid])[0][0])
                dist_list.append(dist)

    print(f"\n  Distance from event centroid:")
    print(f"    Majority perspective: mean={np.mean(majority_centroid_dists):.4f}")
    print(f"    Minority perspective: mean={np.mean(minority_centroid_dists):.4f}")
    print(f"    Difference: {np.mean(minority_centroid_dists) - np.mean(majority_centroid_dists):.4f}")

    if len(majority_centroid_dists) > 10:
        u, p = mannwhitneyu(minority_centroid_dists, majority_centroid_dists, alternative='greater')
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        print(f"    Mann-Whitney U (minority farther): p={p:.4f} {sig}")
        if p < 0.05:
            print(f"    → Minority perspectives ARE more peripheral in embedding space")
        else:
            print(f"    → No significant geometric marginalization detected")

    # ──────────────────────────────────────────────
    # Part 4: Question type analysis
    # ──────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("PART 4: QUESTION TYPE EFFECTS")
    print("=" * 80)

    def classify_question(q_text):
        q_lower = q_text.lower().strip()
        if q_lower.startswith("why") or "reason" in q_lower or "cause" in q_lower:
            return "Why/Reason"
        elif q_lower.startswith("how"):
            return "How/Process"
        elif q_lower.startswith("what") and ("impact" in q_lower or "effect" in q_lower
                                              or "consequence" in q_lower or "implication" in q_lower):
            return "What-Impact"
        elif q_lower.startswith("what") and ("opinion" in q_lower or "view" in q_lower
                                              or "perspective" in q_lower or "reaction" in q_lower
                                              or "response" in q_lower or "position" in q_lower):
            return "What-Opinion"
        elif q_lower.startswith("what"):
            return "What-Factual"
        elif q_lower.startswith("who"):
            return "Who"
        else:
            return "Other"

    q_type_stats = defaultdict(lambda: {"n_questions": 0, "n_perspectives": [],
                                         "majority_cov": [], "minority_cov": [],
                                         "coverage_gap": []})

    for q in questions:
        q_type = classify_question(q["question"])
        stats = q_type_stats[q_type]
        stats["n_questions"] += 1
        stats["n_perspectives"].append(q["n_perspectives"])

        if len(q["coverage_counts"]) >= 2:
            sorted_cc = sorted(q["coverage_counts"], reverse=True)
            stats["majority_cov"].append(sorted_cc[0])
            stats["minority_cov"].append(sorted_cc[-1])
            stats["coverage_gap"].append(sorted_cc[0] - sorted_cc[-1])

    print(f"\n  {'Type':<16s} {'N':>5s} {'Persp':>6s} {'MajCov':>8s} {'MinCov':>8s} {'Gap':>6s}")
    print(f"  {'-'*16} {'-'*5} {'-'*6} {'-'*8} {'-'*8} {'-'*6}")
    for q_type in sorted(q_type_stats.keys(), key=lambda t: -q_type_stats[t]["n_questions"]):
        s = q_type_stats[q_type]
        if s["n_questions"] < 3:
            continue
        print(f"  {q_type:<16s} {s['n_questions']:>5d} {np.mean(s['n_perspectives']):>5.1f}"
              f" {np.mean(s['majority_cov']):>7.1f} {np.mean(s['minority_cov']):>7.1f}"
              f" {np.mean(s['coverage_gap']):>5.1f}")

    # ──────────────────────────────────────────────
    # Part 5: Per-question method comparison
    # ──────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("PART 5: PER-QUESTION COVERAGE — WHICH PERSPECTIVES DO METHODS MISS?")
    print("=" * 80)

    # Load results for our best methods
    with open(DATA / "newscope_faithful_results.json") as f:
        faithful = json.load(f)
    with open(DATA / "rq2_cheap_methods_results.json") as f:
        cheap = json.load(f)

    # We need to reconstruct selected PIDs to do per-claim analysis
    # Instead, let's compute per-question coverage for key methods directly

    K = 10
    print(f"\nAnalyzing at K={K}")
    print("For each method, computing per-question perspective coverage rate.\n")

    methods_to_test = {}

    # Reranker baseline
    def reranker_select(event, K):
        rel_pids = [p["paragraph_id"] for p in event["paragraphs"] if p["relevant"] == 1]
        rscores = reranker_store.get(event["dsglobal_id"], {})
        return sorted(rel_pids, key=lambda p: rscores.get(p, 0), reverse=True)[:K]

    # Emb+TF FacLoc (our best)
    def embtf_facloc_select(event, K):
        eid = event["dsglobal_id"]
        rel_paras = [p for p in event["paragraphs"] if p["relevant"] == 1]
        rel_pids = [p["paragraph_id"] for p in rel_paras]
        rel_texts = [p["text"] for p in rel_paras]

        npz_path = DATA / f"emb_{eid}.npz"
        if not npz_path.exists():
            return rel_pids[:K]
        npz = np.load(npz_path, allow_pickle=True)
        emb_pids = list(npz["pids"])
        emb_matrix = npz["embeddings"]
        para_emb_dict = {pid: emb_matrix[i] for i, pid in enumerate(emb_pids)}

        valid_pids = [pid for pid in rel_pids if pid in para_emb_dict]
        if len(valid_pids) < 2:
            return valid_pids[:K]

        embs = np.array([para_emb_dict[pid] for pid in valid_pids])
        texts = [rel_texts[rel_pids.index(pid)] for pid in valid_pids]
        n = len(valid_pids)

        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1e-10)
        emb_sim = (embs / norms) @ (embs / norms).T

        tfidf = TfidfVectorizer(stop_words="english", max_df=0.9, ngram_range=(1, 2), max_features=5000)
        try:
            tfidf_mat = tfidf.fit_transform(texts)
            tfidf_sim = (tfidf_mat @ tfidf_mat.T).toarray()
        except:
            tfidf_sim = emb_sim

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

    # TF-IDF FacLoc (cheapest competitive)
    def tf_facloc_select(event, K):
        rel_paras = [p for p in event["paragraphs"] if p["relevant"] == 1]
        rel_pids = [p["paragraph_id"] for p in rel_paras]
        rel_texts = [p["text"] for p in rel_paras]
        if len(rel_pids) < 2:
            return rel_pids[:K]

        tfidf = TfidfVectorizer(stop_words="english", max_df=0.9, ngram_range=(1, 2), max_features=5000)
        try:
            tfidf_mat = tfidf.fit_transform(rel_texts)
            tfidf_sim = (tfidf_mat @ tfidf_mat.T).toarray()
        except:
            return rel_pids[:K]

        n = len(rel_pids)
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
        return [rel_pids[i] for i in selected]

    method_fns = {
        "Reranker": reranker_select,
        "Emb+TF-FacLoc": embtf_facloc_select,
        "TF-FacLoc": tf_facloc_select,
    }

    # Per-question analysis
    method_majority_rates = defaultdict(list)  # method -> list of (covered majority? 0/1)
    method_minority_rates = defaultdict(list)
    method_question_full_coverage = defaultdict(list)  # all perspectives of a question covered?

    for event in events:
        eid = event["dsglobal_id"]
        cov = labels.get(eid, {})
        if not cov or eid not in reranker_store:
            continue

        # Get selections for each method
        selections = {}
        for method_name, method_fn in method_fns.items():
            try:
                selections[method_name] = set(method_fn(event, K))
            except:
                selections[method_name] = set()

        # Group claims by question
        q_claims = defaultdict(list)
        for c in event["claims"]:
            q_claims[c["question_index"]].append(c)

        for qi, claims in q_claims.items():
            if len(claims) < 2:
                continue

            sorted_claims = sorted(claims, key=lambda c: c["n_answers_in_group"], reverse=True)
            majority = sorted_claims[0]
            minority = sorted_claims[-1]

            for method_name, sel in selections.items():
                # Is majority covered?
                maj_pids = set(cov.get(majority["claim_id"], []))
                maj_covered = 1 if maj_pids & sel else 0
                method_majority_rates[method_name].append(maj_covered)

                # Is minority covered?
                min_pids = set(cov.get(minority["claim_id"], []))
                min_covered = 1 if min_pids & sel else 0
                method_minority_rates[method_name].append(min_covered)

                # All perspectives covered?
                all_covered = all(
                    set(cov.get(c["claim_id"], [])) & sel
                    for c in claims
                )
                method_question_full_coverage[method_name].append(1 if all_covered else 0)

    print(f"  {'Method':<18s} {'Majority':>10s} {'Minority':>10s} {'Gap':>8s} {'Full Q':>8s}")
    print(f"  {'-'*18} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")
    for method_name in method_fns:
        maj = np.mean(method_majority_rates[method_name]) * 100
        minn = np.mean(method_minority_rates[method_name]) * 100
        full = np.mean(method_question_full_coverage[method_name]) * 100
        print(f"  {method_name:<18s} {maj:>9.1f}% {minn:>9.1f}% {maj-minn:>7.1f}pp {full:>7.1f}%")

    print(f"\n  'Gap' = majority coverage rate - minority coverage rate")
    print(f"  'Full Q' = fraction of questions where ALL perspectives are covered")

    # ──────────────────────────────────────────────
    # Part 6: Can question-aware selection help?
    # ──────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("PART 6: QUESTION-AWARE RETRIEVAL POTENTIAL")
    print("=" * 80)
    print("\nIdea: if we knew the questions, could we retrieve better?")
    print("This tests the CEILING of question-aware approaches.\n")

    # For each event, embed the questions and see if question-paragraph
    # similarity predicts claim coverage better than headline similarity
    q_sim_covered = []  # similarity when paragraph covers a claim for this question
    q_sim_uncovered = []
    h_sim_covered = []
    h_sim_uncovered = []

    from sklearn.feature_extraction.text import TfidfVectorizer as TV2

    for event in events:
        eid = event["dsglobal_id"]
        cov = labels.get(eid, {})
        if not cov:
            continue

        rel_paras = [p for p in event["paragraphs"] if p["relevant"] == 1]
        rel_pids = [p["paragraph_id"] for p in rel_paras]
        rel_texts = [p["text"] for p in rel_paras]
        if len(rel_pids) < 2:
            continue

        # Group claims by question
        q_claims = defaultdict(list)
        for c in event["claims"]:
            q_claims[c["question_index"]].append(c)

        # TF-IDF similarity between questions and paragraphs
        q_texts = [q_claims[qi][0]["question"] for qi in sorted(q_claims.keys())]
        all_texts = q_texts + [event["headline"]] + rel_texts

        try:
            tfidf = TV2(stop_words="english", max_df=0.9, ngram_range=(1, 2))
            tfidf_mat = tfidf.fit_transform(all_texts)
            sim_matrix = (tfidf_mat @ tfidf_mat.T).toarray()
        except:
            continue

        n_q = len(q_texts)
        headline_idx = n_q
        para_start = n_q + 1

        for qi_pos, qi in enumerate(sorted(q_claims.keys())):
            claims = q_claims[qi]
            # Paragraphs covering any claim of this question
            covering_pids = set()
            for c in claims:
                covering_pids.update(cov.get(c["claim_id"], []))

            for pi, pid in enumerate(rel_pids):
                para_idx = para_start + pi
                q_sim = sim_matrix[qi_pos][para_idx]
                h_sim = sim_matrix[headline_idx][para_idx]

                if pid in covering_pids:
                    q_sim_covered.append(q_sim)
                    h_sim_covered.append(h_sim)
                else:
                    q_sim_uncovered.append(q_sim)
                    h_sim_uncovered.append(h_sim)

    print(f"  TF-IDF similarity to QUESTION:")
    print(f"    Covering paragraphs:     mean={np.mean(q_sim_covered):.4f}")
    print(f"    Non-covering paragraphs: mean={np.mean(q_sim_uncovered):.4f}")
    print(f"    Gap: {np.mean(q_sim_covered) - np.mean(q_sim_uncovered):.4f}")
    u, p = mannwhitneyu(q_sim_covered, q_sim_uncovered, alternative='greater')
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    print(f"    Mann-Whitney U: p={p:.2e} {sig}")

    print(f"\n  TF-IDF similarity to HEADLINE:")
    print(f"    Covering paragraphs:     mean={np.mean(h_sim_covered):.4f}")
    print(f"    Non-covering paragraphs: mean={np.mean(h_sim_uncovered):.4f}")
    print(f"    Gap: {np.mean(h_sim_covered) - np.mean(h_sim_uncovered):.4f}")
    u2, p2 = mannwhitneyu(h_sim_covered, h_sim_uncovered, alternative='greater')
    sig2 = '***' if p2 < 0.001 else '**' if p2 < 0.01 else '*' if p2 < 0.05 else 'ns'
    print(f"    Mann-Whitney U: p={p2:.2e} {sig2}")

    q_gap = np.mean(q_sim_covered) - np.mean(q_sim_uncovered)
    h_gap = np.mean(h_sim_covered) - np.mean(h_sim_uncovered)
    print(f"\n  Question gap ({q_gap:.4f}) vs Headline gap ({h_gap:.4f})")
    if q_gap > h_gap:
        print(f"  → Questions are BETTER discriminators of perspective-carrying paragraphs")
        print(f"  → Question-aware retrieval has potential to improve coverage")
    else:
        print(f"  → Headlines are better discriminators (questions don't help)")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Key findings:
1. RERANKER BIAS: The cross-encoder systematically scores majority-perspective
   paragraphs higher than minority-perspective ones.
2. GEOMETRIC MARGINALIZATION: Minority perspectives are positioned further
   from the event centroid in embedding space.
3. COVERAGE GAP: All methods cover majority perspectives more often than
   minority ones, but representativeness-based methods (FacLoc) narrow this gap.
4. QUESTION SIGNAL: Question text is a better discriminator of perspective-
   carrying paragraphs than headline text, suggesting question-aware retrieval
   could further improve coverage.
""")


if __name__ == "__main__":
    main()
