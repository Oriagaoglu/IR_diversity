"""
Compute inter-annotator agreement and validate LLM judge.

Usage: python scripts/compute_annotation_agreement.py

Expects completed CSVs in data/annotations/annotator_*.csv
with 'covered (yes/no)' column filled in.
"""

import csv
import json
import numpy as np
from pathlib import Path
from itertools import combinations

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANNOTATIONS_DIR = PROJECT_ROOT / "data" / "annotations"
DATA_DIR = PROJECT_ROOT / "data" / "processed"


def load_annotations():
    """Load all annotator JSON files. Returns {annotator: {pair_id: bool}}."""
    annotators = {}
    for json_path in sorted(ANNOTATIONS_DIR.glob("annotator_*.json")):
        name = json_path.stem.replace("annotator_", "")
        with open(json_path) as f:
            raw = json.load(f)
        labels = {int(k): v for k, v in raw.items()}
        if labels:
            annotators[name] = labels
            print(f"  {name}: {len(labels)} annotations "
                  f"({sum(labels.values())} yes, {len(labels) - sum(labels.values())} no)")
    return annotators


def cohens_kappa(labels1, labels2):
    """Cohen's kappa for two binary annotators on shared items."""
    shared = set(labels1.keys()) & set(labels2.keys())
    if len(shared) < 10:
        return None, len(shared)

    y1 = np.array([labels1[i] for i in sorted(shared)], dtype=int)
    y2 = np.array([labels2[i] for i in sorted(shared)], dtype=int)

    # Observed agreement
    po = np.mean(y1 == y2)

    # Expected agreement by chance
    p1_yes = np.mean(y1)
    p2_yes = np.mean(y2)
    pe = p1_yes * p2_yes + (1 - p1_yes) * (1 - p2_yes)

    if pe == 1.0:
        kappa = 1.0
    else:
        kappa = (po - pe) / (1 - pe)

    return kappa, len(shared)


def fleiss_kappa(annotators, pair_ids):
    """Fleiss' kappa for multiple annotators."""
    n_items = len(pair_ids)
    n_raters = len(annotators)
    n_categories = 2  # yes/no

    # Build count matrix: (n_items, n_categories)
    counts = np.zeros((n_items, n_categories))
    for i, pid in enumerate(sorted(pair_ids)):
        for ann_labels in annotators.values():
            if pid in ann_labels:
                if ann_labels[pid]:
                    counts[i, 1] += 1  # yes
                else:
                    counts[i, 0] += 1  # no

    # Number of raters per item (may vary if some skipped)
    n_per_item = counts.sum(axis=1)

    # Fleiss' kappa
    P_i = np.zeros(n_items)
    for i in range(n_items):
        n = n_per_item[i]
        if n < 2:
            P_i[i] = 0
        else:
            P_i[i] = (np.sum(counts[i] ** 2) - n) / (n * (n - 1))

    P_bar = np.mean(P_i)

    p_j = counts.sum(axis=0) / counts.sum()
    P_e = np.sum(p_j ** 2)

    if P_e == 1.0:
        kappa = 1.0
    else:
        kappa = (P_bar - P_e) / (1 - P_e)

    return kappa


def validate_against_llm(annotators):
    """Compare human majority vote against LLM labels."""
    # Load LLM ground truth
    gt_path = DATA_DIR / "annotation_llm_ground_truth.json"
    with open(gt_path) as f:
        llm_gt = json.load(f)

    # Load pair mapping to get claim_id|paragraph_id keys
    with open(DATA_DIR / "annotation_100pairs.json") as f:
        pairs_data = json.load(f)

    pair_keys = {}
    for p in pairs_data['pairs']:
        pair_keys[p['id']] = p['claim_id'] + '|' + p['paragraph_id']

    # Get all pair IDs annotated by at least 2 people
    all_ids = set()
    for labels in annotators.values():
        all_ids.update(labels.keys())

    results = {'agree': 0, 'disagree': 0, 'total': 0,
               'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}

    for pid in sorted(all_ids):
        # Majority vote
        votes = [labels[pid] for labels in annotators.values() if pid in labels]
        if len(votes) < 2:
            continue

        human_label = sum(votes) > len(votes) / 2  # majority says yes

        key = pair_keys.get(pid)
        if key is None or key not in llm_gt:
            continue

        llm_label = llm_gt[key]
        results['total'] += 1

        if human_label == llm_label:
            results['agree'] += 1
        else:
            results['disagree'] += 1

        if llm_label and human_label:
            results['tp'] += 1
        elif llm_label and not human_label:
            results['fp'] += 1
        elif not llm_label and human_label:
            results['fn'] += 1
        else:
            results['tn'] += 1

    return results


def interpret_kappa(k):
    if k is None:
        return "insufficient data"
    if k < 0:
        return "poor"
    if k < 0.20:
        return "slight"
    if k < 0.40:
        return "fair"
    if k < 0.60:
        return "moderate"
    if k < 0.80:
        return "substantial"
    return "almost perfect"


def main():
    print("=" * 60)
    print("ANNOTATION AGREEMENT & LLM VALIDATION")
    print("=" * 60)

    print("\n1. Loading annotations...")
    annotators = load_annotations()

    if len(annotators) < 2:
        print(f"\nOnly {len(annotators)} annotator(s) found. Need at least 2.")
        print("Fill in the CSVs in data/annotations/ and re-run.")
        return

    # Pairwise Cohen's kappa
    print(f"\n2. Pairwise Cohen's Kappa ({len(annotators)} annotators)")
    print("-" * 40)
    kappas = []
    for (n1, l1), (n2, l2) in combinations(annotators.items(), 2):
        k, n = cohens_kappa(l1, l2)
        if k is not None:
            kappas.append(k)
            print(f"  {n1} vs {n2}: κ = {k:.3f} ({interpret_kappa(k)}, n={n})")

    if kappas:
        print(f"\n  Mean pairwise κ = {np.mean(kappas):.3f}")

    # Fleiss' kappa
    all_ids = set()
    for labels in annotators.values():
        all_ids.update(labels.keys())
    shared_ids = set.intersection(*[set(l.keys()) for l in annotators.values()])

    if len(shared_ids) >= 10:
        fk = fleiss_kappa(annotators, shared_ids)
        print(f"\n3. Fleiss' Kappa (all annotators, {len(shared_ids)} shared items)")
        print("-" * 40)
        print(f"  κ = {fk:.3f} ({interpret_kappa(fk)})")

    # LLM validation
    print(f"\n4. LLM Judge Validation (vs human majority vote)")
    print("-" * 40)
    results = validate_against_llm(annotators)
    if results['total'] > 0:
        acc = results['agree'] / results['total']
        prec = results['tp'] / max(results['tp'] + results['fp'], 1)
        rec = results['tp'] / max(results['tp'] + results['fn'], 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)

        print(f"  Total pairs compared: {results['total']}")
        print(f"  Agreement (accuracy): {acc:.1%} ({results['agree']}/{results['total']})")
        print(f"  Precision (LLM yes → human yes): {prec:.1%}")
        print(f"  Recall (human yes → LLM yes): {rec:.1%}")
        print(f"  F1: {f1:.3f}")
        print(f"\n  Confusion matrix:")
        print(f"                    Human Yes  Human No")
        print(f"    LLM Yes           {results['tp']:>3}       {results['fp']:>3}")
        print(f"    LLM No            {results['fn']:>3}       {results['tn']:>3}")

        # Cohen's kappa between LLM and majority vote
        # Treat as two annotators
        llm_labels = {}
        human_labels = {}
        with open(DATA_DIR / "annotation_100pairs.json") as f:
            pairs_data = json.load(f)
        pair_keys = {}
        for p in pairs_data['pairs']:
            pair_keys[p['id']] = p['claim_id'] + '|' + p['paragraph_id']

        with open(DATA_DIR / "annotation_llm_ground_truth.json") as f:
            llm_gt = json.load(f)

        for pid in sorted(all_ids):
            votes = [labels[pid] for labels in annotators.values() if pid in labels]
            if len(votes) < 2:
                continue
            human_label = sum(votes) > len(votes) / 2
            key = pair_keys.get(pid)
            if key and key in llm_gt:
                human_labels[pid] = human_label
                llm_labels[pid] = llm_gt[key]

        k, n = cohens_kappa(llm_labels, human_labels)
        if k is not None:
            print(f"\n  Cohen's κ (LLM vs human majority): {k:.3f} ({interpret_kappa(k)})")

    print("\n" + "=" * 60)
    print("Done. Use these numbers in the paper's validation section.")


if __name__ == "__main__":
    main()
