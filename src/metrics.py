"""Coverage@K, APD, PCC, IDR metric implementations."""


def coverage_at_k(retrieved_paragraphs, qa_pairs, k):
    """Compute Coverage@K: fraction of QA pairs answerable from top-k retrieved paragraphs."""
    raise NotImplementedError


def average_pairwise_distance(embeddings):
    """APD: average cosine distance between all pairs of document embeddings."""
    raise NotImplementedError


def proportional_corpus_coverage(embeddings, corpus_embeddings, n_clusters=50):
    """PCC: proportional representation across corpus topic clusters."""
    raise NotImplementedError


def intent_diversity_ratio(embeddings, n_intents=10):
    """IDR: ratio of unique intents covered in retrieved set."""
    raise NotImplementedError
