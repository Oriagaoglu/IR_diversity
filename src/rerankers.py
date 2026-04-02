"""BM25, MMR, NEWSCOPE, and coverage-aware reranking methods."""


def bm25_baseline(index, query, k=10):
    """Standard BM25 retrieval."""
    raise NotImplementedError


def mmr_rerank(query_embedding, doc_embeddings, relevance_scores, k=10, lambda_=0.5):
    """Maximal Marginal Relevance reranking."""
    raise NotImplementedError


def newscope_rerank(query_embedding, doc_embeddings, relevance_scores, k=10):
    """NEWSCOPE diversity-aware reranking."""
    raise NotImplementedError


def coverage_aware_rerank(query_embedding, doc_embeddings, qa_pairs, relevance_scores, k=10):
    """Our coverage-aware reranking method."""
    raise NotImplementedError
