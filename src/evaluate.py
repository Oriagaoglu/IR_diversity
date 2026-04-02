"""Full evaluation pipeline: run retrieval methods and compute all metrics."""


def run_evaluation(events, methods, ks=[5, 10, 20]):
    """Run all retrieval methods on all events and compute metrics at each K."""
    raise NotImplementedError
