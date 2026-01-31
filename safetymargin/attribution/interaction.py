"""Pairwise interaction gain computation for safety margins."""

from typing import Callable, Dict, Iterable, List, Tuple


def compute_interaction_gain(
    scorer_fn: Callable[[str], float],
    minimal_context: List[str],
    spans: List[str],
    question: str,
    pairs: Iterable[Tuple[int, int]],
    assemble_fn: Callable,
) -> Dict[Tuple[int, int], float]:
    """
    Compute interaction gain Δ(s_i, s_j) for span pairs.
    
    Δ(i,j) = f(M∪{s_i,s_j}) - f(M∪{s_i}) - f(M∪{s_j}) + f(M)
    
    Positive values indicate synergy (safety boost).
    Negative values indicate interference (amplifies harm).
    
    Args:
        scorer_fn: Function that takes context string and returns safety score
        minimal_context: Minimal baseline context (M)
        spans: List of specification spans
        question: The user question/prompt
        pairs: Iterable of (i, j) index tuples to compute interactions for
        assemble_fn: Function to assemble context from components
        
    Returns:
        Dictionary mapping (i, j) tuples to interaction gain values
    """
    pairs = list(pairs)
    if not pairs:
        return {}
    
    cache: Dict[str, float] = {}
    
    def f_ctx(ctx: str) -> float:
        """Cached scoring function."""
        if ctx not in cache:
            cache[ctx] = scorer_fn(ctx)
        return cache[ctx]
    
    # Compute baseline with minimal context only
    base_M = f_ctx(assemble_fn(minimal_context, [], question))
    pair_scores: Dict[Tuple[int, int], float] = {}
    
    for i, j in pairs:
        # Get contexts with individual spans
        ctx_i = assemble_fn(minimal_context, [spans[i]], question)
        ctx_j = assemble_fn(minimal_context, [spans[j]], question)
        # Get context with both spans
        ctx_ij = assemble_fn(minimal_context, [spans[i], spans[j]], question)
        
        # Compute interaction gain
        val = f_ctx(ctx_ij) - f_ctx(ctx_i) - f_ctx(ctx_j) + base_M
        pair_scores[(i, j)] = val
    
    return pair_scores
