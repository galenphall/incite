"""Graph analysis for citation-based paper discovery.

Provides Personalized PageRank and co-citation analysis over the local
CitationGraphStore subgraph around a user's library.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import numpy as np
from scipy import sparse

if TYPE_CHECKING:
    from incite.discovery.citation_graph import CitationGraphStore

logger = logging.getLogger(__name__)


def build_subgraph(
    graph: CitationGraphStore,
    seed_s2_ids: set[str],
    depth: int = 2,
) -> tuple[dict[int, str], dict[str, int], sparse.csr_matrix]:
    """Extract a depth-k subgraph around seed papers.

    Args:
        graph: The binary citation graph store.
        seed_s2_ids: S2 IDs of the user's library papers.
        depth: BFS expansion depth (default 2).

    Returns:
        Tuple of (int_to_s2, s2_to_int, adjacency_matrix) where adjacency_matrix
        is a sparse CSR matrix of the subgraph (row i cites column j).
    """
    # Map seed S2 IDs to internal integer IDs
    seed_ints: set[int] = set()
    for s2_id in seed_s2_ids:
        int_id = graph._lookup_id(s2_id)
        if int_id is not None:
            seed_ints.add(int_id)

    if not seed_ints:
        logger.warning("No seed papers found in citation graph")
        return {}, {}, sparse.csr_matrix((0, 0))

    # BFS expansion
    visited: set[int] = set(seed_ints)
    frontier = set(seed_ints)

    for d in range(depth):
        next_frontier: set[int] = set()
        for node in frontier:
            for neighbor in graph.get_references_int(node):
                if neighbor not in visited:
                    next_frontier.add(neighbor)
            for neighbor in graph.get_citations_int(node):
                if neighbor not in visited:
                    next_frontier.add(neighbor)
        visited.update(next_frontier)
        frontier = next_frontier
        logger.debug(
            "Subgraph depth %d: %d new nodes, %d total", d + 1, len(next_frontier), len(visited)
        )

    # Build local index mapping
    sorted_nodes = sorted(visited)
    local_to_global = {i: g for i, g in enumerate(sorted_nodes)}
    global_to_local = {g: i for i, g in enumerate(sorted_nodes)}
    n = len(sorted_nodes)

    # Build sparse adjacency (row cites col)
    rows: list[int] = []
    cols: list[int] = []
    for local_i, global_i in local_to_global.items():
        for ref_global in graph.get_references_int(global_i):
            if ref_global in global_to_local:
                rows.append(local_i)
                cols.append(global_to_local[ref_global])

    data = np.ones(len(rows), dtype=np.float32)
    adj = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))

    # Build S2 ID mappings
    int_to_s2: dict[int, str] = {}
    s2_to_int: dict[str, int] = {}
    for local_i, global_i in local_to_global.items():
        s2_id = graph._lookup_s2_id(global_i)
        int_to_s2[local_i] = s2_id
        s2_to_int[s2_id] = local_i

    logger.info("Subgraph: %d nodes, %d edges (depth=%d)", n, len(rows), depth)
    return int_to_s2, s2_to_int, adj


def compute_pagerank(
    adj: sparse.csr_matrix,
    seed_indices: set[int],
    int_to_s2: dict[int, str],
    damping: float = 0.85,
    iterations: int = 50,
    seed_bias: float = 0.5,
    tol: float = 1e-6,
) -> dict[str, float]:
    """Personalized PageRank over a subgraph.

    Teleportation is biased toward seed (library) papers so that results
    favor the user's subfield.

    Args:
        adj: Sparse adjacency matrix (row cites col).
        seed_indices: Local indices of seed/library papers.
        int_to_s2: Mapping from local index to S2 ID.
        damping: Damping factor (probability of following a link).
        iterations: Maximum power iterations.
        seed_bias: Fraction of teleportation mass allocated to seed papers.
        tol: Convergence tolerance.

    Returns:
        Dict mapping S2 ID to normalized PageRank score in [0, 1].
    """
    n = adj.shape[0]
    if n == 0:
        return {}

    # Build transition matrix from citation links (col-normalize)
    # We want "importance flows from citer to cited", so use adj transposed
    # adj[i,j] = 1 means i cites j, so column j receives importance from row i
    # Column-normalize adj: each column j gets equal share from its citers
    col_sums = np.array(adj.sum(axis=0)).flatten()
    col_sums[col_sums == 0] = 1.0  # avoid division by zero
    inv_col = sparse.diags(1.0 / col_sums)
    transition = adj @ inv_col  # column-stochastic-ish transition

    # Personalization vector: biased toward seeds
    personalization = np.ones(n, dtype=np.float64)
    if seed_indices:
        non_seed_mass = (1.0 - seed_bias) / max(n - len(seed_indices), 1)
        seed_mass = seed_bias / len(seed_indices)
        for i in range(n):
            personalization[i] = seed_mass if i in seed_indices else non_seed_mass
    personalization /= personalization.sum()

    # Power iteration
    rank = np.ones(n, dtype=np.float64) / n
    for it in range(iterations):
        new_rank = damping * transition.T @ rank + (1 - damping) * personalization
        new_rank /= new_rank.sum()
        diff = np.abs(new_rank - rank).sum()
        rank = new_rank
        if diff < tol:
            logger.debug("PageRank converged at iteration %d (diff=%.2e)", it + 1, diff)
            break

    # Normalize to [0, 1]
    max_rank = rank.max()
    if max_rank > 0:
        rank = rank / max_rank

    return {int_to_s2[i]: float(rank[i]) for i in range(n)}


def compute_cocitation(
    graph: CitationGraphStore,
    seed_s2_ids: set[str],
    max_candidates: int = 500,
) -> dict[str, float]:
    """Co-citation analysis with Adamic/Adar weighting.

    For each seed paper's citers, collect what else those citers reference.
    Weight by Adamic/Adar: 1 / log(1 + |refs(citer)|) to penalize review
    articles that cite everything.

    Args:
        graph: The binary citation graph store.
        seed_s2_ids: S2 IDs of the user's library papers.
        max_candidates: Maximum candidates to return.

    Returns:
        Dict mapping S2 ID to normalized co-citation score in [0, 1].
    """
    scores: dict[str, float] = {}

    for seed_id in seed_s2_ids:
        citers = graph.get_citations(seed_id)
        if not citers:
            continue

        for citer_id in citers:
            # Adamic/Adar weight: penalize citers with many references
            citer_refs = graph.get_references(citer_id)
            n_refs = len(citer_refs)
            if n_refs == 0:
                continue
            weight = 1.0 / math.log(1 + n_refs)

            for ref_id in citer_refs:
                if ref_id in seed_s2_ids:
                    continue  # skip library papers themselves
                scores[ref_id] = scores.get(ref_id, 0.0) + weight

    if not scores:
        return {}

    # Keep top candidates
    top_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:max_candidates]

    # Normalize to [0, 1]
    max_score = top_items[0][1] if top_items else 1.0
    return {s2_id: score / max_score for s2_id, score in top_items}
