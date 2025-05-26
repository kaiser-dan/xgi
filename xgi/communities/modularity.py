"""Modularity of a hypergraph partition against a generalized Chung-Lu model."""

import numpy as np

import xgi
from ..exception import XGIError

__all__ = [
    "compute_modularity",
]


def compute_modularity(H, A, mtype="strict"):
    """
    Parameters
    ----------
    H : Hypergraph
        Hypergraph.
    A : array
        Array-like of collections of node indices.
    mtype : str, optional
        Modularity function to compute, default 'strict'.

    Returns
    -------
    float
        The modularity of the partition A on H.

    Raises
    ------
    ValueError
        'mtype' is not supported value ('strict', 'soft', 'majority').
    NotImplementedError
        'mtype' is not currently supported value ('strict').
    """
    if mtype not in ("strict", "soft", "majority"):
        raise ValueError("'mtype' must be 'strict', 'soft', or 'majority'!")

    if mtype != "strict":
        raise NotImplementedError("Only 'strict' modularity is supported right now.")

    pass


def _strict_modularity(H, A):
    """Computes the strict modularity of the vertex partition, A, on the hypergraph, H,
    as defined in [1].

    Parameters
    ----------
    H : Hypergraph
        Hypergraph.
    A : array
        Array-like of collections of node indices.

    Returns
    -------
    float
        The strict modularity of the partition A on H.

    Raises
    ------
    XGIError
        AHHH?


    References
    ----------
    TODO: Add accent mark
    .. [1] Kaminski et al. (2018).
        Clustering via Hypergraph Modularity.
        PLOS One.

    """

    internal_contributions = 0
    for partition in A:
        internal_contributions += xgi.subhypergraph(H, nodes=partition).num_edges

    null_expectations = 0
    for edge_size in range(2, max(H.edges.size.aslist()) + 1):
        H_order = xgi.subhypergraph(H, edges=H.edges.filterby("order", edge_size - 1))

        tmp = 0
        for partition in A:
            tmp += np.power(vol(H, partition) / vol(H, H.nodes), edge_size)
        null_expectations += H_order.num_edges * tmp

    return (1 / H.num_edges) * (internal_contributions - null_expectations)


def vol(H, V):
    return sum(map(lambda node: H.nodes.degree[node], V))
