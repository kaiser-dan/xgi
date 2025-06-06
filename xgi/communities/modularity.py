"""Modularity of a hypergraph partition against a generalized Chung-Lu model."""

import numpy as np

import xgi
from ..exception import XGIError

__all__ = ["compute_modularity", "vol"]


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

    if mtype == "strict":
        return _strict_modularity(H, A)


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

    # Compute edge contributions, e.g., edges within community
    edge_contributions = 0
    for partition in A:
        edge_contributions += xgi.subhypergraph(H, nodes=partition).num_edges

    # Compute degree tax, e.g., expectation in a Chung-Lu model
    degree_tax = 0
    for edge_size in range(2, max(H.edges.size.aslist()) + 1):
        H_order = xgi.subhypergraph(H, edges=H.edges.filterby("order", edge_size - 1))

        total_volume = 0
        for partition in A:
            total_volume += np.power(vol(H, partition) / vol(H, H.nodes), edge_size)
        degree_tax += H_order.num_edges * total_volume

    return (1 / H.num_edges) * (edge_contributions - degree_tax)


def vol(H, V):
    """Volume"""
    return sum(map(lambda node: H.nodes.degree[node], V))
