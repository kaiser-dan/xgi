import numpy as np
import pytest

import random
import networkx as nx

import xgi
from xgi.exception import XGIError


class TestModularityUtil:
    G = xgi.fast_random_hypergraph(100, [0.01])
    H = xgi.fast_random_hypergraph(100, [0.01, 0.01])

    def test_vol_all_dyadic(self):
        assert (
            xgi.communities.modularity.vol(self.G, self.G.nodes) == 2 * self.G.num_edges
        )

    def test_vol_single_dyadic(self):
        assert xgi.communities.modularity.vol(self.G, [0]) == self.G.nodes.degree[0]

    def test_vol_all(self):
        vol_2edges = (
            2
            * xgi.subhypergraph(
                self.H, edges=self.H.edges.filterby("order", 1)
            ).num_edges
        )
        vol_3edges = (
            3
            * xgi.subhypergraph(
                self.H, edges=self.H.edges.filterby("order", 2)
            ).num_edges
        )

        assert (
            xgi.communities.modularity.vol(self.H, self.H.nodes)
            == vol_2edges + vol_3edges
        )


class TestStrictModularity:
    G = xgi.fast_random_hypergraph(100, [0.01])
    H = xgi.fast_random_hypergraph(100, [0.01, 0.01])

    def test_one_group_dyadic(self):
        A = [tuple(self.G.nodes)]

        q = xgi.communities.modularity._strict_modularity(self.G, A)
        expected = 0.0

        assert np.isclose(q, expected)

    def test_max_groups_dyadic(self):
        A = [[node] for node in self.G.nodes]

        q = xgi.communities.modularity._strict_modularity(self.G, A)
        expected = -sum(
            map(lambda node: self.G.nodes.degree[node] ** 2, self.G.nodes)
        ) / (4 * self.G.num_edges**2)

        assert q < 0.0
        assert np.isclose(q, expected)

    def test_one_group(self):
        A = [tuple(self.H.nodes)]

        expected = 0.0
        actual = xgi.communities.modularity._strict_modularity(self.H, A)

        assert np.isclose(actual, expected)

    def test_max_groups(self):
        # TODO: Compute modularity of discrete partition directly
        A = [[node] for node in self.H.nodes]

        actual = xgi.communities.modularity._strict_modularity(self.H, A)

        assert actual < 0

    @pytest.mark.parametrize("num_groups", range(2, 11))
    def test_random_partition_dyadic(self, num_groups):
        _nodes = list(self.G.nodes)
        random.shuffle(_nodes)

        A = [[] for _ in range(num_groups)]
        for node in _nodes:
            A[node % num_groups].append(node)

        G_dyadic = xgi.to_graph(self.G)
        expected = nx.community.modularity(G_dyadic, A)

        actual = xgi.communities.modularity._strict_modularity(self.G, A)

        assert np.isclose(actual, expected)

    def test_fix_descriptive(self):
        H = xgi.Hypergraph(
            [
                (0, 1),
                (0, 2),
                (0, 3),
                (1, 2),
                (1, 3),
                (2, 3),
                (4, 5),
                (4, 6),
                (4, 7),
                (5, 6),
                (5, 7),
                (6, 7),
            ]
        )

        A = [(0, 1), (2, 3), (4, 5), (6, 7)]

        G = xgi.to_graph(H)
        expected = nx.community.modularity(G, A)
        actual = xgi.communities.modularity._strict_modularity(H, A)

        assert np.isclose(actual, expected)

    def test_component_partition(self):
        H = xgi.Hypergraph(
            [
                (0, 1),
                (0, 2),
                (0, 3),
                (1, 2),
                (1, 3),
                (2, 3),
                (4, 5),
                (4, 6),
                (4, 7),
                (5, 6),
                (5, 7),
                (6, 7),
            ]
        )

        A = [(0, 1, 2, 3), (4, 5, 6, 7)]

        expected = 1 / 2
        actual = xgi.communities.modularity._strict_modularity(H, A)

        assert np.isclose(actual, expected)
