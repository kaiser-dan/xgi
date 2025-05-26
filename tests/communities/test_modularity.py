import numpy as np
import pytest

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

        q = xgi.communities.modularity._strict_modularity(self.H, A)
        expected = 0.0

        assert np.isclose(q, expected)

    def test_max_groups(self):
        A = [[node] for node in self.H.nodes]

        q = xgi.communities.modularity._strict_modularity(self.H, A)
        expected = 0.0

        assert q < 0

        # TODO: Compute modularity of discrete partition directly
        # assert np.isclose(q, expected)

    @pytest.mark.skip
    def test_component_partition(self):
        pass
