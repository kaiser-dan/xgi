from itertools import product
import numpy as np
import pytest

import xgi
from xgi.exception import XGIError


class TestGreedyRandomPartition:
    def test_disconnected_cliques(self):
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
        print(H.num_edges)

        expected_group_A = [0, 1, 2, 3]
        expected_group_B = [4, 5, 6, 7]

        actual_A, actual_q = xgi.greedy_random_partition(H, max_steps=100)

        print(actual_A)

        # (Possibly) permute labels to ensure compatibility
        assert (
            sorted(actual_A[0]) == expected_group_A
            or sorted(actual_A[0]) == expected_group_B
        )


class TestSimpleCNM:
    def test_disconnected_cliques(self):
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
        print(H.num_edges)

        expected_group_A = [0, 1, 2, 3]
        expected_group_B = [4, 5, 6, 7]

        actual_A, actual_q = xgi.simple_cnm(H)

        # (Possibly) permute labels to ensure compatibility
        assert (
            sorted(actual_A[0]) == expected_group_A
            or sorted(actual_A[0]) == expected_group_B
        )

    def test_strongly_separable(self):
        p = np.array([[0.3, 0.01], [0.01, 0.3]])
        H = xgi.uniform_HSBM(100, 2, p, [50, 50])

        clusters = xgi.simple_cnm(H)

        assert len(clusters) == 2
