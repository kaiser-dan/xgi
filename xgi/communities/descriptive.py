"""Descriptive methods to cluster a hypergraph."""

import random
import numpy as np

import xgi
from ..exception import XGIError

__all__ = ["greedy_random_partition", "simple_cnm"]

from .modularity import compute_modularity


def greedy_random_partition(H, max_steps=1000):
    # Initialize modularity
    q_opt = -1

    # Iterative search TODO: Improve comment
    for _ in range(max_steps):
        A_best = [{node} for node in H.nodes]
        q_best = xgi.compute_modularity(H, A_best, mtype="strict")
        E_prime = set()

        permuted_edges = list(H.edges)
        random.shuffle(permuted_edges)

        for edge in permuted_edges:
            H_prime = xgi.subhypergraph(H, edges=E_prime | set([edge]))
            A_candidate = list(xgi.connected_components(H_prime))
            q_candidate = compute_modularity(H, A_candidate)

            if q_candidate > q_best:
                q_best = q_candidate
                A_best = A_candidate
                E_prime = E_prime | set([edge])

        if q_best > q_opt:
            q_opt = q_best
            A_opt = A_best

    return A_opt, q_opt


def simple_cnm(H):
    # Initialize communities with trivial memberships
    A_opt = [[node] for node in H.nodes]
    q_opt = xgi.compute_modularity(H, A_opt, mtype="strict")
    print(f"Initial q: {q_opt}")

    A_prime = A_opt

    _iters = 0
    E_0 = set()
    while E_0 != set(H.edges):
        print("Starting iteration: ", _iters)
        q_prime = -1
        for edge in set(H.edges) - E_0:
            E_star = E_0 | {edge}
            H_prime = xgi.subhypergraph(H, edges=E_star)

            A_candidate = list(xgi.connected_components(H_prime))
            q_candidate = compute_modularity(H, A_candidate)

            if q_candidate > q_prime:
                print("Candidate improves modularity! Updating...")
                print(f"Changing Q from {q_prime} to {q_candidate}")
                q_prime = q_candidate
                print(f"Changing A from {A_prime} to {A_candidate}")
                A_prime = A_candidate
                E_prime = E_star

        E_0 = E_prime
        print(f"E_0 is now {E_0}")

        print("q_prime, q_opt", q_prime, q_opt)
        if q_prime >= q_opt:
            print(f"Updating opts with primes...")
            q_opt = q_prime
            A_opt = A_prime

        print(f"Current optimal q: {q_opt}")
        print(f"Current optimal A: {A_opt}")

    return A_opt, q_opt
