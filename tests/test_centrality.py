"""Bonacich centrality: analytic cases, solver agreement and edge cases."""

from __future__ import annotations

import numpy as np
import pytest

from bscores.centrality import (
    _has_cycle,
    bonacich_centrality,
    in_strength,
    neumann_centrality,
    out_strength,
    principal_eigenvector,
)


def cycle(n: int) -> np.ndarray:
    """Loss matrix for a perfect n-cycle: 0 beat 1 beat 2 ... beat 0."""
    w = np.zeros((n, n))
    for i in range(n):
        w[(i + 1) % n, i] = 1.0
    return w


class TestAnalyticCases:
    def test_symmetric_cycle_is_flat(self):
        # Everyone beat exactly one opponent and lost to exactly one, so the
        # network cannot distinguish them.
        for n in (2, 3, 5, 8):
            scores = bonacich_centrality(cycle(n))
            np.testing.assert_allclose(scores, np.full(n, 1.0 / np.sqrt(n)), atol=1e-9)

    def test_unit_euclidean_norm(self):
        rng = np.random.default_rng(0)
        w = rng.random((12, 12))
        np.fill_diagonal(w, 0.0)
        scores = bonacich_centrality(w)
        assert np.linalg.norm(scores) == pytest.approx(1.0)

    def test_scores_are_non_negative(self):
        rng = np.random.default_rng(1)
        w = rng.random((20, 20)) * (rng.random((20, 20)) < 0.3)
        assert np.all(bonacich_centrality(w) >= 0.0)

    def test_beating_a_strong_opponent_beats_beating_a_weak_one(self):
        # A and B each won once.  A beat C, who has beaten everyone else;
        # B beat D, who has never won.  A must outrank B.
        names = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5}
        w = np.zeros((6, 6))
        w[names["C"], names["A"]] = 1.0  # C lost to A
        w[names["D"], names["B"]] = 1.0  # D lost to B
        for loser in ("E", "F"):
            w[names[loser], names["C"]] = 1.0  # C beat E and F
        scores = bonacich_centrality(w, regularization=1e-6)
        assert scores[names["A"]] > scores[names["B"]]

    def test_dominant_node_of_a_star(self):
        # One node beats everybody; it must be the most central.
        n = 7
        w = np.zeros((n, n))
        w[1:, 0] = 1.0
        scores = bonacich_centrality(w, regularization=1e-8)
        assert np.argmax(scores) == 0

    def test_eigenvector_equation_holds(self):
        rng = np.random.default_rng(2)
        w = rng.random((9, 9))
        result = bonacich_centrality(w, return_info=True)
        # x = (1 / rho) W' x
        np.testing.assert_allclose(
            w.T @ result.vector, result.eigenvalue * result.vector, atol=1e-8
        )

    def test_eigenvalue_is_the_spectral_radius(self):
        rng = np.random.default_rng(3)
        w = rng.random((10, 10))
        result = bonacich_centrality(w, return_info=True)
        assert result.eigenvalue == pytest.approx(np.max(np.abs(np.linalg.eigvals(w))), rel=1e-8)


class TestSolverAgreement:
    @pytest.mark.parametrize("seed", range(6))
    def test_power_matches_dense(self, seed):
        rng = np.random.default_rng(seed)
        n = 15
        w = rng.random((n, n)) * (rng.random((n, n)) < 0.5)
        np.fill_diagonal(w, 0.0)
        power = bonacich_centrality(w, method="power", tol=1e-14)
        dense = bonacich_centrality(w, method="dense")
        np.testing.assert_allclose(power, dense, atol=1e-7)

    def test_shift_does_not_move_an_irreducible_answer(self):
        rng = np.random.default_rng(7)
        w = rng.random((11, 11)) + 0.01
        unshifted = bonacich_centrality(w, shift=0.0, tol=1e-14)
        shifted = bonacich_centrality(w, shift=0.5, tol=1e-14)
        np.testing.assert_allclose(unshifted, shifted, atol=1e-8)

    def test_shift_rescues_a_periodic_network(self):
        # Two teams that only ever beat each other, alternately: the unshifted
        # iteration oscillates with period 2.
        w = np.array([[0.0, 1.0], [1.0, 0.0]])
        result = principal_eigenvector(w.T, shift=0.15, tol=1e-13, method="power")
        assert result.converged
        np.testing.assert_allclose(result.vector, [1 / np.sqrt(2)] * 2, atol=1e-9)

    def test_transposed_input_is_equivalent(self):
        rng = np.random.default_rng(8)
        w = rng.random((10, 10))
        np.testing.assert_allclose(
            bonacich_centrality(w), bonacich_centrality(w.T, transposed=True), atol=1e-10
        )

    def test_warm_start_reaches_the_same_answer_sooner(self):
        rng = np.random.default_rng(9)
        w = rng.random((40, 40)) * (rng.random((40, 40)) < 0.2) + 1e-3
        cold = bonacich_centrality(w, return_info=True, tol=1e-13)
        nudged = w + 1e-6 * rng.random((40, 40))
        warm = bonacich_centrality(nudged, x0=cold.vector, return_info=True, tol=1e-13)
        fresh = bonacich_centrality(nudged, return_info=True, tol=1e-13)
        np.testing.assert_allclose(warm.vector, fresh.vector, atol=1e-8)
        assert warm.iterations < fresh.iterations

    def test_regularization_matches_an_explicit_dense_perturbation(self):
        rng = np.random.default_rng(10)
        n = 8
        w = rng.random((n, n)) * (rng.random((n, n)) < 0.4)
        np.fill_diagonal(w, 0.0)
        eps = 1e-3
        implicit = bonacich_centrality(w, regularization=eps, tol=1e-14)
        explicit = bonacich_centrality(w + eps * (np.ones((n, n)) - np.eye(n)), tol=1e-14)
        np.testing.assert_allclose(implicit, explicit, atol=1e-8)

    def test_regularization_grades_an_acyclic_chain(self):
        # A beat B beat C.  Without regularization the network has no cycle, so
        # the spectral radius is zero and only the shift breaks the tie.
        w = np.zeros((3, 3))
        w[1, 0] = 1.0  # B lost to A
        w[2, 1] = 1.0  # C lost to B
        scores = bonacich_centrality(w, regularization=1e-3)
        assert scores[0] > scores[1] > scores[2] > 0.0


class TestSparse:
    def test_sparse_matches_dense(self):
        sp = pytest.importorskip("scipy.sparse")
        rng = np.random.default_rng(11)
        n = 30
        w = rng.random((n, n)) * (rng.random((n, n)) < 0.15)
        np.testing.assert_allclose(
            bonacich_centrality(sp.csr_matrix(w), tol=1e-14),
            bonacich_centrality(w, tol=1e-14),
            atol=1e-8,
        )


class TestEdgeCases:
    def test_empty_matrix(self):
        result = principal_eigenvector(np.zeros((0, 0)), method="auto")
        assert result.vector.shape == (0,)
        assert result.converged

    def test_all_zero_network_scores_zero(self):
        scores = bonacich_centrality(np.zeros((5, 5)))
        np.testing.assert_allclose(scores, 0.0)

    def test_dense_solver_on_a_zero_network(self):
        scores = bonacich_centrality(np.zeros((4, 4)), method="dense")
        np.testing.assert_allclose(scores, 0.0)

    def test_isolated_nodes_score_zero(self):
        w = np.zeros((4, 4))
        w[1, 0] = 1.0  # only nodes 0 and 1 have played
        scores = bonacich_centrality(w, shift=0.15)
        assert scores[2] == 0.0 and scores[3] == 0.0

    def test_non_square_rejected(self):
        with pytest.raises(ValueError, match="square"):
            principal_eigenvector(np.zeros((3, 4)))

    def test_bad_x0_rejected(self):
        with pytest.raises(ValueError, match="x0"):
            principal_eigenvector(np.eye(3), x0=np.ones(2))

    def test_negative_knobs_rejected(self):
        with pytest.raises(ValueError, match="shift"):
            principal_eigenvector(np.eye(2), shift=-1.0)
        with pytest.raises(ValueError, match="regularization"):
            principal_eigenvector(np.eye(2), regularization=-1.0)

    def test_unknown_method_rejected(self):
        with pytest.raises(ValueError, match="unknown method"):
            principal_eigenvector(np.eye(2), method="magic")

    def test_zero_x0_falls_back_to_uniform(self):
        w = cycle(4)
        scores = bonacich_centrality(w, x0=np.zeros(4))
        np.testing.assert_allclose(scores, 0.5, atol=1e-9)

    def test_power_method_reports_non_convergence(self):
        rng = np.random.default_rng(12)
        w = rng.random((10, 10))
        result = principal_eigenvector(w.T, max_iter=1, method="power", tol=1e-15)
        assert not result.converged


class TestCycleDetection:
    def test_dag_has_no_cycle(self):
        w = np.zeros((4, 4))
        w[1, 0] = w[2, 1] = w[3, 2] = 1.0
        assert not _has_cycle(w)

    def test_cycle_is_detected(self):
        assert _has_cycle(cycle(3))

    def test_self_loop_is_a_cycle(self):
        w = np.zeros((2, 2))
        w[0, 0] = 1.0
        assert _has_cycle(w)

    def test_isolated_nodes_do_not_create_cycles(self):
        assert not _has_cycle(np.zeros((5, 5)))

    def test_empty(self):
        assert not _has_cycle(np.zeros((0, 0)))

    def test_regularization_always_connects(self):
        w = np.zeros((3, 3))
        assert _has_cycle(w, regularization=1e-6)
        assert not _has_cycle(np.zeros((1, 1)), regularization=1e-6)

    def test_a_cycle_plus_a_dag_tail(self):
        w = cycle(3)
        big = np.zeros((5, 5))
        big[:3, :3] = w
        big[3, 0] = 1.0  # node 3 lost to node 0, and never won
        big[4, 3] = 1.0
        assert _has_cycle(big)

    def test_sparse_pattern(self):
        sp = pytest.importorskip("scipy.sparse")
        assert _has_cycle(sp.csr_matrix(cycle(4)))
        dag = np.zeros((3, 3))
        dag[1, 0] = dag[2, 1] = 1.0
        assert not _has_cycle(sp.csr_matrix(dag))


class TestAcyclicFallback:
    def test_a_perfect_hierarchy_is_graded_not_collapsed(self):
        # A beat B beat C beat D, no upsets: the spectral radius is 0 and the
        # eigenvector equation has nothing to say, so the Neumann series ranks.
        n = 4
        w = np.zeros((n, n))
        for i in range(1, n):
            w[i, i - 1] = 1.0
        result = bonacich_centrality(w, return_info=True)
        assert result.method == "acyclic"
        assert result.eigenvalue == 0.0
        scores = result.vector
        assert scores[0] > scores[1] > scores[2] > 0.0
        assert scores[3] == 0.0  # D never won anything

    def test_never_winning_scores_zero(self):
        w = np.zeros((3, 3))
        w[1, 0] = w[2, 0] = 1.0  # 0 beat both 1 and 2
        scores = bonacich_centrality(w)
        assert scores[0] > 0.0
        assert scores[1] == 0.0 and scores[2] == 0.0

    def test_single_result_matches_the_obvious_answer(self):
        w = np.zeros((2, 2))
        w[1, 0] = 1.0
        np.testing.assert_allclose(bonacich_centrality(w), [1.0, 0.0])

    def test_unit_norm_is_preserved(self):
        w = np.zeros((5, 5))
        for i in range(1, 5):
            w[i, i - 1] = 1.0
        assert np.linalg.norm(bonacich_centrality(w)) == pytest.approx(1.0)

    def test_is_fast_on_a_deep_hierarchy(self):
        # The pre-check must catch this; iterating a Jordan block to 1e-12
        # would take on the order of 1e12 steps.
        n = 200
        w = np.zeros((n, n))
        for i in range(1, n):
            w[i, i - 1] = 1.0
        result = bonacich_centrality(w, return_info=True)
        assert result.method == "acyclic"
        assert result.iterations <= n + 2

    def test_all_solvers_agree_on_the_degenerate_case(self):
        w = np.zeros((4, 4))
        for i in range(1, 4):
            w[i, i - 1] = 1.0
        for method in ("auto", "power", "dense"):
            result = bonacich_centrality(w, method=method, return_info=True)
            assert result.method == "acyclic"

    def test_scale_invariant_like_the_eigenvector_solve(self):
        # Doubling every arc weight must not move a rating.
        w = np.zeros((5, 5))
        for i in range(1, 5):
            w[i, i - 1] = float(i)
        np.testing.assert_allclose(
            bonacich_centrality(w), bonacich_centrality(7.5 * w), atol=1e-12
        )

    def test_damping_controls_how_far_credit_travels(self):
        w = np.zeros((3, 3))
        w[1, 0] = w[2, 1] = 1.0
        shallow = neumann_centrality(w, damping=0.1)
        deep = neumann_centrality(w, damping=0.9)
        # More damping credit means beating a winner counts for more, so the
        # top of the chain pulls further ahead of the middle.
        assert deep[0] / deep[1] > shallow[0] / shallow[1]

    def test_regularization_switches_back_to_the_eigenvector(self):
        w = np.zeros((4, 4))
        for i in range(1, 4):
            w[i, i - 1] = 1.0
        result = bonacich_centrality(w, regularization=1e-3, return_info=True)
        assert result.method == "power"
        assert np.all(result.vector > 0.0)

    def test_public_helper_rejects_cyclic_networks(self):
        np.testing.assert_allclose(
            neumann_centrality(np.array([[0.0, 0.0], [1.0, 0.0]])), [1.0, 0.0]
        )
        with pytest.raises(ValueError, match="directed cycle"):
            neumann_centrality(cycle(3))

    def test_public_helper_accepts_a_transposed_operator(self):
        w = np.zeros((3, 3))
        w[1, 0] = w[2, 1] = 1.0
        np.testing.assert_allclose(
            neumann_centrality(w), neumann_centrality(w.T, transposed=True)
        )


class TestStrength:
    def test_in_and_out_strength(self):
        w = np.array([[0.0, 2.0], [3.0, 0.0]])
        np.testing.assert_allclose(in_strength(w), [3.0, 2.0])
        np.testing.assert_allclose(out_strength(w), [2.0, 3.0])

    def test_in_strength_counts_wins(self):
        # Node 0 beat 1 and 2; its in-strength is 2.
        w = np.zeros((3, 3))
        w[1, 0] = w[2, 0] = 1.0
        np.testing.assert_allclose(in_strength(w), [2.0, 0.0, 0.0])
