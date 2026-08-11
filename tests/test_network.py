"""The loss network: Eq. 1 assembly, causality and the streaming fast path."""

from __future__ import annotations

import numpy as np
import pytest

from bscores.decay import Exponential, Hyperbolic, Uniform
from bscores.network import LossNetwork, NodeIndex


class TestNodeIndex:
    def test_ids_are_assigned_in_first_seen_order(self):
        index = NodeIndex()
        assert index.add("b") == 0
        assert index.add("a") == 1
        assert index.add("b") == 0
        assert index.names == ["b", "a"]

    def test_construct_from_names(self):
        index = NodeIndex(["x", "y", "x"])
        assert len(index) == 2
        assert index[1] == "y"
        assert list(index) == ["x", "y"]

    def test_membership_and_lookup(self):
        index = NodeIndex(["x"])
        assert "x" in index and "z" not in index
        assert index.get("x") == 0
        with pytest.raises(KeyError, match="unknown competitor"):
            index.get("z")

    def test_add_many(self):
        index = NodeIndex()
        np.testing.assert_array_equal(index.add_many(["a", "b", "a"]), [0, 1, 0])

    def test_repr(self):
        assert "2 nodes" in repr(NodeIndex(["a", "b"]))


def reference_matrix(network: LossNetwork, at: float, *, inclusive: bool = True) -> np.ndarray:
    """Eq. 1 written out literally, as a check on the vectorised assembly."""
    events = network.events()
    w = np.zeros((network.n_nodes, network.n_nodes))
    for loser, winner, time, weight in zip(
        events["loser"], events["winner"], events["time"], events["weight"]
    ):
        if time > at or (time == at and not inclusive):
            continue
        if network.max_age is not None and at - time > network.max_age:
            continue
        w[loser, winner] += weight * network.kernel(at - time)
    return w


class TestAssembly:
    def test_matches_the_literal_definition(self):
        network = LossNetwork(Hyperbolic(365.0), n_nodes=4)
        network.extend([1, 2, 3, 1], [0, 0, 1, 2], [0.0, 10.0, 40.0, 100.0])
        for at in (0.0, 5.0, 10.0, 99.0, 100.0, 500.0):
            np.testing.assert_allclose(network.matrix(at), reference_matrix(network, at))

    def test_repeat_losses_accumulate(self):
        network = LossNetwork(Uniform(), n_nodes=2)
        network.extend([1, 1, 1], [0, 0, 0], [0.0, 1.0, 2.0])
        assert network.matrix(2.0)[1, 0] == pytest.approx(3.0)

    def test_arc_points_from_loser_to_winner(self):
        network = LossNetwork(Uniform(), n_nodes=2)
        network.add(loser=1, winner=0, at=0.0)
        w = network.matrix(0.0)
        assert w[1, 0] == 1.0 and w[0, 1] == 0.0

    def test_decay_shrinks_old_results(self):
        network = LossNetwork(Hyperbolic(100.0), n_nodes=2)
        network.add(1, 0, at=0.0)
        assert network.matrix(0.0)[1, 0] == pytest.approx(1.0)
        assert network.matrix(100.0)[1, 0] == pytest.approx(0.5)
        assert network.matrix(300.0)[1, 0] == pytest.approx(0.25)

    def test_weights_scale_the_arc(self):
        network = LossNetwork(Uniform(), n_nodes=2)
        network.add(1, 0, at=0.0, weight=2.5)
        assert network.matrix(0.0)[1, 0] == pytest.approx(2.5)

    def test_transposed_is_the_transpose(self):
        network = LossNetwork(Hyperbolic(50.0), n_nodes=5)
        rng = np.random.default_rng(0)
        network.extend(rng.integers(0, 5, 40), rng.integers(0, 5, 40), rng.random(40) * 200)
        np.testing.assert_allclose(
            network.matrix(150.0, transposed=True), network.matrix(150.0).T
        )

    def test_default_time_is_the_latest_result(self):
        network = LossNetwork(Uniform(), n_nodes=2)
        network.extend([1, 1], [0, 0], [0.0, 10.0])
        np.testing.assert_allclose(network.matrix(), network.matrix(10.0))

    def test_empty_network(self):
        network = LossNetwork(Uniform(), n_nodes=3)
        np.testing.assert_allclose(network.matrix(0.0), np.zeros((3, 3)))
        assert network.latest == 0.0 and network.earliest == 0.0


class TestCausality:
    def test_inclusive_controls_simultaneous_results(self):
        network = LossNetwork(Uniform(), n_nodes=2)
        network.add(1, 0, at=10.0)
        assert network.matrix(10.0, inclusive=True)[1, 0] == 1.0
        assert network.matrix(10.0, inclusive=False)[1, 0] == 0.0

    def test_future_results_never_leak(self):
        network = LossNetwork(Uniform(), n_nodes=3)
        network.extend([1, 2], [0, 0], [10.0, 20.0])
        np.testing.assert_allclose(network.matrix(15.0).sum(), 1.0)


class TestOrdering:
    def test_out_of_order_inserts_are_sorted(self):
        network = LossNetwork(Uniform(), n_nodes=3)
        network.extend([1, 2], [0, 0], [50.0, 10.0])
        network.add(2, 1, at=30.0)
        np.testing.assert_allclose(network.times, [10.0, 30.0, 50.0])
        np.testing.assert_allclose(network.matrix(30.0), reference_matrix(network, 30.0))

    def test_ties_keep_a_stable_order(self):
        network = LossNetwork(Uniform(), n_nodes=4)
        network.extend([1, 2, 3], [0, 0, 0], [5.0, 5.0, 5.0])
        np.testing.assert_array_equal(network.events()["loser"], [1, 2, 3])


class TestMaxAge:
    def test_truncates_the_tail(self):
        network = LossNetwork(Hyperbolic(365.0), n_nodes=2, max_age=100.0)
        network.extend([1, 1], [0, 0], [0.0, 90.0])
        assert network.matrix(95.0)[1, 0] > 0.0
        # At t=150 the first result is 150 days old and drops out.
        expected = network.kernel(60.0)
        assert network.matrix(150.0)[1, 0] == pytest.approx(expected)

    def test_matches_the_reference(self):
        network = LossNetwork(Hyperbolic(50.0), n_nodes=4, max_age=120.0)
        rng = np.random.default_rng(1)
        network.extend(rng.integers(0, 4, 60), rng.integers(0, 4, 60), rng.random(60) * 400)
        for at in (50.0, 200.0, 399.0):
            np.testing.assert_allclose(network.matrix(at), reference_matrix(network, at))

    def test_must_be_positive(self):
        with pytest.raises(ValueError, match="max_age"):
            LossNetwork(Uniform(), max_age=0.0)


class TestSparseRepresentation:
    def test_matches_dense(self):
        pytest.importorskip("scipy.sparse")
        rng = np.random.default_rng(2)
        losers = rng.integers(0, 6, 50)
        winners = (losers + 1 + rng.integers(0, 5, 50)) % 6
        times = np.sort(rng.random(50) * 300)

        dense = LossNetwork(Hyperbolic(90.0), n_nodes=6, sparse=False)
        sparse = LossNetwork(Hyperbolic(90.0), n_nodes=6, sparse=True)
        dense.extend(losers, winners, times)
        sparse.extend(losers, winners, times)
        np.testing.assert_allclose(sparse.matrix(200.0).toarray(), dense.matrix(200.0))
        np.testing.assert_allclose(
            sparse.matrix(200.0, transposed=True).toarray(), dense.matrix(200.0).T
        )

    def test_auto_switches_on_node_count(self):
        pytest.importorskip("scipy.sparse")
        assert not LossNetwork(n_nodes=10, dense_max_nodes=512).use_sparse
        assert LossNetwork(n_nodes=1000, dense_max_nodes=512).use_sparse

    def test_empty_sparse_matrix(self):
        pytest.importorskip("scipy.sparse")
        network = LossNetwork(Uniform(), n_nodes=3, sparse=True)
        np.testing.assert_allclose(network.matrix(0.0).toarray(), np.zeros((3, 3)))


class TestIterMatrices:
    def _network(self, kernel, seed=3):
        rng = np.random.default_rng(seed)
        losers = rng.integers(0, 8, 200)
        winners = (losers + 1 + rng.integers(0, 7, 200)) % 8
        times = np.sort(rng.integers(0, 900, 200).astype(float))
        network = LossNetwork(kernel, n_nodes=8)
        network.extend(losers, winners, times)
        return network

    @pytest.mark.parametrize("kernel", [Hyperbolic(365.0), Exponential(120.0), Uniform()])
    @pytest.mark.parametrize("inclusive", [True, False])
    def test_agrees_with_per_time_assembly(self, kernel, inclusive):
        network = self._network(kernel)
        stamps = np.unique(network.times)
        streamed = list(network.iter_matrices(stamps, inclusive=inclusive))
        for stamp, got in zip(stamps, streamed):
            np.testing.assert_allclose(
                got, network.matrix(stamp, inclusive=inclusive), atol=1e-12
            )

    def test_memoryless_kernels_take_the_streaming_path(self):
        stamps = np.arange(0.0, 500.0)
        assert self._network(Exponential(90.0)).can_stream(stamps)
        assert self._network(Uniform()).can_stream(stamps)
        assert not self._network(Hyperbolic(90.0)).can_stream(stamps)

    def test_streaming_declined_for_unsorted_or_windowed_queries(self):
        network = self._network(Exponential(90.0))
        assert not network.can_stream(np.array([10.0, 5.0, 20.0]))
        windowed = LossNetwork(Exponential(90.0), n_nodes=2, max_age=30.0)
        windowed.add(1, 0, at=0.0)
        assert not windowed.can_stream(np.arange(0.0, 10.0))

    def test_streaming_stays_exact_over_a_long_sweep(self):
        # Long enough to cross the internal exact-refresh boundary.
        network = LossNetwork(Exponential(200.0), n_nodes=6)
        rng = np.random.default_rng(4)
        losers = rng.integers(0, 6, 300)
        winners = (losers + 1 + rng.integers(0, 5, 300)) % 6
        network.extend(losers, winners, np.sort(rng.random(300) * 6000))
        stamps = np.arange(0.0, 6000.0, 1.0)
        for stamp, got in zip(stamps, network.iter_matrices(stamps)):
            if stamp % 997 == 0:
                np.testing.assert_allclose(got, network.matrix(stamp), atol=1e-10)

    def test_transposed_streaming(self):
        network = self._network(Exponential(120.0))
        stamps = np.unique(network.times)
        for stamp, got in zip(stamps, network.iter_matrices(stamps, transposed=True)):
            np.testing.assert_allclose(got, network.matrix(stamp).T, atol=1e-12)


class TestValidation:
    def test_length_mismatches(self):
        network = LossNetwork(Uniform())
        with pytest.raises(ValueError, match="losers/winners"):
            network.extend([0, 1], [0], [0.0, 1.0])
        with pytest.raises(ValueError, match="times"):
            network.extend([0, 1], [1, 0], [0.0])
        with pytest.raises(ValueError, match="weights"):
            network.extend([0, 1], [1, 0], [0.0, 1.0], [1.0])

    def test_negative_ids_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            LossNetwork(Uniform()).extend([-1], [0], [0.0])

    def test_non_finite_times_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            LossNetwork(Uniform()).extend([0], [1], [np.inf])

    def test_empty_extend_is_a_no_op(self):
        network = LossNetwork(Uniform())
        network.extend([], [], [])
        assert network.n_events == 0

    def test_nodes_cannot_shrink(self):
        network = LossNetwork(Uniform(), n_nodes=5)
        with pytest.raises(ValueError, match="shrink"):
            network.resize(3)

    def test_node_count_grows_to_fit_ids(self):
        network = LossNetwork(Uniform())
        network.add(3, 7, at=0.0)
        assert network.n_nodes == 8

    def test_repr(self):
        network = LossNetwork(Hyperbolic(365.0), n_nodes=3)
        assert "n_nodes=3" in repr(network)


class TestGrowth:
    def test_many_appends_preserve_history(self):
        network = LossNetwork(Uniform(), n_nodes=2)
        for i in range(500):
            network.add(1, 0, at=float(i))
        assert network.n_events == 500
        assert network.matrix(499.0)[1, 0] == pytest.approx(500.0)

    def test_resizing_nodes_keeps_earlier_results(self):
        network = LossNetwork(Uniform(), n_nodes=2)
        network.add(1, 0, at=0.0)
        network.resize(4)
        w = network.matrix(0.0)
        assert w.shape == (4, 4)
        assert w[1, 0] == 1.0
