"""Decay kernels, including the paper's Eq. 2."""

from __future__ import annotations

import math

import numpy as np
import pytest

from bscores.decay import Exponential, Hyperbolic, Uniform, Window, as_kernel


class TestHyperbolic:
    def test_matches_equation_2(self):
        alpha = 365.0
        kernel = Hyperbolic(alpha)
        ages = np.array([0.0, 10.0, 365.0, 1000.0])
        expected = 1.0 / (1.0 + ages / alpha)
        np.testing.assert_allclose(kernel(ages), expected)

    def test_fresh_results_weigh_one(self):
        assert Hyperbolic(365.0)(0.0) == pytest.approx(1.0)

    def test_alpha_is_the_half_life(self):
        # "if the difference is alpha then f = 0.5"
        for alpha in (1.0, 30.0, 365.0, 1e4):
            assert Hyperbolic(alpha)(alpha) == pytest.approx(0.5)
            assert Hyperbolic(alpha).half_life == alpha

    def test_decreasing_in_age(self):
        kernel = Hyperbolic(100.0)
        weights = kernel(np.arange(0.0, 1000.0, 7.0))
        assert np.all(np.diff(weights) < 0)

    def test_tail_never_reaches_zero(self):
        assert Hyperbolic(365.0)(1e6) > 0.0

    def test_infinite_alpha_is_uniform(self):
        kernel = Hyperbolic(math.inf)
        np.testing.assert_allclose(kernel([0.0, 1e6]), [1.0, 1.0])
        assert kernel.effective_age() == math.inf

    def test_future_results_do_not_count(self):
        assert Hyperbolic(365.0)(-1.0) == 0.0

    def test_alpha_must_be_positive(self):
        with pytest.raises(ValueError, match="alpha"):
            Hyperbolic(0.0)
        with pytest.raises(ValueError, match="alpha"):
            Hyperbolic(-5.0)

    def test_equality_and_hash(self):
        assert Hyperbolic(365.0) == Hyperbolic(365.0)
        assert Hyperbolic(365.0) != Hyperbolic(30.0)
        assert len({Hyperbolic(365.0), Hyperbolic(365.0)}) == 1


class TestExponential:
    def test_half_life_semantics(self):
        kernel = Exponential(30.0)
        assert kernel(0.0) == pytest.approx(1.0)
        assert kernel(30.0) == pytest.approx(0.5)
        assert kernel(60.0) == pytest.approx(0.25)

    def test_memoryless_identity(self):
        kernel = Exponential(45.0)
        a, b = 12.0, 33.0
        assert kernel(a + b) == pytest.approx(kernel(a) * kernel(b))
        assert kernel.decay_factor(b) == pytest.approx(kernel(b))

    def test_decay_factor_rejects_negative_elapsed(self):
        with pytest.raises(ValueError):
            Exponential(10.0).decay_factor(-1.0)

    def test_effective_age(self):
        kernel = Exponential(30.0)
        assert kernel(kernel.effective_age(1e-6)) == pytest.approx(1e-6)

    def test_half_life_must_be_positive(self):
        with pytest.raises(ValueError, match="half_life"):
            Exponential(-1.0)


class TestUniform:
    def test_constant(self):
        np.testing.assert_allclose(Uniform()([0.0, 1.0, 1e9]), 1.0)

    def test_memoryless(self):
        assert Uniform().memoryless
        assert Uniform().decay_factor(1234.0) == 1.0
        assert Uniform().half_life == math.inf


class TestWindow:
    def test_hard_cutoff(self):
        kernel = Window(100.0)
        assert kernel(99.0) == 1.0
        assert kernel(100.0) == 1.0
        assert kernel(100.1) == 0.0

    def test_wraps_an_inner_kernel(self):
        inner = Hyperbolic(50.0)
        kernel = Window(100.0, inner)
        assert kernel(50.0) == pytest.approx(inner(50.0))
        assert kernel(150.0) == 0.0
        assert kernel.effective_age() == 100.0

    def test_width_must_be_positive(self):
        with pytest.raises(ValueError, match="width"):
            Window(0.0)


class TestAsKernel:
    def test_none_is_the_paper_default(self):
        assert as_kernel(None) == Hyperbolic(365.0)
        assert as_kernel(None, alpha=30.0) == Hyperbolic(30.0)

    def test_number_reads_as_alpha(self):
        assert as_kernel(90.0) == Hyperbolic(90.0)

    def test_names(self):
        assert as_kernel("hyperbolic", alpha=10.0) == Hyperbolic(10.0)
        assert as_kernel("exponential", alpha=10.0) == Exponential(10.0)
        assert as_kernel("uniform") == Uniform()

    def test_passthrough(self):
        kernel = Window(5.0)
        assert as_kernel(kernel) is kernel

    def test_rejects_nonsense(self):
        with pytest.raises(ValueError, match="unknown kernel"):
            as_kernel("wat")
        with pytest.raises(TypeError):
            as_kernel(object())


def test_scalar_input_returns_scalar():
    value = Hyperbolic(365.0)(10.0)
    assert np.ndim(value) == 0
