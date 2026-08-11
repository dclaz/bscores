"""Time-decay kernels for the match network.

The B-score network weights every past result by how old it is.  The paper's
kernel (Eq. 2 of Arcagni, Candila & Grassi, 2022) is hyperbolic,

.. math::

    f(t^*, t, \\alpha) = \\left(1 + \\frac{t - t^*}{\\alpha}\\right)^{-1},
    \\qquad \\alpha > 0,

so a result loses half its weight after ``alpha`` days and *all* results keep
some weight forever.  ``alpha -> inf`` recovers the unweighted network.

Alternative kernels are provided because the hyperbolic one has a heavy tail
that some sports do not want, and because memoryless kernels admit an exact
O(1) streaming update (see :class:`~bscores.network.LossNetwork`).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

__all__ = [
    "DecayKernel",
    "Hyperbolic",
    "Exponential",
    "Uniform",
    "Window",
    "as_kernel",
]


class DecayKernel(ABC):
    """Base class for age -> weight maps.

    Subclasses implement :meth:`weight`, which must be non-negative and is
    only ever called with ages ``>= 0``.  Negative ages (results from the
    future) are always mapped to zero by :meth:`__call__`.
    """

    #: ``True`` when ``weight(a + b) == weight(a) * weight(b)``, which lets the
    #: network age its accumulator in place instead of rebuilding it.
    memoryless: bool = False

    @abstractmethod
    def weight(self, age: np.ndarray) -> np.ndarray:
        """Weight for a non-negative array of ages, in days."""

    def __call__(self, age: Any) -> np.ndarray:
        ages = np.asarray(age, dtype=np.float64)
        out = np.where(ages >= 0.0, self.weight(np.maximum(ages, 0.0)), 0.0)
        return out if out.ndim else out[()]

    def decay_factor(self, elapsed: float) -> float:
        """Factor by which existing weights shrink over ``elapsed`` days.

        Only meaningful for memoryless kernels; others raise.
        """
        raise NotImplementedError(f"{type(self).__name__} is not memoryless")

    @property
    def half_life(self) -> float:
        """Age at which a result's weight halves, in days."""
        raise NotImplementedError

    def effective_age(self, tol: float = 1e-6) -> float:
        """Age beyond which weights fall below ``tol``; ``inf`` if never."""
        return math.inf


class Hyperbolic(DecayKernel):
    """The paper's kernel: ``1 / (1 + age / alpha)``.

    Parameters
    ----------
    alpha
        Memory parameter in days.  ``f(alpha) == 0.5``, so ``alpha`` *is* the
        half-life.  The paper uses 365 to mirror the ATP/WTA ranking window.
        ``inf`` gives every result equal weight.

    Notes
    -----
    The tail is heavier than the formula suggests, and it matters.  At
    ``alpha=365`` a ten-year-old result still carries weight 0.09; a long
    archive holds thousands of them, and collectively they drown out recent
    form.  On the bundled AFL data this costs about 0.01 of log-loss against
    :class:`Exponential`, and either fix closes the gap:

    * swap to :class:`Exponential`, whose tail decays properly, or
    * keep this kernel and pass ``max_age`` to
      :class:`~bscores.network.LossNetwork` to truncate it.

    Tuning ``alpha`` alone does not fix it: a short half-life suppresses the
    tail only by also discarding useful recent history.  See
    :mod:`bscores.tuning`.
    """

    __slots__ = ("alpha",)

    def __init__(self, alpha: float = 365.0) -> None:
        alpha = float(alpha)
        if not alpha > 0.0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        self.alpha = alpha

    def weight(self, age: np.ndarray) -> np.ndarray:
        if math.isinf(self.alpha):
            return np.ones_like(age)
        return 1.0 / (1.0 + age / self.alpha)

    @property
    def half_life(self) -> float:
        return self.alpha

    def effective_age(self, tol: float = 1e-6) -> float:
        if math.isinf(self.alpha):
            return math.inf
        return self.alpha * (1.0 / tol - 1.0)

    def __repr__(self) -> str:
        return f"Hyperbolic(alpha={self.alpha!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Hyperbolic) and other.alpha == self.alpha

    def __hash__(self) -> int:
        return hash((type(self).__name__, self.alpha))


class Exponential(DecayKernel):
    """Memoryless kernel: ``0.5 ** (age / half_life)``.

    Lighter-tailed than :class:`Hyperbolic` and, being memoryless, it lets the
    network update its accumulator in O(1) per epoch rather than rebuilding
    from the full history.
    """

    __slots__ = ("_half_life", "rate")

    memoryless = True

    def __init__(self, half_life: float = 365.0) -> None:
        half_life = float(half_life)
        if not half_life > 0.0:
            raise ValueError(f"half_life must be positive, got {half_life}")
        self._half_life = half_life
        self.rate = math.log(2.0) / half_life

    def weight(self, age: np.ndarray) -> np.ndarray:
        return np.exp(-self.rate * age)

    def decay_factor(self, elapsed: float) -> float:
        if elapsed < 0.0:
            raise ValueError("elapsed must be non-negative")
        return math.exp(-self.rate * elapsed)

    @property
    def half_life(self) -> float:
        return self._half_life

    def effective_age(self, tol: float = 1e-6) -> float:
        return -math.log(tol) / self.rate

    def __repr__(self) -> str:
        return f"Exponential(half_life={self._half_life!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Exponential) and other._half_life == self._half_life

    def __hash__(self) -> int:
        return hash((type(self).__name__, self._half_life))


class Uniform(DecayKernel):
    """Every result counts the same, i.e. the ``alpha -> inf`` limit."""

    __slots__ = ()

    memoryless = True

    def weight(self, age: np.ndarray) -> np.ndarray:
        return np.ones_like(age)

    def decay_factor(self, elapsed: float) -> float:
        if elapsed < 0.0:
            raise ValueError("elapsed must be non-negative")
        return 1.0

    @property
    def half_life(self) -> float:
        return math.inf

    def __repr__(self) -> str:
        return "Uniform()"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Uniform)

    def __hash__(self) -> int:
        return hash(type(self).__name__)


class Window(DecayKernel):
    """Hard cut-off: ``inner(age)`` while ``age <= width``, zero after.

    Truncating the tail bounds how much history the network has to touch, which
    is the cheapest way to speed up a long back-test.
    """

    __slots__ = ("width", "inner")

    def __init__(self, width: float, inner: DecayKernel | None = None) -> None:
        width = float(width)
        if not width > 0.0:
            raise ValueError(f"width must be positive, got {width}")
        self.width = width
        self.inner = inner if inner is not None else Uniform()

    def weight(self, age: np.ndarray) -> np.ndarray:
        return np.where(age <= self.width, self.inner.weight(age), 0.0)

    @property
    def half_life(self) -> float:
        return min(self.inner.half_life, self.width)

    def effective_age(self, tol: float = 1e-6) -> float:
        return min(self.width, self.inner.effective_age(tol))

    def __repr__(self) -> str:
        return f"Window(width={self.width!r}, inner={self.inner!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Window) and other.width == self.width and other.inner == self.inner

    def __hash__(self) -> int:
        return hash((type(self).__name__, self.width, self.inner))


def as_kernel(kernel: DecayKernel | float | str | None, *, alpha: float = 365.0) -> DecayKernel:
    """Coerce a user-supplied value into a :class:`DecayKernel`.

    ``None`` gives ``Hyperbolic(alpha)``; a bare number is read as a hyperbolic
    ``alpha``; the strings ``"hyperbolic"``, ``"exponential"`` and ``"uniform"``
    select a kernel with the given ``alpha`` as its half-life.
    """
    if kernel is None:
        return Hyperbolic(alpha)
    if isinstance(kernel, DecayKernel):
        return kernel
    if isinstance(kernel, str):
        name = kernel.strip().lower()
        if name in ("hyperbolic", "paper", "bscore"):
            return Hyperbolic(alpha)
        if name in ("exponential", "exp"):
            return Exponential(alpha)
        if name in ("uniform", "none", "constant"):
            return Uniform()
        raise ValueError(f"unknown kernel {kernel!r}")
    if isinstance(kernel, (int, float)):
        return Hyperbolic(float(kernel))
    raise TypeError(f"cannot interpret {kernel!r} as a decay kernel")
