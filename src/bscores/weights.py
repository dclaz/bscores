"""Weighting results by how much they should count.

The paper's network counts each loss once.  Two extensions are worth having,
because both are free — the network already carries a per-arc weight — and both
are commonly wanted:

* **Margin of victory.** A 100-point thrashing says more about the gap between
  two sides than a one-point escape.  :func:`margin_weight` turns a margin into
  an arc weight.
* **Match importance.** A final can be made to count for more than a
  regular-season fixture; see :func:`importance_weight`.

Weights scale the arc, not the rating: multiplying *every* weight by a constant
leaves the B-scores untouched, because the principal eigenvector is invariant to
the scale of the matrix.  Only the *relative* weighting matters.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

__all__ = ["MARGIN_SCHEMES", "margin_weight", "importance_weight"]

MarginScheme = Literal["uniform", "linear", "sqrt", "log"]

#: Schemes accepted by :func:`margin_weight`, in increasing order of how hard
#: they let a blowout dominate a narrow win.
MARGIN_SCHEMES: tuple[str, ...] = ("uniform", "sqrt", "log", "linear")


def margin_weight(
    margin: Any,
    *,
    scheme: MarginScheme = "sqrt",
    scale: float = 36.0,
    cap: float | None = 3.0,
) -> np.ndarray:
    """Arc weight for each result, as a function of the winning margin.

    Every scheme returns 1.0 for a margin of zero and grows from there, so the
    weights stay comparable with the paper's unweighted network.

    Parameters
    ----------
    margin
        Signed or unsigned margins; only the magnitude is used.
    scheme
        ``"uniform"`` ignores the margin (the paper's model).  ``"linear"`` uses
        ``1 + |m|/scale``, ``"sqrt"`` uses ``sqrt(1 + |m|/scale)`` and ``"log"``
        uses ``1 + log1p(|m|/scale)`` — increasingly flat, so a blowout counts
        for more but not proportionally more.
    scale
        Margin at which a linear weight reaches 2.  The default, 36 points, is
        roughly one standard deviation of the AFL margin distribution; set it to
        the typical margin of whatever sport you are rating.
    cap
        Upper bound on the weight, or ``None`` for unbounded.  Caps the
        influence of freak scorelines.

    Returns
    -------
    numpy.ndarray

    Examples
    --------
    >>> from bscores.weights import margin_weight
    >>> [round(float(w), 3) for w in margin_weight([0, 36, 144], scheme="linear")]
    [1.0, 2.0, 3.0]
    """
    values = np.abs(np.asarray(margin, dtype=np.float64))
    if not scale > 0.0:
        raise ValueError(f"scale must be positive, got {scale}")

    if scheme == "uniform":
        out = np.ones_like(values)
    elif scheme == "linear":
        out = 1.0 + values / scale
    elif scheme == "sqrt":
        out = np.sqrt(1.0 + values / scale)
    elif scheme == "log":
        out = 1.0 + np.log1p(values / scale)
    else:
        raise ValueError(f"unknown scheme {scheme!r}; choose from {MARGIN_SCHEMES}")

    if cap is not None:
        if not cap >= 1.0:
            raise ValueError(f"cap must be at least 1, got {cap}")
        out = np.minimum(out, cap)
    return out


def importance_weight(flag: Any, *, weight: float = 2.0) -> np.ndarray:
    """Arc weight of ``weight`` where ``flag`` is true and 1.0 elsewhere.

    Built for a play-off indicator, but any boolean mask works.

    Examples
    --------
    >>> from bscores.weights import importance_weight
    >>> [float(w) for w in importance_weight([True, False], weight=2.0)]
    [2.0, 1.0]
    """
    mask = np.asarray(flag).astype(bool)
    if not weight > 0.0:
        raise ValueError(f"weight must be positive, got {weight}")
    return np.where(mask, float(weight), 1.0)
