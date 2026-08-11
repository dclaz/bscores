"""Bonacich (directed eigenvector) centrality.

For a weighted digraph with adjacency matrix :math:`W`, where ``W[i, j]`` is the
weight of the arc ``i -> j``, the Bonacich centrality is the principal
eigenvector of :math:`W'` (Eq. 11 of the paper):

.. math::

    x = \\frac{1}{\\rho} W' x, \\qquad \\lVert x \\rVert_2 = 1,

with :math:`\\rho` the spectral radius.  Read on the B-score network — where an
arc points from the loser to the winner — this says a competitor scores highly
when the competitors *pointing at* it score highly, i.e. when it beats strong
opponents.

Two solvers are available.  The default is a shifted power iteration, which is
O(nnz) per step, supports warm starts (worth a large constant factor when
sweeping a rating history) and never materialises a dense factorisation.  The
``"dense"`` solver calls LAPACK through :func:`numpy.linalg.eig`; it is O(n^3)
but does not iterate, and serves as the reference implementation in the tests.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

__all__ = [
    "EigenResult",
    "principal_eigenvector",
    "bonacich_centrality",
    "neumann_centrality",
    "in_strength",
    "out_strength",
]

#: Above this many nodes the dense fallback is refused rather than silently
#: allocating an n-by-n array and running an O(n^3) decomposition.
DENSE_LIMIT = 2048

#: ``||A x_k||`` below this fraction of its starting value means the operator is
#: (numerically) nilpotent, i.e. the network has no directed cycle yet.
_DEGENERATE_RATIO = 1e-10


@dataclass(frozen=True)
class EigenResult:
    """Outcome of a principal-eigenvector solve."""

    vector: np.ndarray
    """Unit-norm, non-negative principal eigenvector."""

    eigenvalue: float
    """Spectral radius (Rayleigh quotient of the returned vector)."""

    iterations: int
    """Power-iteration steps taken; 0 for the direct solver."""

    converged: bool
    """Whether the requested tolerance was reached."""

    method: str
    """Solver that produced the result."""


def _matvec(matrix: Any, x: np.ndarray) -> np.ndarray:
    out = matrix @ x
    return np.asarray(out, dtype=np.float64).ravel()


def _as_transposed(matrix: Any, transposed: bool) -> Any:
    """Return the operator ``A`` such that ``A @ x`` evaluates ``W' x``."""
    if transposed:
        return matrix
    transpose = matrix.T
    # A CSR matrix transposes to CSC, whose mat-vec is markedly slower; pay the
    # one-off conversion so the iteration runs on CSR.
    tocsr = getattr(transpose, "tocsr", None)
    if tocsr is not None:
        return tocsr()
    return transpose


def principal_eigenvector(
    matrix: Any,
    *,
    x0: np.ndarray | None = None,
    tol: float = 1e-12,
    max_iter: int = 10_000,
    shift: float = 0.15,
    regularization: float = 0.0,
    method: Literal["auto", "power", "dense"] = "auto",
) -> EigenResult:
    """Dominant eigenvector of a non-negative square operator.

    Parameters
    ----------
    matrix
        Square, non-negative ``(n, n)`` array or SciPy sparse matrix.  The
        returned vector ``x`` satisfies ``matrix @ x == rho * x``.
    x0
        Warm start.  Ignored when it is all zeros.  Successive epochs of a
        rating history barely move the eigenvector, so seeding with the
        previous answer typically cuts the iteration count by an order of
        magnitude.
    tol
        Convergence threshold on ``max|x_{k+1} - x_k|``.
    max_iter
        Iteration cap before giving up (and falling back, under ``"auto"``).
    shift
        Spectral shift as a fraction of the estimated spectral radius: the
        iteration runs on ``matrix + sigma * I``.  For an irreducible
        non-negative matrix this leaves the principal eigenvector exactly
        unchanged while making the operator aperiodic, so the bipartite-ish
        networks common in an opening round converge instead of oscillating.
        Set to ``0`` for the unshifted iteration.
    regularization
        Adds ``eps * (J - I)`` to the underlying network — a uniform "everyone
        has played everyone a little" term that makes the graph irreducible, so
        the Perron-Frobenius theorem applies and the answer is unique.  Applied
        implicitly, without densifying the operator, and it takes an acyclic
        network back onto the eigenvector path.
    method
        ``"power"`` for the iterative solver, ``"dense"`` for LAPACK, or
        ``"auto"`` (default) to run the power iteration and fall back to dense
        if it fails to converge.  All three first check whether the network is
        acyclic, in which case there is no eigenvector and the answer comes
        from :func:`neumann_centrality` regardless of ``method``.

    Returns
    -------
    EigenResult
        ``method`` records which route produced the answer: ``"power"``,
        ``"dense"``, or ``"acyclic"`` for the degenerate fallback.
    """
    n = matrix.shape[0]
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"matrix must be square, got shape {matrix.shape}")
    if n == 0:
        return EigenResult(np.zeros(0), 0.0, 0, True, "empty")
    if shift < 0.0:
        raise ValueError("shift must be non-negative")
    if regularization < 0.0:
        raise ValueError("regularization must be non-negative")

    if method not in ("auto", "power", "dense"):
        raise ValueError(f"unknown method {method!r}")

    # A non-negative operator has spectral radius zero exactly when its digraph
    # is acyclic — nobody has beaten anybody who beat anybody else, in a loop.
    # There is then no eigenvector to find: every eigenvalue is 0, the matrix is
    # a single Jordan chain, and power iteration crawls towards an arbitrary
    # basis vector.  Testing for a cycle up front costs one sparsity-pattern
    # sweep and keeps the degenerate case both fast and well defined.
    if not _has_cycle(matrix, regularization):
        fallback = _neumann_solve(matrix, regularization=regularization, max_terms=n)
        if fallback is not None:
            return fallback

    if method == "dense":
        return _dense_solve(matrix, regularization=regularization)

    result = _power_solve(
        matrix,
        x0=x0,
        tol=tol,
        max_iter=max_iter,
        shift=shift,
        regularization=regularization,
    )
    if result.converged or method == "power":
        return result
    if n <= DENSE_LIMIT:
        return _dense_solve(matrix, regularization=regularization)
    warnings.warn(
        f"power iteration did not converge in {max_iter} iterations "
        f"(n={n} exceeds the dense fallback limit of {DENSE_LIMIT}); "
        "raise max_iter, loosen tol, or increase shift",
        RuntimeWarning,
        stacklevel=2,
    )
    return result


def _has_cycle(matrix: Any, regularization: float = 0.0) -> bool:
    """Whether the operator's digraph contains a directed cycle.

    Kahn's algorithm, peeling every source at once so each round is a couple of
    vector operations.  A cyclic network stalls on the first round, which is the
    case that matters: the check is then essentially free.
    """
    n = matrix.shape[0]
    if n == 0:
        return False
    if regularization > 0.0 and n > 1:
        # The uniform term connects every pair, so 1 -> 2 -> 1 is a cycle.
        return True

    rows, cols = matrix.nonzero()
    rows = np.asarray(rows, dtype=np.int64).ravel()
    cols = np.asarray(cols, dtype=np.int64).ravel()
    if rows.size == 0:
        return False

    live_edge = np.ones(rows.size, dtype=bool)
    live_node = np.ones(n, dtype=bool)
    while True:
        in_degree = np.bincount(cols[live_edge], minlength=n)
        peel = live_node & (in_degree == 0)
        if not peel.any():
            return True
        live_node &= ~peel
        if not live_node.any():
            return False
        live_edge &= live_node[rows]


def _power_solve(
    matrix: Any,
    *,
    x0: np.ndarray | None,
    tol: float,
    max_iter: int,
    shift: float,
    regularization: float,
) -> EigenResult:
    n = matrix.shape[0]

    if x0 is None:
        x = np.full(n, 1.0 / np.sqrt(n), dtype=np.float64)
    else:
        x = np.array(x0, dtype=np.float64).ravel()
        if x.shape != (n,):
            raise ValueError(f"x0 must have shape ({n},), got {x.shape}")
        x = np.abs(x)
        norm = np.linalg.norm(x)
        if not norm > 0.0:
            x = np.full(n, 1.0 / np.sqrt(n), dtype=np.float64)
        else:
            x /= norm

    sigma = 0.0
    initial_norm = 0.0
    previous = x
    for iteration in range(1, max_iter + 1):
        y = _matvec(matrix, x)
        if regularization:
            y += regularization * (x.sum() - x)
        raw_norm = np.linalg.norm(y)

        if iteration == 1:
            if not raw_norm > 0.0:
                # No edges reachable from the start vector: the network carries
                # no information yet, and every score is zero.
                return EigenResult(np.zeros(n), 0.0, iteration, True, "power")
            sigma = shift * raw_norm
            initial_norm = raw_norm
        elif not raw_norm > _DEGENERATE_RATIO * initial_norm:
            # Safety net for a network the cycle test cleared but whose spectral
            # radius is numerically zero: there is nothing left to converge to.
            return EigenResult(x, 0.0, iteration, False, "power")

        y += sigma * x
        norm = np.linalg.norm(y)
        if not norm > 0.0:  # pragma: no cover - implied by the raw_norm guard
            return EigenResult(np.zeros(n), 0.0, iteration, True, "power")
        previous, x = x, y / norm

        if np.max(np.abs(x - previous)) <= tol:
            return EigenResult(x, _rayleigh(matrix, x, regularization), iteration, True, "power")

    return EigenResult(x, _rayleigh(matrix, x, regularization), max_iter, False, "power")


def _rayleigh(matrix: Any, x: np.ndarray, regularization: float) -> float:
    y = _matvec(matrix, x)
    if regularization:
        y += regularization * (x.sum() - x)
    return float(x @ y)


def _apply(matrix: Any, x: np.ndarray, regularization: float) -> np.ndarray:
    y = _matvec(matrix, x)
    if regularization:
        y += regularization * (x.sum() - x)
    return y


#: Weight each extra link in the "beat someone who beat someone" chain carries,
#: relative to the one before it, in the acyclic fallback.
DEFAULT_DAMPING = 0.5


def _neumann_solve(
    matrix: Any,
    *,
    regularization: float,
    max_terms: int,
    damping: float = DEFAULT_DAMPING,
    tol: float = 1e-14,
) -> EigenResult | None:
    """Finite Neumann series, for networks with no directed cycle.

    Returns ``None`` when the series does not terminate, which means the
    operator is not nilpotent and the ordinary eigenvector solve applies.

    The step factor is ``damping / ||A||_inf`` rather than a bare constant, so
    the result is invariant to rescaling the whole network — doubling every
    arc weight must not move a rating, exactly as it does not for the
    eigenvector solve.
    """
    n = matrix.shape[0]
    if n == 0:
        return EigenResult(np.zeros(0), 0.0, 0, True, "acyclic")

    term = _apply(matrix, np.ones(n, dtype=np.float64), regularization)
    scale = float(np.linalg.norm(term))
    if not scale > 0.0:
        return EigenResult(np.zeros(n), 0.0, 1, True, "acyclic")

    # (A 1)_i is row i's weight sum, so its maximum is ||A||_inf.
    beta = damping / float(np.max(np.abs(term)))
    total = term.copy()

    for step in range(2, max(max_terms, 2) + 2):
        term = beta * _apply(matrix, term, regularization)
        magnitude = float(np.linalg.norm(term))
        if magnitude <= tol * scale:
            vector = np.clip(total, 0.0, None)
            norm = float(np.linalg.norm(vector))
            if not norm > 0.0:  # pragma: no cover - guarded by the scale check
                return EigenResult(np.zeros(n), 0.0, step, True, "acyclic")
            return EigenResult(vector / norm, 0.0, step, True, "acyclic")
        total += term
    return None


def _dense_solve(matrix: Any, *, regularization: float) -> EigenResult:
    n = matrix.shape[0]
    if n > DENSE_LIMIT:
        raise ValueError(
            f"dense solve refused for n={n} (limit {DENSE_LIMIT}); use method='power'"
        )
    dense = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix, dtype=np.float64)
    dense = np.array(dense, dtype=np.float64, copy=True)
    if regularization:
        dense += regularization * (np.ones((n, n)) - np.eye(n))

    if not dense.any():
        return EigenResult(np.zeros(n), 0.0, 0, True, "dense")

    values, vectors = np.linalg.eig(dense)
    index = int(np.argmax(np.abs(values)))
    vector = vectors[:, index]

    # The Perron vector of a non-negative matrix is real and single-signed;
    # discard the numerical imaginary part and orient it positively.
    vector = np.real_if_close(vector, tol=1e6)
    vector = np.real(vector)
    dominant = vector[int(np.argmax(np.abs(vector)))]
    if dominant < 0:
        vector = -vector
    vector = np.clip(vector, 0.0, None)

    norm = np.linalg.norm(vector)
    if not norm > 0.0:  # pragma: no cover - degenerate spectra
        return EigenResult(np.zeros(n), 0.0, 0, True, "dense")
    vector = vector / norm
    return EigenResult(vector, float(np.real(values[index])), 0, True, "dense")


def bonacich_centrality(
    weights: Any,
    *,
    transposed: bool = False,
    x0: np.ndarray | None = None,
    tol: float = 1e-12,
    max_iter: int = 10_000,
    shift: float = 0.15,
    regularization: float = 0.0,
    method: Literal["auto", "power", "dense"] = "auto",
    return_info: bool = False,
) -> np.ndarray | EigenResult:
    """B-scores for a weighted loss network.

    Parameters
    ----------
    weights
        The weighted adjacency matrix ``W`` with ``W[i, j]`` the decayed number
        of matches ``i`` lost to ``j``.
    transposed
        Set when ``weights`` is already ``W'`` (which :meth:`bscores.network.
        LossNetwork.matrix` can build directly, saving a transpose in the hot
        loop).
    return_info
        Return the full :class:`EigenResult` instead of just the vector.

    Returns
    -------
    numpy.ndarray or EigenResult
        Unit-norm, non-negative scores, one per node.

    Other parameters are forwarded to :func:`principal_eigenvector`.
    """
    operator = _as_transposed(weights, transposed)
    result = principal_eigenvector(
        operator,
        x0=x0,
        tol=tol,
        max_iter=max_iter,
        shift=shift,
        regularization=regularization,
        method=method,
    )
    return result if return_info else result.vector


def neumann_centrality(
    weights: Any,
    *,
    transposed: bool = False,
    damping: float = DEFAULT_DAMPING,
    max_terms: int | None = None,
) -> np.ndarray:
    """Graded scores for a network with no directed cycle.

    When nobody has beaten anybody who beat anybody else in a loop — the first
    rounds of any competition — the loss matrix is nilpotent, its spectral
    radius is zero and Eq. 11 has no solution.  This is the documented
    fallback that :func:`bonacich_centrality` reaches for automatically:

    .. math::

        x \\propto \\sum_{k \\ge 1} \\beta^{k-1} (W')^k \\mathbf{1}

    The first term is each competitor's decayed win count; the second adds the
    win counts of everyone it beat; and so on.  Competitors who have never won
    still score exactly zero, and the series is finite, so this costs at most
    ``n`` mat-vecs.

    Parameters
    ----------
    damping
        How much each extra link in the chain is discounted, relative to
        ``||W||_inf``.  Like the eigenvector solve, the result does not depend
        on the overall scale of ``weights``.

    Raises
    ------
    ValueError
        If the network *does* contain a cycle, where
        :func:`bonacich_centrality` is the right call.
    """
    operator = _as_transposed(weights, transposed)
    terms = operator.shape[0] if max_terms is None else int(max_terms)
    result = _neumann_solve(
        operator, regularization=0.0, max_terms=terms, damping=damping
    )
    if result is None:
        raise ValueError(
            "the network contains a directed cycle, so the Neumann series does "
            "not terminate; use bonacich_centrality() instead"
        )
    return result.vector


def in_strength(weights: Any) -> np.ndarray:
    """Sum of incoming arc weights per node (``s_in = W' 1``).

    The degree-style counterpart to the B-score: it counts *how much* was won
    without caring *who* it was won against.  Useful as a diagnostic baseline.
    """
    ones = np.ones(weights.shape[0], dtype=np.float64)
    return np.asarray(weights.T @ ones, dtype=np.float64).ravel()


def out_strength(weights: Any) -> np.ndarray:
    """Sum of outgoing arc weights per node (``s_out = W 1``)."""
    ones = np.ones(weights.shape[0], dtype=np.float64)
    return np.asarray(weights @ ones, dtype=np.float64).ravel()
