"""The time-weighted loss network.

A B-score network is a directed graph whose nodes are competitors and whose arc
``i -> j`` means "``i`` lost to ``j``".  Every result carries the time it
happened, and the weighted adjacency matrix at time ``t`` is Eq. 1 of the
paper,

.. math::

    W_t = \\sum_{t^* \\in [t_0, t]} f(t^*, t, \\alpha) \\, L_{t^*},

with :math:`L_{t^*}` the matrix counting losses at :math:`t^*` and ``f`` the
decay kernel.  Because the weights depend on ``t - t*`` rather than on ``t*``
alone, the matrix has to be re-derived whenever the clock moves; this module
does so in vectorised O(events) time rather than by looping over history, and
takes an exact O(1)-per-epoch shortcut when the kernel is memoryless.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from typing import Any, Literal

import numpy as np

from ._time import as_day, as_days
from .decay import DecayKernel, as_kernel

__all__ = ["NodeIndex", "LossNetwork"]

#: Node count above which the matrix is assembled as SciPy CSR rather than as a
#: dense ``n x n`` array (which costs 8*n^2 bytes per epoch).
DEFAULT_DENSE_MAX_NODES = 512

#: Epochs between exact rebuilds while streaming, to stop rounding error from
#: accumulating across a long in-place decay chain.
_STREAM_REFRESH = 4096


def _scipy_sparse() -> Any:
    try:
        import scipy.sparse as sp
    except ImportError:  # pragma: no cover - exercised only without SciPy
        return None
    return sp


class NodeIndex:
    """A stable, insertion-ordered name -> integer id map.

    Ids are handed out in first-seen order and never change, so a score vector
    can be indexed by id for the life of a model.
    """

    __slots__ = ("_ids", "_names")

    def __init__(self, names: Iterable[str] = ()) -> None:
        self._ids: dict[str, int] = {}
        self._names: list[str] = []
        for name in names:
            self.add(name)

    def add(self, name: str) -> int:
        """Return the id for ``name``, allocating one if it is new."""
        existing = self._ids.get(name)
        if existing is not None:
            return existing
        index = len(self._names)
        self._ids[name] = index
        self._names.append(name)
        return index

    def add_many(self, names: Iterable[str]) -> np.ndarray:
        """Vectorised :meth:`add`."""
        return np.fromiter((self.add(n) for n in names), dtype=np.int64)

    def get(self, name: str) -> int:
        """Return the id for ``name``, raising :class:`KeyError` if unknown."""
        try:
            return self._ids[name]
        except KeyError:
            raise KeyError(f"unknown competitor {name!r}") from None

    def __contains__(self, name: object) -> bool:
        return name in self._ids

    def __len__(self) -> int:
        return len(self._names)

    def __iter__(self) -> Iterator[str]:
        return iter(self._names)

    def __getitem__(self, index: int | np.integer) -> str:
        return self._names[int(index)]

    @property
    def names(self) -> list[str]:
        """Names in id order."""
        return list(self._names)

    def __repr__(self) -> str:
        return f"NodeIndex({len(self._names)} nodes)"


class LossNetwork:
    """Append-only store of dated results that materialises ``W_t`` on demand.

    Parameters
    ----------
    kernel
        Decay kernel, or an ``alpha`` in days, or a kernel name.  Defaults to
        the paper's :class:`~bscores.decay.Hyperbolic` with ``alpha=365``.
    n_nodes
        Initial node count.  The network grows automatically via
        :meth:`resize`; setting it up front avoids reallocation.
    sparse
        ``"auto"`` picks CSR once the node count exceeds ``dense_max_nodes``.
    max_age
        Drop results older than this many days.  Purely an optimisation for
        heavy-tailed kernels: it bounds the work per epoch at the cost of a
        (documented, controllable) truncation error.

    Notes
    -----
    Results may be appended out of chronological order; the store re-sorts
    lazily on the next query.
    """

    def __init__(
        self,
        kernel: DecayKernel | float | str | None = None,
        *,
        n_nodes: int = 0,
        sparse: Literal["auto"] | bool = "auto",
        dense_max_nodes: int = DEFAULT_DENSE_MAX_NODES,
        max_age: float | None = None,
    ) -> None:
        self.kernel = as_kernel(kernel)
        self._n_nodes = int(n_nodes)
        self.sparse = sparse
        self.dense_max_nodes = int(dense_max_nodes)
        self.max_age = None if max_age is None else float(max_age)
        if self.max_age is not None and not self.max_age > 0:
            raise ValueError("max_age must be positive")

        self._size = 0
        self._capacity = 0
        self._loser = np.empty(0, dtype=np.int64)
        self._winner = np.empty(0, dtype=np.int64)
        self._time = np.empty(0, dtype=np.float64)
        self._weight = np.empty(0, dtype=np.float64)
        self._sorted = True
        self._flat: np.ndarray | None = None
        self._flat_key: tuple[int, bool] | None = None

    # ------------------------------------------------------------------
    # ingestion
    # ------------------------------------------------------------------
    def resize(self, n_nodes: int) -> None:
        """Grow the node count.  Shrinking is not allowed."""
        n_nodes = int(n_nodes)
        if n_nodes < self._n_nodes:
            raise ValueError(f"cannot shrink from {self._n_nodes} to {n_nodes} nodes")
        if n_nodes != self._n_nodes:
            self._n_nodes = n_nodes
            self._flat = None

    def _reserve(self, extra: int) -> None:
        needed = self._size + extra
        if needed <= self._capacity:
            return
        capacity = max(16, self._capacity)
        while capacity < needed:
            capacity *= 2
        self._loser = np.resize(self._loser, capacity)
        self._winner = np.resize(self._winner, capacity)
        self._time = np.resize(self._time, capacity)
        self._weight = np.resize(self._weight, capacity)
        self._capacity = capacity

    def add(self, loser: int, winner: int, at: Any, weight: float = 1.0) -> None:
        """Record one result: ``loser`` lost to ``winner`` at time ``at``."""
        self.extend([loser], [winner], [at], [weight])

    def extend(
        self,
        losers: Sequence[int] | np.ndarray,
        winners: Sequence[int] | np.ndarray,
        times: Any,
        weights: Sequence[float] | np.ndarray | None = None,
    ) -> None:
        """Record many results at once."""
        loser = np.asarray(losers, dtype=np.int64).ravel()
        winner = np.asarray(winners, dtype=np.int64).ravel()
        time = np.atleast_1d(as_days(times)).astype(np.float64, copy=False).ravel()
        if loser.shape != winner.shape:
            raise ValueError(f"losers/winners length mismatch: {loser.shape} vs {winner.shape}")
        if time.shape != loser.shape:
            raise ValueError(f"times length mismatch: {time.shape} vs {loser.shape}")
        if weights is None:
            weight = np.ones(loser.shape, dtype=np.float64)
        else:
            weight = np.asarray(weights, dtype=np.float64).ravel()
            if weight.shape != loser.shape:
                raise ValueError(f"weights length mismatch: {weight.shape} vs {loser.shape}")
        if loser.size == 0:
            return
        if np.any(loser < 0) or np.any(winner < 0):
            raise ValueError("node ids must be non-negative")
        if np.any(~np.isfinite(time)):
            raise ValueError("times must be finite")

        self.resize(max(self._n_nodes, int(loser.max()) + 1, int(winner.max()) + 1))

        n = loser.size
        self._reserve(n)
        stop = self._size + n
        self._loser[self._size : stop] = loser
        self._winner[self._size : stop] = winner
        self._time[self._size : stop] = time
        self._weight[self._size : stop] = weight

        if self._sorted and self._size and time[0] < self._time[self._size - 1]:
            self._sorted = False
        if self._sorted and n > 1 and np.any(np.diff(time) < 0):
            self._sorted = False

        self._size = stop
        self._flat = None

    # ------------------------------------------------------------------
    # views
    # ------------------------------------------------------------------
    def _ensure_sorted(self) -> None:
        if self._sorted:
            return
        order = np.argsort(self._time[: self._size], kind="stable")
        self._loser[: self._size] = self._loser[: self._size][order]
        self._winner[: self._size] = self._winner[: self._size][order]
        self._time[: self._size] = self._time[: self._size][order]
        self._weight[: self._size] = self._weight[: self._size][order]
        self._sorted = True
        self._flat = None

    @property
    def n_nodes(self) -> int:
        """Number of nodes the matrix is built over."""
        return self._n_nodes

    @property
    def n_events(self) -> int:
        """Number of recorded results."""
        return self._size

    def __len__(self) -> int:
        return self._size

    @property
    def times(self) -> np.ndarray:
        """Event times in days since the epoch, ascending."""
        self._ensure_sorted()
        return self._time[: self._size].copy()

    @property
    def latest(self) -> float:
        """Time of the most recent result, or ``0.0`` when empty."""
        self._ensure_sorted()
        return float(self._time[self._size - 1]) if self._size else 0.0

    @property
    def earliest(self) -> float:
        """Time of the first result, or ``0.0`` when empty."""
        self._ensure_sorted()
        return float(self._time[0]) if self._size else 0.0

    @property
    def use_sparse(self) -> bool:
        """Whether matrices are assembled as SciPy CSR."""
        if self.sparse == "auto":
            return self._n_nodes > self.dense_max_nodes and _scipy_sparse() is not None
        if self.sparse and _scipy_sparse() is None:
            raise ImportError("sparse=True requires SciPy; install scipy or pass sparse=False")
        return bool(self.sparse)

    def _flat_index(self, transposed: bool) -> np.ndarray:
        key = (self._n_nodes, transposed)
        if self._flat is None or self._flat_key != key:
            self._ensure_sorted()
            rows, cols = self._loser, self._winner
            if transposed:
                rows, cols = cols, rows
            self._flat = rows[: self._size] * self._n_nodes + cols[: self._size]
            self._flat_key = key
        return self._flat

    def _bounds(self, at: float, inclusive: bool) -> tuple[int, int]:
        """Half-open slice of the event arrays contributing at time ``at``."""
        self._ensure_sorted()
        times = self._time[: self._size]
        stop = int(np.searchsorted(times, at, side="right" if inclusive else "left"))
        start = 0
        if self.max_age is not None:
            start = int(np.searchsorted(times, at - self.max_age, side="left"))
        return start, stop

    # ------------------------------------------------------------------
    # matrix assembly
    # ------------------------------------------------------------------
    def matrix(self, at: Any = None, *, inclusive: bool = True, transposed: bool = False) -> Any:
        """Assemble ``W_t`` (or ``W_t'``) at time ``at``.

        Parameters
        ----------
        at
            Evaluation time.  ``None`` means "now", i.e. the most recent result.
        inclusive
            Whether results timestamped exactly ``at`` count.  Pass ``False`` to
            build the matrix a prediction may legitimately use for a match
            starting at ``at``.
        transposed
            Build ``W_t'`` directly.  The centrality solver wants the transpose,
            and assembling it costs the same as assembling ``W_t``.

        Returns
        -------
        numpy.ndarray or scipy.sparse.csr_matrix
        """
        time = self.latest if at is None else as_day(at)
        start, stop = self._bounds(time, inclusive)
        n = self._n_nodes

        if stop <= start or n == 0:
            return self._empty(n)

        ages = time - self._time[start:stop]
        weights = self._weight[start:stop] * self.kernel(ages)

        if self.use_sparse:
            sp = _scipy_sparse()
            rows, cols = self._loser[start:stop], self._winner[start:stop]
            if transposed:
                rows, cols = cols, rows
            return sp.coo_matrix((weights, (rows, cols)), shape=(n, n)).tocsr()

        flat = self._flat_index(transposed)[start:stop]
        return np.bincount(flat, weights=weights, minlength=n * n).reshape(n, n)

    def _empty(self, n: int) -> Any:
        if self.use_sparse:
            return _scipy_sparse().csr_matrix((n, n), dtype=np.float64)
        return np.zeros((n, n), dtype=np.float64)

    def can_stream(self, times: np.ndarray) -> bool:
        """Whether :meth:`iter_matrices` can take the O(1)-per-epoch path."""
        return (
            self.kernel.memoryless
            and self.max_age is None
            and not self.use_sparse
            and times.size > 1
            and bool(np.all(np.diff(times) >= 0))
        )

    def iter_matrices(
        self,
        times: Any,
        *,
        inclusive: bool = True,
        transposed: bool = False,
    ) -> Iterator[Any]:
        """Yield ``W_t`` for each time in ``times``.

        For a memoryless kernel on ascending times the accumulator is aged in
        place, which is exact and costs O(n^2 + new results) per epoch instead
        of O(all results).  Otherwise each epoch is rebuilt from scratch.
        """
        stamps = np.atleast_1d(as_days(times)).astype(np.float64, copy=False)
        if not self.can_stream(stamps):
            for stamp in stamps:
                yield self.matrix(stamp, inclusive=inclusive, transposed=transposed)
            return

        self._ensure_sorted()
        n = self._n_nodes
        flat = self._flat_index(transposed)
        event_times = self._time[: self._size]
        acc = np.zeros(n * n, dtype=np.float64)
        pos = 0
        previous: float | None = None

        for epoch, stamp in enumerate(stamps):
            if epoch and epoch % _STREAM_REFRESH == 0:
                exact = self.matrix(stamp, inclusive=inclusive, transposed=transposed)
                acc = exact.ravel()
                refresh_side: Literal["left", "right"] = "right" if inclusive else "left"
                pos = int(np.searchsorted(event_times, stamp, side=refresh_side))
                previous = float(stamp)
                yield exact
                continue
            if previous is not None:
                factor = self.kernel.decay_factor(float(stamp) - previous)
                if factor != 1.0:
                    acc *= factor
            side: Literal["left", "right"] = "right" if inclusive else "left"
            stop = int(np.searchsorted(event_times, stamp, side=side))
            if stop > pos:
                ages = float(stamp) - event_times[pos:stop]
                weights = self._weight[pos:stop] * self.kernel(ages)
                acc += np.bincount(flat[pos:stop], weights=weights, minlength=n * n)
                pos = stop
            previous = float(stamp)
            # Copy so the caller owns each epoch's matrix; the accumulator is
            # reused in place and would otherwise alias every yielded array.
            yield acc.reshape(n, n).copy()

    def events(self) -> dict[str, np.ndarray]:
        """Copy of the stored results, ascending by time."""
        self._ensure_sorted()
        return {
            "loser": self._loser[: self._size].copy(),
            "winner": self._winner[: self._size].copy(),
            "time": self._time[: self._size].copy(),
            "weight": self._weight[: self._size].copy(),
        }

    def __repr__(self) -> str:
        return (
            f"LossNetwork(kernel={self.kernel!r}, n_nodes={self._n_nodes}, "
            f"n_events={self._size})"
        )
