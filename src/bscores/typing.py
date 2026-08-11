"""Type aliases for the shapes that turn up throughout the public API.

Every function that takes competitor names or per-match numbers accepts a
numpy array just as readily as a list — the bundled dataset hands out object
arrays, and so does pandas — so the annotations say so rather than pretending
the inputs are always ``Sequence``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Union

import numpy as np

__all__ = ["Names", "Numbers", "Times", "MatrixLike"]

#: Competitor names: a list, a tuple, or a numpy array of strings/objects.
Names = Union[Sequence[str], np.ndarray]  # noqa: UP007 - runtime alias, not an annotation

#: Per-match numbers — outcomes, weights, margins.
Numbers = Union[Sequence[float], np.ndarray]  # noqa: UP007

#: Timestamps in any form :func:`bscores._time.as_days` understands: dates,
#: datetimes, ``datetime64`` arrays, ISO strings or plain numbers.
Times = Any

#: A square matrix: a dense ``ndarray`` or a SciPy sparse matrix.
MatrixLike = Any
