"""Time handling.

The B-score memory parameter ``alpha`` is expressed in *days* (the paper uses
``alpha = 365``), so every timestamp the package touches is normalised to a
float number of days since the Unix epoch.  Accepting ``datetime``,
``numpy.datetime64``, ISO strings and plain floats keeps call sites tidy
without forcing a pandas dependency on the core.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["as_days", "as_day", "MICROSECONDS_PER_DAY"]

MICROSECONDS_PER_DAY = 86_400_000_000.0


def as_days(values: Any) -> np.ndarray:
    """Convert timestamps to a float64 array of days since the epoch.

    Numeric input is passed through unchanged, which lets callers use an
    arbitrary integer match counter as the clock when calendar dates are not
    available.

    Parameters
    ----------
    values
        Scalar or array-like of ``datetime``/``date``/``numpy.datetime64``/ISO
        strings/numbers.

    Returns
    -------
    numpy.ndarray
        float64 array (0-d for scalar input).
    """
    arr = np.asarray(values)

    if arr.dtype.kind == "M":
        return arr.astype("datetime64[us]").astype("int64") / MICROSECONDS_PER_DAY
    if arr.dtype.kind in "iuf":
        return arr.astype(np.float64)
    if arr.dtype.kind == "b":
        raise TypeError("booleans are not valid timestamps")

    # Objects and strings: let numpy parse them as datetimes.  Anything it
    # cannot parse raises here with a message naming the offending value.
    try:
        return np.asarray(values, dtype="datetime64[us]").astype("int64") / MICROSECONDS_PER_DAY
    except (ValueError, TypeError) as exc:  # pragma: no cover - message passthrough
        raise TypeError(f"cannot interpret {values!r} as a timestamp") from exc


def as_day(value: Any) -> float:
    """Convert a single timestamp to a float number of days since the epoch."""
    days = as_days(value)
    if days.ndim != 0:
        raise ValueError(f"expected a scalar timestamp, got shape {days.shape}")
    return float(days)
