"""Timestamp normalisation."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from bscores._time import as_day, as_days


def test_iso_string_matches_datetime():
    assert as_day("2020-01-01") == as_day(dt.date(2020, 1, 1))
    assert as_day("1970-01-01") == 0.0


def test_one_day_is_one_unit():
    assert as_day("2020-01-02") - as_day("2020-01-01") == pytest.approx(1.0)


def test_sub_day_resolution_survives():
    noon = as_day("2020-01-01T12:00:00")
    midnight = as_day("2020-01-01T00:00:00")
    assert noon - midnight == pytest.approx(0.5)


def test_numbers_pass_through():
    assert as_day(7.5) == 7.5
    np.testing.assert_allclose(as_days([1, 2, 3]), [1.0, 2.0, 3.0])


def test_datetime64_array():
    stamps = np.array(["2020-01-01", "2020-01-11"], dtype="datetime64[D]")
    np.testing.assert_allclose(np.diff(as_days(stamps)), [10.0])


def test_object_array_of_dates():
    values = [dt.date(2020, 1, 1), dt.datetime(2020, 1, 2, 12, 0, 0)]
    np.testing.assert_allclose(np.diff(as_days(values)), [1.5])


def test_dates_outside_the_nanosecond_range():
    # datetime64[ns] tops out in 2262; microsecond resolution must not.
    assert as_day("2400-01-01") > as_day("2200-01-01")


def test_booleans_rejected():
    with pytest.raises(TypeError):
        as_days(np.array([True, False]))


def test_unparseable_rejected():
    with pytest.raises(TypeError):
        as_days(["not-a-date"])


def test_as_day_requires_a_scalar():
    with pytest.raises(ValueError, match="scalar"):
        as_day([1.0, 2.0])
