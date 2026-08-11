"""The packaged AFL dataset."""

from __future__ import annotations

import numpy as np
import pytest

from bscores.datasets import MatchData, afl_data_path, load_afl

EXPECTED_MATCHES = 3533
EXPECTED_TEAMS = 18
EXPECTED_COLUMNS = [
    "date",
    "date_time",
    "home_team",
    "away_team",
    "venue",
    "home_score",
    "away_score",
    "margin",
    "outcome",
    "final",
    "home_odds",
    "away_odds",
]


@pytest.fixture(scope="module")
def matches() -> MatchData:
    return load_afl(as_frame=False)


class TestNumpyLoader:
    def test_size(self, matches):
        assert len(matches) == EXPECTED_MATCHES
        assert len(matches.teams) == EXPECTED_TEAMS

    def test_date_range(self, matches):
        assert str(matches.date.min()) == "2009-06-19"
        assert str(matches.date.max()) == "2026-08-02"

    def test_chronological(self, matches):
        assert np.all(np.diff(matches.date_time.astype("int64")) >= 0)

    def test_outcome_encodes_the_margin(self, matches):
        np.testing.assert_allclose(
            matches.outcome, np.clip(matches.margin + 0.5, 0.0, 1.0)
        )
        assert set(np.unique(matches.outcome).tolist()) == {0.0, 0.5, 1.0}

    def test_margin_is_home_minus_away(self, matches):
        np.testing.assert_array_equal(
            matches.margin, matches.home_score - matches.away_score
        )

    def test_draws_are_scored_as_a_half(self, matches):
        assert int((matches.outcome == 0.5).sum()) == 30

    def test_home_advantage_is_visible(self, matches):
        assert 0.5 < matches.outcome.mean() < 0.62

    def test_nobody_plays_themselves(self, matches):
        assert not np.any(matches.home_team == matches.away_team)

    def test_finals_are_flagged(self, matches):
        assert int(matches.final.sum()) == 145

    def test_odds_are_present_and_plausible(self, matches):
        # One 2012 fixture is quoted at a flat 1.00; everything else clears it.
        assert np.all(matches.home_odds >= 1.0)
        assert np.all(matches.away_odds >= 1.0)
        assert not np.any(np.isnan(matches.home_odds))
        # The bookmaker's book is over-round: implied probabilities exceed 1.
        implied = 1.0 / matches.home_odds + 1.0 / matches.away_odds
        assert implied.mean() > 1.0

    def test_repr(self, matches):
        assert "3533 matches" in repr(matches)


class TestPandasLoader:
    def test_columns_match_the_r_dataset(self):
        pd = pytest.importorskip("pandas")
        frame = load_afl()
        assert isinstance(frame, pd.DataFrame)
        assert list(frame.columns) == EXPECTED_COLUMNS
        assert len(frame) == EXPECTED_MATCHES

    def test_dates_are_parsed(self):
        pytest.importorskip("pandas")
        frame = load_afl()
        assert frame["date"].dtype.kind == "M"
        assert frame["date_time"].dtype.kind == "M"

    def test_agrees_with_the_numpy_loader(self, matches):
        pytest.importorskip("pandas")
        frame = load_afl()
        np.testing.assert_array_equal(frame["home_team"].to_numpy(), matches.home_team)
        np.testing.assert_allclose(frame["outcome"].to_numpy(), matches.outcome)
        np.testing.assert_array_equal(frame["margin"].to_numpy(), matches.margin)


def test_data_file_is_packaged():
    assert afl_data_path().name == "afl_matches.csv"
    assert afl_data_path().is_file()
