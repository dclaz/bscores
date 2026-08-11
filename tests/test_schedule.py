"""Recovering seasons and rounds from a bare fixture list."""

from __future__ import annotations

import numpy as np
import pytest

from bscores.datasets import load_afl
from bscores.schedule import infer_rounds, infer_seasons, round_labels, round_positions


class TestInferSeasons:
    def test_splits_on_the_off_season(self):
        times = [0.0, 7.0, 14.0, 400.0, 407.0]
        np.testing.assert_array_equal(infer_seasons(times), [1, 1, 1, 2, 2])

    def test_mid_season_breaks_do_not_split(self):
        # A three-week gap is a bye round, not an off-season.
        np.testing.assert_array_equal(infer_seasons([0.0, 21.0, 28.0]), [1, 1, 1])

    def test_gap_is_configurable(self):
        np.testing.assert_array_equal(infer_seasons([0.0, 30.0], gap=10.0), [1, 2])

    def test_unsorted_input_is_labelled_correctly(self):
        np.testing.assert_array_equal(infer_seasons([400.0, 0.0, 407.0, 7.0]), [2, 1, 2, 1])

    def test_empty(self):
        assert infer_seasons([]).size == 0

    def test_gap_must_be_positive(self):
        with pytest.raises(ValueError, match="gap"):
            infer_seasons([0.0], gap=0.0)


class TestInferRounds:
    def test_a_round_ends_when_a_competitor_repeats(self):
        home = ["a", "c", "a", "c"]
        away = ["b", "d", "d", "b"]
        times = [0.0, 0.0, 7.0, 7.0]
        np.testing.assert_array_equal(infer_rounds(home, away, times), [1, 1, 2, 2])

    def test_byes_leave_a_short_round(self):
        # Six teams, but only two matches in round 1.
        home = ["a", "c", "a", "c", "e"]
        away = ["b", "d", "d", "b", "f"]
        times = [0.0, 0.0, 7.0, 7.0, 7.0]
        np.testing.assert_array_equal(infer_rounds(home, away, times), [1, 1, 2, 2, 2])

    def test_numbering_restarts_each_season(self):
        home = ["a", "a", "a"]
        away = ["b", "b", "b"]
        times = [0.0, 7.0, 400.0]
        np.testing.assert_array_equal(infer_rounds(home, away, times), [1, 2, 1])

    def test_explicit_seasons_are_respected(self):
        home, away, times = ["a", "a"], ["b", "b"], [0.0, 7.0]
        np.testing.assert_array_equal(
            infer_rounds(home, away, times, season=[10, 11]), [1, 1]
        )

    def test_rounds_never_decrease_within_a_season(self):
        rng = np.random.default_rng(0)
        teams = [f"t{i}" for i in range(10)]
        home, away, times = [], [], []
        for day in range(0, 200, 7):
            order = rng.permutation(teams)
            for i in range(0, 10, 2):
                home.append(order[i])
                away.append(order[i + 1])
                times.append(float(day))
        rounds = infer_rounds(home, away, times)
        order = np.argsort(times, kind="stable")
        assert np.all(np.diff(rounds[order]) >= 0)

    def test_row_order_does_not_change_the_answer(self):
        # Same-day matches are the norm at day resolution; the greedy boundary
        # must not depend on which row happened to come first.
        rng = np.random.default_rng(0)
        teams = [f"t{i}" for i in range(12)]
        home, away, times = [], [], []
        for day in range(0, 140, 7):
            order = rng.permutation(teams)
            for i in range(0, 12, 2):
                home.append(order[i])
                away.append(order[i + 1])
                times.append(float(day))
        home, away, times = np.array(home), np.array(away), np.array(times)
        base = infer_rounds(home, away, times)
        for _ in range(10):
            p = rng.permutation(times.size)
            inverse = np.empty_like(p)
            inverse[p] = np.arange(p.size)
            shuffled = infer_rounds(home[p], away[p], times[p])
            np.testing.assert_array_equal(shuffled[inverse], base)

    def test_length_mismatch_rejected(self):
        with pytest.raises(ValueError, match="same length"):
            infer_rounds(["a"], ["b", "c"], [0.0])

    def test_season_length_mismatch_rejected(self):
        with pytest.raises(ValueError, match="season length mismatch"):
            infer_rounds(["a"], ["b"], [0.0], season=[1, 2])

    def test_empty(self):
        assert infer_rounds([], [], []).size == 0


class TestRoundPositions:
    def test_consecutive_and_gapless_across_seasons(self):
        positions = round_positions([1, 1, 2, 2], [1, 2, 1, 2])
        np.testing.assert_array_equal(positions, [0, 1, 2, 3])

    def test_matches_in_the_same_round_share_a_position(self):
        positions = round_positions([1, 1, 1], [1, 1, 2])
        np.testing.assert_array_equal(positions, [0, 0, 1])

    def test_increasing_in_season_then_round(self):
        positions = round_positions([2, 1], [1, 5])
        assert positions[1] < positions[0]

    def test_length_mismatch_rejected(self):
        with pytest.raises(ValueError, match="length mismatch"):
            round_positions([1, 1], [1])

    def test_empty(self):
        assert round_positions([], []).size == 0


class TestRoundLabels:
    def test_plain_rounds(self):
        np.testing.assert_array_equal(round_labels([2023, 2023], [1, 2]), ["2023 R1", "2023 R2"])

    def test_finals_are_numbered_from_the_first_final(self):
        labels = round_labels(
            [2023] * 4, [23, 24, 25, 26], final=[False, True, True, True]
        )
        np.testing.assert_array_equal(labels, ["2023 R23", "2023 F1", "2023 F2", "2023 F3"])

    def test_length_mismatch_rejected(self):
        with pytest.raises(ValueError, match="length mismatch"):
            round_labels([2023], [1, 2])
        with pytest.raises(ValueError, match="final length mismatch"):
            round_labels([2023], [1], final=[True, False])


class TestAgainstTheAflArchive:
    @pytest.fixture(scope="class")
    @classmethod
    def afl(cls):
        return load_afl(as_frame=False)

    def test_seasons_are_calendar_years(self, afl):
        years = afl.date.astype("datetime64[Y]").astype(int) + 1970
        np.testing.assert_array_equal(afl.season, years)

    def test_a_full_season_runs_to_about_27_rounds(self, afl):
        # 23 home-and-away rounds plus four finals weeks.
        for season in (2017, 2018, 2019, 2021, 2022):
            assert 26 <= int(afl.round[afl.season == season].max()) <= 28

    def test_the_covid_season_is_shorter(self, afl):
        assert int(afl.round[afl.season == 2020].max()) < 26

    def test_every_season_ends_in_finals(self, afl):
        for season in (2019, 2021, 2023, 2025):
            block = afl.season == season
            assert afl.final[block][np.argmax(afl.round[block])]

    def test_no_competitor_plays_twice_in_a_round(self, afl):
        for season in np.unique(afl.season):
            block = afl.season == season
            for number in np.unique(afl.round[block]):
                inside = block & (afl.round == number)
                teams = np.concatenate([afl.home_team[inside], afl.away_team[inside]])
                assert len(set(teams.tolist())) == teams.size

    def test_the_stored_column_is_reproducible(self, afl):
        recomputed = infer_rounds(
            afl.home_team, afl.away_team, afl.date, season=afl.season
        )
        np.testing.assert_array_equal(recomputed, afl.round)

    def test_round_labels_read_naturally(self, afl):
        labels = set(afl.round_label[afl.season == 2023].tolist())
        assert "2023 R1" in labels
        assert "2023 F1" in labels
