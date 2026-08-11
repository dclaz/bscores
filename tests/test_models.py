"""The model API, its openskill-shaped surface, and causality guarantees."""

from __future__ import annotations

import numpy as np
import pytest

from bscores import BScoreModel, Rating
from bscores.decay import Exponential, Hyperbolic, Uniform
from bscores.models import _team_names


def round_robin(teams, rounds=3, seed=0, upset_rate=0.0):
    """A synthetic competition with a known strength order.

    Team ``i`` beats team ``j`` whenever ``i < j``, except on an ``upset_rate``
    fraction of matches.  With no upsets the loss network is a perfect DAG,
    which is the degenerate case the centrality solver has to special-case; with
    upsets it develops cycles and behaves like a real competition.
    """
    rng = np.random.default_rng(seed)
    home, away, outcome, times = [], [], [], []
    day = 0.0
    for _ in range(rounds):
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                swap = rng.random() < 0.5
                a, b = (j, i) if swap else (i, j)
                stronger_at_home = a < b
                if rng.random() < upset_rate:
                    stronger_at_home = not stronger_at_home
                home.append(teams[a])
                away.append(teams[b])
                outcome.append(1.0 if stronger_at_home else 0.0)
                times.append(day)
                day += 1.0
    return home, away, np.array(outcome), np.array(times)


class TestTeamParsing:
    def test_accepts_names_lists_and_ratings(self):
        assert _team_names("a") == ["a"]
        assert _team_names(["a", "b"]) == ["a", "b"]
        assert _team_names(Rating("a", 0.1)) == ["a"]
        assert _team_names([Rating("a", 0.1), "b"]) == ["a", "b"]

    def test_rejects_nonsense(self):
        with pytest.raises(TypeError):
            _team_names(3)
        with pytest.raises(TypeError):
            _team_names([1, 2])
        with pytest.raises(ValueError, match="must not be empty"):
            _team_names([])


class TestRating:
    def test_ordinal_is_a_monotone_rescaling(self):
        assert Rating("a", 0.25).ordinal() == 250.0
        assert Rating("a", 0.25).ordinal(scale=4.0, target=1.0) == 2.0

    def test_repr_shows_rank_when_present(self):
        assert "rank=2" in repr(Rating("a", 0.1, rank=2))
        assert "rank" not in repr(Rating("a", 0.1))


class TestOpenskillShapedApi:
    def test_rate_returns_the_shape_it_was_given(self):
        model = BScoreModel()
        result = model.rate([["a", "b"], ["c", "d"]], at="2020-01-01")
        assert [[r.name for r in team] for team in result] == [["a", "b"], ["c", "d"]]
        assert all(isinstance(r, Rating) for team in result for r in team)

    def test_bare_names_are_one_person_teams(self):
        model = BScoreModel()
        result = model.rate(["a", "b"], at="2020-01-01")
        assert [[r.name for r in team] for team in result] == [["a"], ["b"]]

    def test_ratings_round_trip_into_rate(self):
        model = BScoreModel()
        first = model.rate(["a", "b"], at="2020-01-01")
        second = model.rate([first[0], first[1]], at="2020-01-08")
        assert second[0][0].name == "a"
        assert second[0][0].matches == 2

    def test_predict_win_sums_to_one(self):
        model = BScoreModel()
        model.rate(["a", "b"], at="2020-01-01")
        model.rate(["b", "c"], at="2020-01-08")
        model.rate(["c", "a"], at="2020-01-15")
        for teams in ([["a"], ["b"]], [["a"], ["b"], ["c"]]):
            assert sum(model.predict_win(teams)) == pytest.approx(1.0)

    def test_predict_win_is_uniform_with_no_history(self):
        assert BScoreModel().predict_win([["a"], ["b"]]) == [0.5, 0.5]

    def test_predict_win_needs_two_teams(self):
        with pytest.raises(ValueError, match="at least two teams"):
            BScoreModel().predict_win([["a"]])

    def test_predict_rank_is_dense_and_one_based(self):
        model = BScoreModel()
        model.rate(["a", "b"], at="2020-01-01")
        model.rate(["b", "c"], at="2020-01-08")
        model.rate(["c", "a"], at="2020-01-15")
        ranks = model.predict_rank([["a"], ["b"], ["c"]])
        assert sorted(r for r, _ in ranks) == [1, 2, 3]

    def test_predict_rank_ties_share_a_rank(self):
        # Equal times and no decay, so the two results cancel exactly.
        model = BScoreModel(kernel=Uniform())
        model.rate(["a", "b"], at="2020-01-01")
        model.rate(["b", "a"], at="2020-01-01")
        ranks = model.predict_rank([["a"], ["b"]])
        assert ranks[0][0] == ranks[1][0] == 1

    def test_multi_member_teams_aggregate_scores(self):
        model = BScoreModel()
        model.rate([["a", "b"], ["c", "d"]], at="2020-01-01")
        combined = model.predict_win([["a", "b"], ["c"]])
        assert combined[0] > combined[1]

    def test_rate_needs_two_teams(self):
        with pytest.raises(ValueError, match="at least two teams"):
            BScoreModel().rate([["a"]], at="2020-01-01")


class TestRatingsAndLeaderboard:
    def test_leaderboard_is_sorted_and_ranked(self):
        teams = [f"t{i}" for i in range(6)]
        home, away, outcome, times = round_robin(teams, rounds=3)
        model = BScoreModel(alpha=1e6).add_matches(home, away, outcome, times)
        board = model.leaderboard()
        assert [r.name for r in board] == teams
        assert [r.score for r in board] == sorted((r.score for r in board), reverse=True)
        assert model.ratings()[0].rank is not None

    def test_leaderboard_top_n(self):
        teams = [f"t{i}" for i in range(6)]
        model = BScoreModel(alpha=1e6).add_matches(*round_robin(teams, rounds=2))
        assert len(model.leaderboard(top=3)) == 3

    def test_unknown_competitor_scores_zero(self):
        model = BScoreModel()
        model.rate(["a", "b"], at="2020-01-01")
        assert model.rating("nobody").score == 0.0
        with pytest.raises(KeyError):
            model.rating("nobody", default=False)

    def test_counters_track_results(self):
        model = BScoreModel()
        model.rate(["a", "b"], at="2020-01-01")
        model.rate(["b", "a"], at="2020-01-08")
        model.rate_result("a", "b", at="2020-01-15", draw=True)
        rating = model.rating("a")
        assert rating.matches == 3
        assert rating.wins == pytest.approx(1.5)
        assert rating.losses == pytest.approx(1.5)

    def test_players_are_in_id_order(self):
        model = BScoreModel()
        model.rate(["z", "a"], at="2020-01-01")
        assert model.players == ["z", "a"]
        assert len(model) == 2
        assert "z" in model and "q" not in model


class TestNetworkSemantics:
    def test_every_match_moves_every_rating(self):
        """The paper's headline property (Sect. 2 and the conclusion)."""
        model = BScoreModel(alpha=365.0)
        model.rate(["a", "b"], at="2020-01-01")
        model.rate(["b", "c"], at="2020-01-08")
        model.rate(["c", "a"], at="2020-01-15")
        model.rate(["d", "a"], at="2020-01-22")
        before = model.rating("c").score

        # d and b play; c is nowhere near the match.
        model.rate(["d", "b"], at="2020-01-29")
        assert model.rating("c").score != pytest.approx(before, abs=1e-9)

    def test_beating_a_strong_opponent_is_worth_more(self):
        strong = BScoreModel(alpha=1e6)
        strong.add_matches(*round_robin([f"t{i}" for i in range(5)], rounds=4))
        board = strong.leaderboard()
        best, worst = board[0].name, board[-1].name

        challenger_beats_best = BScoreModel(alpha=1e6)
        challenger_beats_best.add_matches(*round_robin([f"t{i}" for i in range(5)], rounds=4))
        challenger_beats_best.rate_result("x", best, at=1e5)

        challenger_beats_worst = BScoreModel(alpha=1e6)
        challenger_beats_worst.add_matches(*round_robin([f"t{i}" for i in range(5)], rounds=4))
        challenger_beats_worst.rate_result("x", worst, at=1e5)

        assert (
            challenger_beats_best.rating("x").score
            > challenger_beats_worst.rating("x").score
        )

    def test_idle_competitors_decay_relative_to_active_ones(self):
        model = BScoreModel(alpha=100.0)
        model.add_matches(*round_robin(["a", "b", "c", "d"], rounds=2))
        early = model.rating("a").score

        # Everyone but "a" keeps playing for two more years.
        home, away, outcome, times = round_robin(["b", "c", "d"], rounds=8)
        model.add_matches(home, away, outcome, np.asarray(times) + 400.0)
        assert model.rating("a").score < early

    def test_scores_have_unit_euclidean_norm(self):
        model = BScoreModel(alpha=365.0)
        model.add_matches(*round_robin([f"t{i}" for i in range(8)], rounds=2))
        assert np.linalg.norm(model.scores()) == pytest.approx(1.0)


class TestIngestionEquivalence:
    def test_rate_many_matches_repeated_rate(self):
        one = BScoreModel(alpha=365.0)
        for winner, loser, day in [("a", "b", 0.0), ("b", "c", 7.0), ("c", "a", 14.0)]:
            one.rate_result(winner, loser, at=day)

        bulk = BScoreModel(alpha=365.0)
        bulk.rate_many(["a", "b", "c"], ["b", "c", "a"], [0.0, 7.0, 14.0])

        assert one.players == bulk.players
        np.testing.assert_allclose(one.scores(), bulk.scores(), atol=1e-10)

    def test_add_matches_matches_rate_many_when_home_always_wins(self):
        home = ["a", "b", "c"]
        away = ["b", "c", "a"]
        times = [0.0, 7.0, 14.0]
        fixtures = BScoreModel(alpha=365.0).add_matches(home, away, [1.0, 1.0, 1.0], times)
        results = BScoreModel(alpha=365.0).rate_many(home, away, times)
        np.testing.assert_allclose(fixtures.scores(), results.scores(), atol=1e-10)

    def test_away_wins_reverse_the_arc(self):
        model = BScoreModel(alpha=1e9)
        model.add_matches(["a"], ["b"], [0.0], [0.0])
        assert model.rating("b").score > model.rating("a").score

    def test_draws_transfer_both_ways(self):
        model = BScoreModel(alpha=1e9, draw_weight=0.5)
        model.add_matches(["a"], ["b"], [0.5], [0.0])
        w = model.network.matrix()
        assert w[0, 1] == pytest.approx(0.5)
        assert w[1, 0] == pytest.approx(0.5)

    def test_draw_weight_zero_ignores_draws(self):
        model = BScoreModel(draw_weight=0.0)
        model.add_matches(["a", "a"], ["b", "b"], [0.5, 1.0], [0.0, 1.0])
        assert model.network.n_events == 1

    def test_negative_draw_weight_rejected(self):
        with pytest.raises(ValueError, match="draw_weight"):
            BScoreModel(draw_weight=-1.0)

    def test_weights_scale_a_result(self):
        light = BScoreModel(alpha=1e9).add_matches(["a"], ["b"], [1.0], [0.0], weights=[1.0])
        heavy = BScoreModel(alpha=1e9).add_matches(["a"], ["b"], [1.0], [0.0], weights=[5.0])
        assert heavy.network.matrix()[1, 0] == pytest.approx(5.0)
        assert light.network.matrix()[1, 0] == pytest.approx(1.0)

    def test_missing_times_advance_an_internal_clock(self):
        model = BScoreModel(kernel=Uniform())
        model.rate(["a", "b"])
        model.rate(["b", "c"])
        np.testing.assert_allclose(model.network.times, [1.0, 2.0])

    def test_rate_many_without_times_advances_the_clock(self):
        model = BScoreModel(kernel=Uniform())
        model.rate_many(["a", "b"], ["b", "c"])
        np.testing.assert_allclose(model.network.times, [1.0, 2.0])

    def test_rate_many_broadcasts_a_single_time(self):
        model = BScoreModel()
        model.rate_many(["a", "b"], ["b", "c"], "2020-01-01")
        assert len(set(model.network.times.tolist())) == 1

    def test_ingestion_length_checks(self):
        model = BScoreModel()
        with pytest.raises(ValueError, match="winners/losers"):
            model.rate_many(["a"], ["b", "c"])
        with pytest.raises(ValueError, match="home/away"):
            model.add_matches(["a"], ["b", "c"], [1.0], [0.0])
        with pytest.raises(ValueError, match="outcome length"):
            model.add_matches(["a"], ["b"], [1.0, 0.0], [0.0])
        with pytest.raises(ValueError, match="times length"):
            model.add_matches(["a"], ["b"], [1.0], [0.0, 1.0])
        with pytest.raises(ValueError, match="weights length"):
            model.add_matches(["a"], ["b"], [1.0], [0.0], weights=[1.0, 2.0])
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            model.add_matches(["a"], ["b"], [2.0], [0.0])

    def test_ranks_argument_controls_the_winner(self):
        model = BScoreModel(alpha=1e9)
        model.rate([["a"], ["b"]], ranks=[2, 1], at=0.0)
        assert model.rating("b").score > model.rating("a").score

    def test_ranks_length_checked(self):
        with pytest.raises(ValueError, match="ranks"):
            BScoreModel().rate([["a"], ["b"]], ranks=[1], at=0.0)

    def test_multi_team_free_for_all_generates_pairwise_arcs(self):
        model = BScoreModel(alpha=1e9)
        model.rate([["a"], ["b"], ["c"]], at=0.0)
        w = model.network.matrix()
        assert w[1, 0] == 1.0  # b lost to a
        assert w[2, 0] == 1.0  # c lost to a
        assert w[2, 1] == 1.0  # c lost to b
        assert w[0, 1] == 0.0

    def test_empty_ingestion_is_a_no_op(self):
        model = BScoreModel()
        model.rate_many([], [])
        model.add_matches([], [], [], [])
        assert model.network.n_events == 0


class TestCausality:
    def test_score_history_excludes_simultaneous_results(self):
        model = BScoreModel(alpha=1e9)
        model.add_matches(["a", "b"], ["b", "c"], [1.0, 1.0], [0.0, 10.0])
        history = model.score_history([0.0, 10.0], inclusive=False)
        np.testing.assert_allclose(history.scores[0], 0.0)
        assert history.scores[1][model.index.get("a")] > 0.0

    def test_inclusive_history_sees_the_current_epoch(self):
        model = BScoreModel(alpha=1e9)
        model.add_matches(["a"], ["b"], [1.0], [0.0])
        history = model.score_history([0.0], inclusive=True)
        assert history.scores[0][model.index.get("a")] > 0.0

    def test_match_scores_are_pre_match(self):
        model = BScoreModel(alpha=1e9)
        home = ["a", "b", "c"]
        away = ["b", "c", "a"]
        times = [0.0, 10.0, 20.0]
        model.add_matches(home, away, [1.0, 1.0, 1.0], times)
        home_scores, away_scores = model.match_scores(home, away, times)
        # Nobody has played before the first match.
        assert home_scores[0] == 0.0 and away_scores[0] == 0.0
        # Before the second match only "a beat b" has happened, so "a" carries
        # the only positive score and "b" (the home side) still has none.
        assert home_scores[1] == 0.0 and away_scores[1] == 0.0
        # By the third match "a" is rated and appears as the away side.
        assert away_scores[2] > 0.0

    def test_future_results_cannot_change_past_scores(self):
        home, away, outcome, times = round_robin([f"t{i}" for i in range(6)], rounds=4, seed=1)
        cut = len(times) // 2

        full = BScoreModel(alpha=200.0)
        full.add_matches(home, away, outcome, times)
        original = full.match_scores(home, away, times)

        tampered_outcome = outcome.copy()
        tampered_outcome[cut:] = 1.0 - tampered_outcome[cut:]
        tampered = BScoreModel(alpha=200.0)
        tampered.add_matches(home, away, tampered_outcome, times)
        changed = tampered.match_scores(home, away, times)

        np.testing.assert_allclose(original[0][:cut], changed[0][:cut], atol=1e-12)
        np.testing.assert_allclose(original[1][:cut], changed[1][:cut], atol=1e-12)

    def test_match_scores_length_checks(self):
        model = BScoreModel()
        with pytest.raises(ValueError, match="home/away"):
            model.match_scores(["a"], ["b", "c"], [0.0])
        with pytest.raises(ValueError, match="times length"):
            model.match_scores(["a"], ["b"], [0.0, 1.0])

    def test_unknown_competitors_score_zero_in_match_scores(self):
        model = BScoreModel(alpha=1e9)
        model.add_matches(["a"], ["b"], [1.0], [0.0])
        home_scores, away_scores = model.match_scores(["a"], ["ghost"], [10.0])
        assert away_scores[0] == 0.0


class TestRatingHistory:
    def test_series_lookup(self):
        model = BScoreModel(alpha=1e9)
        model.add_matches(["a", "a"], ["b", "b"], [1.0, 1.0], [0.0, 10.0])
        history = model.score_history([0.0, 10.0, 20.0])
        assert history.of("a").shape == (3,)
        with pytest.raises(KeyError):
            history.of("nobody")

    def test_at_returns_the_prevailing_scores(self):
        model = BScoreModel(alpha=1e9)
        model.add_matches(["a"], ["b"], [1.0], [0.0])
        history = model.score_history([0.0, 10.0])
        np.testing.assert_allclose(history.at(-5.0), 0.0)
        np.testing.assert_allclose(history.at(15.0), history.scores[1])

    def test_columns_subsets_the_output(self):
        model = BScoreModel(alpha=1e9)
        model.add_matches(["a", "b"], ["b", "c"], [1.0, 1.0], [0.0, 10.0])
        history = model.score_history([10.0], columns=[model.index.get("a")])
        assert history.scores.shape == (1, 1)
        assert history.names == ["a"]

    def test_empty_model_history(self):
        history = BScoreModel().score_history([0.0, 1.0])
        assert history.scores.shape == (2, 0)

    def test_repr(self):
        model = BScoreModel(alpha=1e9)
        model.add_matches(["a"], ["b"], [1.0], [0.0])
        assert "epochs" in repr(model.score_history([0.0, 1.0]))


class TestCalibratedPrediction:
    def test_fit_then_predict(self):
        teams = [f"t{i}" for i in range(8)]
        home, away, outcome, times = round_robin(teams, rounds=6, seed=2, upset_rate=0.25)
        model = BScoreModel(alpha=400.0)
        model.fit(home, away, outcome, times)
        assert model.is_calibrated
        probabilities = model.predict_proba(home, away, times)
        assert probabilities.shape == (len(home),)
        assert np.all((probabilities > 0.0) & (probabilities < 1.0))
        # The strongest team should be favoured against the weakest.
        assert model.predict_win([[teams[0]], [teams[-1]]])[0] > 0.5

    def test_uncalibrated_predict_proba_uses_score_shares(self):
        model = BScoreModel(alpha=1e9)
        home, away = ["a", "b"], ["b", "a"]
        times = [0.0, 10.0]
        model.add_matches(home, away, [1.0, 0.0], times)
        probabilities = model.predict_proba(home, away, times)
        assert probabilities[0] == 0.5  # nothing known before the first match

    def test_fit_without_ingesting(self):
        teams = [f"t{i}" for i in range(6)]
        home, away, outcome, times = round_robin(teams, rounds=4, seed=3)
        model = BScoreModel(alpha=400.0)
        model.add_matches(home, away, outcome, times)
        model.fit(home, away, outcome, times, ingest=False)
        assert model.network.n_events == len(home)

    def test_predict_win_uses_the_calibrator_for_two_teams(self):
        teams = [f"t{i}" for i in range(6)]
        home, away, outcome, times = round_robin(teams, rounds=5, seed=4, upset_rate=0.25)
        model = BScoreModel(alpha=400.0).fit(home, away, outcome, times)
        favourite, underdog = teams[0], teams[-1]
        probability = model.predict_win([[favourite], [underdog]])[0]
        expected = model.calibrator.predict_proba(
            [model.rating(favourite).score], [model.rating(underdog).score]
        )[0]
        assert probability == pytest.approx(expected)

    def test_draw_model_is_opt_in(self):
        teams = [f"t{i}" for i in range(6)]
        home, away, outcome, times = round_robin(teams, rounds=5, seed=5, upset_rate=0.25)
        outcome = outcome.copy()
        outcome[::7] = 0.5
        model = BScoreModel(alpha=400.0)
        model.fit(home, away, outcome, times, model_draws=False)
        assert model.predict_draw([["t0"], ["t1"]]) == 0.0

        model.fit(home, away, outcome, times, ingest=False, model_draws=True)
        drawn = model.predict_draw([["t0"], ["t1"]])
        assert 0.0 < drawn < 0.5

    def test_predict_draw_requires_two_teams(self):
        teams = [f"t{i}" for i in range(4)]
        home, away, outcome, times = round_robin(teams, rounds=4, seed=6, upset_rate=0.25)
        outcome = outcome.copy()
        outcome[::5] = 0.5
        model = BScoreModel(alpha=400.0).fit(home, away, outcome, times, model_draws=True)
        with pytest.raises(ValueError, match="exactly two teams"):
            model.predict_draw([["t0"], ["t1"], ["t2"]])


class TestConfiguration:
    def test_alpha_reports_the_half_life(self):
        assert BScoreModel(alpha=30.0).alpha == 30.0
        assert BScoreModel(kernel=Exponential(45.0)).alpha == 45.0

    def test_kernel_override_wins(self):
        model = BScoreModel(alpha=30.0, kernel=Hyperbolic(90.0))
        assert model.kernel == Hyperbolic(90.0)

    def test_warm_start_does_not_change_the_answer(self):
        teams = [f"t{i}" for i in range(9)]
        args = round_robin(teams, rounds=3, seed=7)
        warm = BScoreModel(alpha=200.0, warm_start=True).add_matches(*args)
        cold = BScoreModel(alpha=200.0, warm_start=False).add_matches(*args)
        np.testing.assert_allclose(warm.scores(), cold.scores(), atol=1e-9)
        np.testing.assert_allclose(
            warm.score_history(args[3]).scores,
            cold.score_history(args[3]).scores,
            atol=1e-9,
        )

    def test_solver_choices_agree(self):
        teams = [f"t{i}" for i in range(7)]
        args = round_robin(teams, rounds=3, seed=8)
        power = BScoreModel(alpha=200.0, solver="power").add_matches(*args)
        dense = BScoreModel(alpha=200.0, solver="dense").add_matches(*args)
        np.testing.assert_allclose(power.scores(), dense.scores(), atol=1e-7)

    def test_sparse_and_dense_agree(self):
        pytest.importorskip("scipy.sparse")
        teams = [f"t{i}" for i in range(10)]
        args = round_robin(teams, rounds=3, seed=9)
        dense = BScoreModel(alpha=200.0, sparse=False).add_matches(*args)
        sparse = BScoreModel(alpha=200.0, sparse=True).add_matches(*args)
        np.testing.assert_allclose(dense.scores(), sparse.scores(), atol=1e-8)

    def test_max_age_ignores_ancient_results(self):
        model = BScoreModel(kernel=Uniform(), max_age=50.0)
        model.add_matches(["a", "c"], ["b", "d"], [1.0, 1.0], [0.0, 100.0])
        assert model.network.matrix(100.0).sum() == pytest.approx(1.0)

    def test_repr_reports_state(self):
        model = BScoreModel(alpha=365.0)
        assert "uncalibrated" in repr(model)
        teams = [f"t{i}" for i in range(4)]
        model.fit(*round_robin(teams, rounds=4, seed=10, upset_rate=0.25))
        assert "calibrated" in repr(model)


class TestCaching:
    def test_repeated_queries_are_consistent(self):
        model = BScoreModel(alpha=365.0)
        model.add_matches(*round_robin(["a", "b", "c", "d"], rounds=2))
        first = model.scores().copy()
        np.testing.assert_array_equal(first, model.scores())

    def test_new_results_invalidate_the_cache(self):
        model = BScoreModel(alpha=365.0)
        model.add_matches(*round_robin(["a", "b", "c"], rounds=2))
        before = model.scores().copy()
        model.rate_result("c", "a", at=1000.0)
        assert not np.allclose(before, model.scores()[: before.size])

    def test_scores_of_an_empty_model(self):
        assert BScoreModel().scores().shape == (0,)
