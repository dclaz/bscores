"""The command line interface."""

from __future__ import annotations

import csv
import json

import pytest

from bscores import __version__
from bscores.cli import build_parser, main
from bscores.datasets import load_afl


@pytest.fixture(scope="module")
def fixture_csv(tmp_path_factory):
    """A small CSV in the default column layout."""
    afl = load_afl(as_frame=False)
    keep = afl.season >= 2024
    path = tmp_path_factory.mktemp("data") / "fixtures.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["home_team", "away_team", "outcome", "date"])
        for i in range(len(afl)):
            if keep[i]:
                writer.writerow(
                    [afl.home_team[i], afl.away_team[i], afl.outcome[i], afl.date[i]]
                )
    return path


def run(capsys, *argv):
    """Invoke the CLI and return (exit code, stdout)."""
    code = main(list(argv))
    return code, capsys.readouterr().out


class TestParser:
    def test_every_subcommand_is_registered(self):
        parser = build_parser()
        actions = [a for a in parser._actions if a.dest == "command"]
        assert set(actions[0].choices) == {"info", "rate", "forecast", "tune", "simulate"}

    def test_a_subcommand_is_required(self):
        with pytest.raises(SystemExit):
            main([])

    def test_version_flag(self, capsys):
        with pytest.raises(SystemExit) as exit_info:
            main(["--version"])
        assert exit_info.value.code == 0
        assert __version__ in capsys.readouterr().out


class TestInfo:
    def test_text_output(self, capsys):
        code, out = run(capsys, "info")
        assert code == 0
        assert __version__ in out
        assert "bundled dataset" in out

    def test_json_output(self, capsys):
        code, out = run(capsys, "info", "--format", "json")
        payload = json.loads(out)
        assert code == 0
        assert payload["version"] == __version__
        assert payload["bundled_dataset"]["matches"] == 3533


class TestRate:
    def test_bundled_data(self, capsys):
        code, out = run(capsys, "rate", "--data", "afl", "--top", "3")
        assert code == 0
        assert len(out.strip().splitlines()) == 4  # header plus three rows

    def test_json_carries_ratings_and_network(self, capsys):
        code, out = run(capsys, "rate", "--data", "afl", "--top", "5", "--format", "json")
        payload = json.loads(out)
        assert code == 0
        assert len(payload["ratings"]) == 5
        assert payload["ratings"][0]["rank"] == 1
        assert payload["network"]["competitors"] == 18

    def test_kernel_and_alpha_change_the_answer(self, capsys):
        _, slow = run(capsys, "rate", "--data", "afl", "--alpha", "2000", "--format", "json")
        _, fast = run(capsys, "rate", "--data", "afl", "--alpha", "30", "--format", "json")
        assert json.loads(slow)["ratings"] != json.loads(fast)["ratings"]

    def test_reads_a_csv(self, capsys, fixture_csv):
        code, out = run(capsys, "rate", "--data", str(fixture_csv), "--format", "json")
        payload = json.loads(out)
        assert code == 0
        assert payload["source"] == str(fixture_csv)
        assert payload["network"]["competitors"] == 18

    def test_custom_column_names(self, capsys, tmp_path):
        path = tmp_path / "odd.csv"
        path.write_text("h,a,y,when\nx,y,1,2020-01-01\ny,x,0,2020-01-08\n", encoding="utf-8")
        code, out = run(
            capsys, "rate", "--data", str(path),
            "--home-col", "h", "--away-col", "a", "--outcome-col", "y", "--date-col", "when",
            "--format", "json",
        )
        assert code == 0
        assert {r["name"] for r in json.loads(out)["ratings"]} == {"x", "y"}

    def test_missing_file_is_reported(self):
        with pytest.raises(SystemExit, match="no such file"):
            main(["rate", "--data", "nope.csv"])

    def test_missing_columns_are_reported(self, tmp_path):
        path = tmp_path / "bad.csv"
        path.write_text("a,b\n1,2\n", encoding="utf-8")
        with pytest.raises(SystemExit, match="missing column"):
            main(["rate", "--data", str(path)])

    def test_empty_file_is_reported(self, tmp_path):
        path = tmp_path / "empty.csv"
        path.write_text("home_team,away_team,outcome,date\n", encoding="utf-8")
        with pytest.raises(SystemExit, match="no rows"):
            main(["rate", "--data", str(path)])


class TestForecast:
    def test_reports_the_metrics(self, capsys):
        code, out = run(
            capsys, "forecast", "--data", "afl",
            "--alpha", "120", "--kernel", "exponential", "--initial-train", "2023-01-01",
        )
        assert code == 0
        assert "log-loss" in out and "accuracy" in out

    def test_json_carries_coefficients(self, capsys):
        code, out = run(
            capsys, "forecast", "--data", "afl",
            "--initial-train", "2023-01-01", "--format", "json",
        )
        payload = json.loads(out)
        assert code == 0
        assert payload["forecast"] == 828
        assert len(payload["coefficients"][0]) == 3
        assert 0.0 < payload["metrics"]["log_loss"] < 1.0

    def test_fraction_split(self, capsys):
        code, out = run(
            capsys, "forecast", "--data", "afl", "--initial-train", "0.9", "--format", "json"
        )
        assert code == 0
        assert json.loads(out)["forecast"] == pytest.approx(3533 * 0.1, rel=0.02)


class TestTune:
    def test_searches_and_reports_the_best(self, capsys):
        code, out = run(
            capsys, "tune", "--data", "afl",
            "--alpha-grid", "60", "365", "--kernel-grid", "exponential",
            "--validation-start", "2023-01-01", "--top", "2", "--format", "json",
        )
        payload = json.loads(out)
        assert code == 0
        assert payload["configurations"] == 2
        assert len(payload["top"]) == 2
        assert payload["best"]["log_loss"] <= payload["top"][-1]["log_loss"]

    def test_text_output_names_the_winner(self, capsys):
        code, out = run(
            capsys, "tune", "--data", "afl", "--alpha-grid", "120",
            "--kernel-grid", "exponential", "--validation-start", "2023-01-01",
        )
        assert code == 0
        assert "best log_loss" in out

    def test_metric_is_selectable(self, capsys):
        code, out = run(
            capsys, "tune", "--data", "afl", "--alpha-grid", "60", "120",
            "--kernel-grid", "exponential", "--validation-start", "2023-01-01",
            "--metric", "accuracy", "--format", "json",
        )
        payload = json.loads(out)
        assert code == 0
        assert payload["metric"] == "accuracy"
        assert payload["best"]["accuracy"] >= payload["top"][-1]["accuracy"]


class TestSimulate:
    def test_probabilities_sum_correctly(self, capsys):
        code, out = run(
            capsys, "simulate", "--data", "afl", "--alpha", "120", "--kernel", "exponential",
            "--simulations", "500", "--seed", "0", "--format", "json",
        )
        payload = json.loads(out)
        assert code == 0
        assert len(payload["standings"]) == 18
        assert sum(row["p_first"] for row in payload["standings"]) == pytest.approx(1.0)
        assert sum(row["p_top4"] for row in payload["standings"]) == pytest.approx(4.0)

    def test_text_output(self, capsys):
        code, out = run(
            capsys, "simulate", "--data", "afl", "--simulations", "200", "--seed", "0"
        )
        assert code == 0
        assert "simulated seasons" in out
        assert "exp. points" in out
