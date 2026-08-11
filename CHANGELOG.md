# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `bscores.schedule` recovers seasons and rounds from a bare fixture list.
  A season is a run of matches separated from the next by the off-season; a
  round is a maximal run in which no competitor plays twice, which recovers
  byes, split rounds and finals weeks without a fixture template.
- `season` and `round` columns on the bundled AFL dataset, plus
  `MatchData.round_label` for `"2023 R5"` / `"2023 F2"` strings.
- `plot_ratings(..., x="round", schedule=matches)` draws ratings against
  playing rounds instead of the calendar, so the off-seasons no longer take up
  more width than the seasons.
- Command line interface: `bscores info | rate | forecast | tune | simulate`,
  installed as a console script and runnable as `python -m bscores`. Reads the
  bundled archive or any CSV, and emits text or JSON.
- `mypy` and coverage configuration, and a release workflow that publishes to
  PyPI on a tag.
- `TUTORIAL.md`: a guided tour of the whole package, with every output and
  figure generated from the bundled data.
- `bscores.baselines.tune_elo` searches Elo's hyperparameters on a validation
  window, so the baseline gets the same budget the B-score grid does.

### Changed

- The AFL comparison now tunes Elo on the same validation window rather than
  running it on the paper's defaults. This changes the headline result: a
  tuned Elo reaches 0.5817 log-loss against the tuned B-score's 0.5853, and
  Diebold-Mariano cannot separate them (+0.59, p = 0.55). The previous
  "B-scores beat Elo by −1.98, p = 0.048" compared a searched model against an
  unsearched one. A B-score model absorbs home advantage through the
  calibrating logit's intercept; the paper's Elo has no such term, so it has to
  be given one for the comparison to be like for like.

### Fixed

- `infer_rounds` returned different round numbers depending on the order
  matches were passed in, because same-day matches — most of them, at day
  resolution — were walked in row order. Ties are now broken on competitor
  name, so the result depends only on the set of matches. The bundled `round`
  column shifted for 4 of 3533 matches and is now exactly reproducible.
- `plot_tuning` drew categorical parameters as bars from zero, which hid
  differences of a fraction of a percent — exactly the differences the plot
  exists to show. It now draws dots on a zoomed axis.
- The CLI printed a `BrokenPipeError` traceback when its output was piped into
  a command that exits early, such as `head`.

## [0.2.0] - 2026-08-11

### Added

- `bscores.tuning` — `grid_search` over a validation window, `refit_best` for
  a held-out test window, and `AFL_TUNED`. Configurations that differ only in
  how the logit is fitted share one causal rating pass.
- `bscores.weights` — `margin_weight` and `importance_weight` for per-result
  arc weights.
- `bscores.diagnostics` — `explain_rating` decomposes a rating into the wins
  that produced it, plus `calibration_curve`, `reliability_table`, `sharpness`,
  `upset_rate`, `network_summary`, `head_to_head` and `rating_churn`.
- `bscores.simulation.simulate_season` — Monte Carlo season outcomes.
- `bscores.plotting` — optional matplotlib figures, behind the `plot` extra.
- `backtest.walk_forward` and `BacktestResult.between`.

### Changed

- The bundled AFL dataset now runs to 2026-08-02: 3533 matches, up from 2534.
- The R package that previously lived in this repository has been removed.

### Removed

- `sweep_alpha`, which tuned and reported on the same window. Use
  `bscores.tuning.grid_search`, which separates the two.

## [0.1.0] - 2026-08-11

### Added

- First Python implementation of the B-score method of Arcagni, Candila &
  Grassi (2023): time-weighted loss network, Bonacich centrality, logit
  calibration, expanding-window back-testing and an Elo baseline.
- Bundled AFL archive, 2009-2022.

[Unreleased]: https://github.com/dclaz/bscores/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/dclaz/bscores/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/dclaz/bscores/releases/tag/v0.1.0
