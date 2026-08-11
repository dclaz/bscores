# Contributing

## Getting set up

```bash
git clone -b claude/bscores-python-impl-dsbd3i https://github.com/dclaz/bscores
cd bscores
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Before opening a pull request

```bash
pytest                                      # unit tests and doctests
ruff check src tests scripts examples       # lint
ruff format --check src tests               # formatting
mypy                                        # types
```

The test suite runs in well under a minute; there is no reason to skip it.

## How the package is laid out

| module | responsibility |
| --- | --- |
| `network` | dated results in, the weighted loss matrix out |
| `centrality` | the eigenvector solve and its degenerate fallback |
| `decay`, `weights` | how much a result counts, by age and by margin |
| `models` | the user-facing `BScoreModel` |
| `calibration` | ratings to probabilities |
| `backtest`, `tuning` | out-of-sample evaluation and hyperparameter search |
| `diagnostics`, `plotting`, `simulation` | inspecting what the model did |
| `schedule`, `datasets` | fixture-list structure and the bundled archive |

The core depends on numpy alone. `pandas`, `scipy` and `matplotlib` are
optional extras and must stay that way — import them lazily, inside the
function that needs them, so `import bscores` keeps working without them.

## Conventions worth knowing

**Causality is a hard invariant.** Anything that produces a rating used to
forecast a match must evaluate the network with `inclusive=False`, so it sees
only results that happened strictly before that match. There are tests that
rewrite the second half of a fixture list and assert the first half's forecasts
do not move; if you touch `score_history`, `match_scores`, `rolling_forecast` or
`grid_search`, keep them passing.

**Tuning and reporting are separate windows.** `grid_search` scores on a
validation window and `refit_best` reports on a test window the search never
saw. Please do not add a helper that collapses the two.

**Tests state properties, not snapshots.** A test that asserts a particular
team tops the ladder expires the next time the data is refreshed. Prefer
assertions about behaviour — that a shorter memory tracks recent form more
closely than a longer one, say — which stay true whatever the archive says.

**Numbers in prose go stale.** If an example prints a claim about its own
results, compute the claim from the run rather than typing it into a string.

## Refreshing the bundled data

The AFL archive comes from
<https://www.aussportsbetting.com/historical_data/afl.xlsx>, which now sits
behind a bot challenge and has to be downloaded by hand. Drop the workbook at
`data/afl.xlsx` and run:

```bash
python scripts/build_afl_dataset.py
```

Then re-run `python examples/afl_tuning.py`, since the documented results and
`AFL_TUNED` are derived from that archive.

## Releasing

1. Move the `Unreleased` section of `CHANGELOG.md` under the new version.
2. Bump `version` in `pyproject.toml` and `__version__` in `src/bscores/__init__.py`.
3. Tag `vX.Y.Z` and push the tag; the release workflow builds and publishes.
