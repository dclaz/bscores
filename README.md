# bscores

Rate and forecast pairwise contests using the eigenvector centrality of the
network of results.

> Arcagni, A., Candila, V. & Grassi, R. (2023). *A new model for predicting the
> winner in tennis based on the eigenvector centrality.* Annals of Operations
> Research 325, 615–632. <https://doi.org/10.1007/s10479-022-04594-7>

Every result becomes an arc in a directed network, pointing from the loser to
the winner and weighted by how recently it happened. A competitor's rating — its
**B-score** — is its entry in the principal eigenvector of that network, so a
rating is high when the competitors it has beaten are themselves highly rated.
The consequence that sets the method apart from Elo, Glicko and Bradley-Terry:

> every new match updates the full network, rather than only the ratings of the
> players involved

A side that does not play still moves, because the standing of everyone it has
beaten moves.

## Install

```bash
pip install -e .              # numpy only
pip install -e ".[all]"       # + pandas, scipy, matplotlib
pip install -e ".[dev]"       # + pytest, ruff
```

`numpy` is the only hard requirement. `pandas` is for the DataFrame adapters,
`scipy` for the sparse path (thousands of competitors), `matplotlib` for
`bscores.plotting`.

## Quick start

The API mirrors [openskill.py](https://github.com/vivekjoshy/openskill.py), so
swapping rating systems is mostly an import change.

```python
from bscores import BScoreModel

model = BScoreModel(alpha=365.0)          # results halve in weight after a year

# Teams in finishing order, best first.
model.rate([["Geelong"], ["Carlton"]],  at="2021-03-18")
model.rate([["Geelong"], ["Essendon"]], at="2021-03-25")
model.rate([["Carlton"], ["Essendon"]], at="2021-04-01")
model.rate([["Essendon"], ["Geelong"]], at="2021-04-08")
model.rate([["Carlton"], ["Geelong"]],  at="2021-04-15")

model.rating("Geelong")
# Rating(name='Geelong', score=0.636926, matches=4)

model.leaderboard()
# [Rating(name='Carlton',  score=0.658014, matches=3, rank=1),
#  Rating(name='Geelong',  score=0.636926, matches=4, rank=2),
#  Rating(name='Essendon', score=0.401675, matches=3, rank=3)]

model.predict_win([["Geelong"], ["Carlton"]])            # [0.492, 0.508]
model.predict_rank([["Geelong"], ["Carlton"], ["Essendon"]])
# [(2, 0.375), (1, 0.388), (3, 0.237)]
```

The distinguishing behaviour is easiest to see by playing a match Essendon is
not in:

```python
model.rating("Essendon").score            # 0.4017
model.rate([["Carlton"], ["Geelong"]], at="2021-04-22")
model.rating("Essendon").score            # 0.2999
```

Essendon's rating fell by a quarter without taking the field: its one win was
over Geelong, and Geelong just lost again. Elo would not have moved it at all.

`Rating` carries `.score` (an entry in a unit-norm vector) and `.ordinal()` for
a friendlier display scale. `rate_result`, `predict_draw`, multi-member teams and
explicit `ranks` all behave as you would expect; see the docstrings.

## Fitting and forecasting

The paper does not read a probability straight off the centralities. It fits a
logit on the two ratings (Eq. 3):

$$p_{i,j,t+1} = \frac{\exp(\beta_0 + \beta_1 CR_{i,t} + \beta_2 CR_{j,t})}
{1 + \exp(\beta_0 + \beta_1 CR_{i,t} + \beta_2 CR_{j,t})}$$

Keeping $\beta_1$ and $\beta_2$ free rather than forcing $\beta_2 = -\beta_1$
matters for a home/away sport: the intercept absorbs home advantage.

```python
from bscores import BScoreModel, rolling_forecast
from bscores.datasets import load_afl

afl = load_afl(as_frame=False)
model = BScoreModel(alpha=120.0, kernel="exponential").fit(
    afl.home_team, afl.away_team, afl.outcome, afl.date
)
model.predict_win([["Melbourne"], ["Carlton"]])     # calibrated home-win probability

result = rolling_forecast(                          # expanding-window evaluation
    afl.home_team, afl.away_team, afl.outcome, afl.date,
    alpha=120.0, initial_train=0.5, refit_every=300,
)
result.metrics()      # {'log_loss': ..., 'brier_score': ..., 'accuracy': ...}
```

`fit` ingests the fixtures and calibrates in one pass. The features it
calibrates on are computed *causally* — each match sees only results that
happened strictly before it — so ingesting first cannot leak.

## Tune before you trust a default

`alpha` moves accuracy further than the choice of rating system does. The paper
fixes it at 365 to mirror the 52-week ATP/WTA ranking window and flags the
choice as open; on a competition with a different rhythm it is simply wrong.

```python
from bscores.tuning import grid_search, refit_best
from bscores.weights import margin_weight

options = {"mov": margin_weight(afl.margin, scheme="linear", scale=24.0, cap=3.0)}
search = grid_search(
    afl.home_team, afl.away_team, afl.outcome, afl.date,
    grid={
        "alpha": [30.0, 60.0, 120.0, 365.0],
        "kernel": ["hyperbolic", "exponential"],
        "transform": ["identity", "sqrt"],
        "weights": ["mov"],
    },
    weight_options=options,
    validation_start="2016-01-01", validation_end="2019-01-01",   # tune here
)
search.best_params            # {'alpha': 120.0, 'kernel': 'exponential', ...}
search.sensitivity("alpha")   # how much does this knob actually matter?

held_out = refit_best(        # ... report on a window the search never saw
    afl.home_team, afl.away_team, afl.outcome, afl.date,
    search.best_params, weight_options=options, test_start="2019-01-01",
)
```

The three-way split is the point. Tuning and reporting on the same matches
inflates the result by however hard you searched:

```
2009-06 .. 2015-12   warm-up      1301 matches, the model builds a history
2016-01 .. 2018-12   validation    621 matches, the search runs here
2019-01 .. 2022-04   test          612 matches, reported once at the end
```

Configurations that differ only in how the logit is fitted share one causal
rating pass, so a 480-point grid costs 120 sweeps, not 480.

## What the sweep found on AFL

`python examples/afl_tuning.py`. Scored on the held-out 2019–2022 window:

| model | log-loss | Brier | accuracy |
| --- | ---: | ---: | ---: |
| B-score, tuned | **0.6358** | **0.2200** | **0.6503** |
| B-score, paper defaults (α = 365, hyperbolic) | 0.6735 | 0.2386 | 0.5605 |
| Elo (Kovalchik K-schedule) | 0.6729 | 0.2373 | 0.5931 |
| home-ground base rate | 0.6872 | 0.2450 | 0.5523 |

Diebold-Mariano against the tuned model: −3.08 vs the paper's defaults
(p = 0.002), −3.00 vs Elo (p = 0.003). The winning configuration:

```python
{"alpha": 120.0, "kernel": "exponential", "transform": "sqrt",
 "regularization": 0.03, "weights": "mov"}      # bscores.tuning.AFL_TUNED
```

Which knobs actually mattered, by best achievable validation log-loss:

| knob | best | worst tried | verdict |
| --- | ---: | ---: | --- |
| `alpha` | 0.5856 @ 120 d | 0.6734 @ 3650 d | dominant — always tune it |
| `kernel` | 0.5971 exponential | 0.6070 hyperbolic | worth switching |
| `weights` | 0.5879 margin-linear | 0.5978 finals-weighted | second-biggest lever |
| `transform` | 0.5879 sqrt | 0.5939 identity | small but free |
| `regularization` | 0.5856 @ 0.03 | 0.5871 @ 0 | marginal |
| `draw_weight` | 0.5879 @ 0.5 | 0.5881 @ 0 | noise — leave it |
| `symmetric` | 0.5879 True | 0.5883 False | noise — leave it |
| `refit_every` | 0.5971 @ 300 | 0.5977 @ 100 | noise — leave it |

Three findings worth stating plainly:

- **The hyperbolic kernel's problem is its tail, not its shape.** At α = 365 a
  decade-old result still carries weight 0.09, and there are thousands of them.
  Capping it — `Hyperbolic(120)` with `max_age=365` — scores 0.5973, matching the
  exponential kernel's 0.5971. Either fix works; doing neither costs a full
  0.01 of log-loss.
- **Margin of victory is worth as much as the kernel choice.** A 100-point
  thrashing says more than a one-point escape, and `bscores.weights.margin_weight`
  is the cheapest accuracy on offer.
- **The betting result does not replicate.** Applying Definition 1's staking rule
  to the bundled closing odds gives a negative ROI at every threshold tested
  (−3.5% to −5.8% for the tuned model, −0.7% to −4.7% at the paper's defaults).
  The paper's positive returns were on tennis markets; the AFL head-to-head
  market in this sample is not beatable this way, and tuning the model for
  accuracy made the betting result *worse*, not better.

Test-window log-loss is worse than validation-window log-loss for *every* model:
2019–2022 was simply less predictable. Only the gaps within a window mean
anything.

## Exploring ratings

`python examples/afl_explore.py --plot out/`

**Why is a competitor rated where it is?** The eigenvector equation says a
rating *is* the weighted sum of the ratings pointing at it, so it decomposes
exactly — no attribution heuristic required.

```python
from bscores.diagnostics import explain_rating

for part in explain_rating(model, "Melbourne", top=3):
    print(part.opponent, f"{part.share:.1%}", part.opponent_score)
```

**Is the network healthy enough to rate on?**

```python
from bscores.diagnostics import network_summary
network_summary(model)
# {'competitors': 18, 'arcs': 306, 'density': 1.0, 'has_cycle': True,
#  'spectral_radius': 7.47, 'unrated': 0, 'solver': 'power', ...}
```

**Are the probabilities honest?** `calibration_curve`, `reliability_table`,
`sharpness` and `upset_rate`. Calibration alone is easy to fake by always
predicting the base rate; sharpness is the other half of the picture.

**How much does the order churn?** `rating_churn` — a short memory tracks form
and churns, a long one is steadier. On AFL, mean rank change per match day is
0.96 at a 30-day half-life, 0.43 at 120 days and 0.12 at 1095.

**What are our chances?** `simulate_season` plays the remaining fixtures a few
thousand times:

```python
from bscores.simulation import simulate_season
season = simulate_season(model, home_fixtures, away_fixtures, n_simulations=20_000)
season.top_n_probability(4)      # {'Melbourne': 0.848, 'Sydney': 0.622, ...}
season.position_distribution("Geelong")
```

**Figures.** `bscores.plotting` gives `plot_ratings`, `plot_calibration`,
`plot_tuning`, `plot_network`, `plot_backtest` and `plot_decay`. Each takes an
optional `ax` and returns it, so they compose into a dashboard.

## What is in the box

| module | what it holds |
| --- | --- |
| `bscores.models` | `BScoreModel`, `Rating`, `RatingHistory` — the API you use |
| `bscores.network` | `LossNetwork`: dated results in, $W_t$ out (Eq. 1) |
| `bscores.centrality` | `bonacich_centrality` (Eq. 11), solvers, `neumann_centrality` |
| `bscores.decay` | `Hyperbolic` (Eq. 2), `Exponential`, `Uniform`, `Window` |
| `bscores.weights` | `margin_weight`, `importance_weight` — per-result arc weights |
| `bscores.calibration` | `LogitCalibrator` (Eq. 3), `fit_logistic` — IRLS, numpy only |
| `bscores.metrics` | `log_loss`, `brier_score`, `accuracy`, `diebold_mariano`, `roi` |
| `bscores.backtest` | `rolling_forecast`, `walk_forward`, `sweep_alpha` |
| `bscores.tuning` | `grid_search`, `refit_best`, `AFL_TUNED` |
| `bscores.diagnostics` | `explain_rating`, calibration, network health, churn |
| `bscores.simulation` | `simulate_season` |
| `bscores.plotting` | matplotlib figures (optional extra) |
| `bscores.baselines` | `Elo` (Eqs. 4–5), for comparison |
| `bscores.datasets` | `load_afl` — 2534 AFL matches, bundled |

## Design notes

**Causality is enforced, not assumed.** `score_history`, `match_scores`,
`predict_proba`, `rolling_forecast` and `grid_search` all evaluate the network
with `inclusive=False`: a rating used to forecast a match starting at time *t* is
built only from results strictly before *t*, so same-round fixtures cannot see
each other. The test suite rewrites the second half of a fixture list and asserts
the first half's forecasts do not move, and does the same for the validation and
test windows of a search.

**Degenerate networks are handled explicitly.** Before anyone has beaten someone
who beat someone else in a loop, the loss matrix is nilpotent: its spectral
radius is zero and Eq. 11 has no solution. Power iteration would crawl towards an
arbitrary basis vector. `bonacich_centrality` tests for a cycle up front — one
sparsity sweep, essentially free when a cycle exists — and falls back to a finite
Neumann series that grades the same "beat strong opponents" idea and terminates
in at most *n* steps. The choice is reported in `EigenResult.method`, never
silently. Pass `regularization=ε` to force the network irreducible and stay on the
eigenvector throughout.

**Performance.** `python examples/benchmark.py`:

| competitors | matches | epochs | seconds |
| ---: | ---: | ---: | ---: |
| 18 | 2 500 | 1 795 | 0.9 |
| 64 | 10 000 | 3 404 | 1.8 |
| 128 | 25 000 | 3 648 | 2.9 |
| 256 | 50 000 | 3 650 | 6.5 |

An epoch is one distinct match date, and one centrality solve. The full AFL
back-test — 2534 matches, ~1400 causal solves, five logit refits — runs in about
half a second. What makes that work:

- $W_t$ is assembled with a scatter-add over a pre-sorted event array, so an
  epoch costs one pass over history rather than a Python loop over it.
- Repeated times are solved once and shared.
- Each solve warm-starts from the previous epoch's eigenvector; consecutive
  networks barely differ.
- The power iteration runs shifted, which makes the bipartite-ish networks of an
  opening round converge instead of oscillating.
- Memoryless kernels (`Exponential`, `Uniform`) age the accumulator in place
  instead of rebuilding it — exact, and roughly 30% faster on long histories.
- `max_age` bounds the work per epoch when a heavy kernel tail is not worth
  paying for.
- Networks above `dense_max_nodes` competitors switch to SciPy CSR.

## Development

```bash
pip install -e ".[dev]"
pytest                                    # 474 tests, ~25s
ruff check src tests scripts examples
python examples/afl_tuning.py             # hyperparameter search
python examples/afl_diagnostics.py        # ratings, forecasts, ROI
python examples/afl_explore.py --plot out/
python scripts/build_afl_dataset.py       # rebuild the bundled data
```

## Licence

GPL-3.0-or-later.
