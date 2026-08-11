# bscores

A Python implementation of **B-scores** — rating and forecasting pairwise
contests using the eigenvector centrality of the network of results.

> Arcagni, A., Candila, V. & Grassi, R. (2023). *A new model for predicting the
> winner in tennis based on the eigenvector centrality.* Annals of Operations
> Research 325, 615–632. <https://doi.org/10.1007/s10479-022-04594-7>

Every result becomes an arc in a directed network, pointing from the loser to
the winner and weighted by how recently it happened. A competitor's rating is
its entry in the principal eigenvector of that network, so a rating is high when
the competitors it has beaten are themselves highly rated. The consequence that
sets the method apart from Elo, Glicko and Bradley-Terry:

> every new match updates the full network, rather than only the ratings of the
> players involved

A team that does not play still moves, because the standing of everyone it has
beaten moves.

## Install

```bash
pip install -e .            # numpy only
pip install -e ".[dev]"     # + pandas, scipy, pytest, ruff
```

`numpy` is the only hard requirement. `pandas` is needed for the DataFrame
loaders, `scipy` only for the sparse path (thousands of competitors).

## Quick start

The API deliberately mirrors [openskill.py](https://github.com/vivekjoshy/openskill.py),
so swapping rating systems is mostly an import change.

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

`Rating` carries `.score` (the B-score, an entry in a unit-norm vector) and
`.ordinal()` for a friendlier display scale. Other openskill-shaped calls —
`rate_result`, `predict_draw`, multi-member teams, explicit `ranks` — behave as
you would expect; see the docstrings.

## Fitting and forecasting

The paper does not read a probability straight off the centralities. It fits a
logit on the two ratings (Eq. 3):

$$p_{i,j,t+1} = \frac{\exp(\beta_0 + \beta_1 CR_{i,t} + \beta_2 CR_{j,t})}
{1 + \exp(\beta_0 + \beta_1 CR_{i,t} + \beta_2 CR_{j,t})}$$

Keeping $\beta_1$ and $\beta_2$ free rather than forcing $\beta_2 = -\beta_1$
matters for a home/away sport: the intercept absorbs home advantage.

```python
from bscores import BScoreModel
from bscores.datasets import load_afl

afl = load_afl(as_frame=False)
model = BScoreModel(alpha=21.0).fit(
    afl.home_team, afl.away_team, afl.outcome, afl.date
)
model.predict_win([["Melbourne"], ["Carlton"]])     # calibrated home-win probability
```

`fit` ingests the fixtures and calibrates in one pass. The features it
calibrates on are computed *causally* — each match sees only results that
happened strictly before it — so ingesting first cannot leak.

## Evaluating out of sample

`rolling_forecast` runs the paper's protocol: fit on everything up to a cut-off,
forecast the next block, fold it in, refit, repeat.

```python
from bscores import rolling_forecast, sweep_alpha

result = rolling_forecast(
    afl.home_team, afl.away_team, afl.outcome, afl.date,
    alpha=21.0, initial_train=0.5, refit_every=300,
)
result.metrics()          # {'log_loss': 0.6296, 'brier_score': 0.2182, ...}
result.to_frame()         # per-match forecasts, ratings and training-set sizes

sweep_alpha(afl.home_team, afl.away_team, afl.outcome, afl.date,
            [7, 21, 90, 365, 1095])     # tune the memory parameter
```

## Results on the bundled AFL data

2534 AFL matches, 2009–2022, forecasting the last 1267 out of sample
(`python examples/afl_diagnostics.py`):

| model | log-loss | Brier | accuracy |
| --- | ---: | ---: | ---: |
| B-score, α = 21 days | **0.6296** | **0.2182** | **0.6440** |
| B-score, α = 365 days (the paper's) | 0.6540 | 0.2292 | 0.6077 |
| Elo (Kovalchik K-schedule) | 0.6508 | 0.2267 | 0.6172 |
| home-ground base rate | 0.6836 | 0.2433 | 0.5651 |

At α = 21 the Diebold-Mariano test puts B-scores significantly ahead of Elo
(DM = −2.49, p = 0.013). At the paper's α = 365 the two are statistically
indistinguishable (DM = +0.36, p = 0.72) — a season of AFL is 22 rounds, and
form turns over much faster than the 52-week ATP/WTA ranking window α = 365 was
chosen to mirror. The paper flags the choice of α as an open question; on this
competition the answer is "much shorter".

Two things worth stating plainly:

- **The betting results do not carry over.** Applying Definition 1's staking
  rule to the bundled closing odds gives a negative ROI at every threshold
  tested (−0.7% to −4.7%). The paper's positive returns were on tennis markets;
  the AFL head-to-head market in this sample is not beatable this way.
- **α matters more than anything else here.** It moves log-loss by more than the
  entire gap between B-scores and Elo.

## What is in the box

| module | what it holds |
| --- | --- |
| `bscores.models` | `BScoreModel`, `Rating`, `RatingHistory` — the API you use |
| `bscores.network` | `LossNetwork`: dated results in, $W_t$ out (Eq. 1) |
| `bscores.centrality` | `bonacich_centrality` (Eq. 11), solvers, `neumann_centrality` |
| `bscores.decay` | `Hyperbolic` (Eq. 2), `Exponential`, `Uniform`, `Window` |
| `bscores.calibration` | `LogitCalibrator` (Eq. 3), `fit_logistic` — IRLS, numpy only |
| `bscores.metrics` | `log_loss`, `brier_score`, `accuracy`, `diebold_mariano`, `roi` |
| `bscores.backtest` | `rolling_forecast`, `sweep_alpha` |
| `bscores.baselines` | `Elo` (Eqs. 4–5), for comparison |
| `bscores.datasets` | `load_afl` — 2534 AFL matches, bundled |

## Design notes

**Causality is enforced, not assumed.** `score_history`, `match_scores`,
`predict_proba` and `rolling_forecast` all evaluate the network with
`inclusive=False`: a rating used to forecast a match starting at time *t* is
built only from results strictly before *t*, so same-round fixtures cannot see
each other. The test suite rewrites the second half of a fixture list and
asserts the first half's forecasts do not move.

**Degenerate networks are handled explicitly.** Before any competitor has beaten
someone who beat someone else in a loop, the loss matrix is nilpotent: its
spectral radius is zero and Eq. 11 has no solution. Power iteration would crawl
towards an arbitrary basis vector. `bonacich_centrality` tests for a cycle up
front — one sparsity sweep, essentially free when a cycle exists — and falls
back to a finite Neumann series that grades the same "beat strong opponents"
idea and terminates in at most *n* steps. The choice is reported in
`EigenResult.method`, never silently. Pass `regularization=ε` to force the
network irreducible and stay on the eigenvector throughout.

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
- `max_age` bounds the work per epoch when the hyperbolic tail is not worth
  paying for.
- Networks above `dense_max_nodes` competitors switch to SciPy CSR.

## Development

```bash
pip install -e ".[dev]"
pytest                                    # 350 tests, ~9s
ruff check src tests scripts
python examples/afl_diagnostics.py        # ratings, forecasts, alpha sweep, ROI
python scripts/build_afl_dataset.py       # rebuild the bundled data from data-raw/
```

## Relationship to the R package

This replaces the R package that previously lived here; the R sources and
`data/afl_matches_df.rda` remain only until the migration is signed off. The
bundled Python dataset is byte-for-byte equivalent to the R `afl_matches_df`
(same 2534 rows, same 12 columns, verified field by field), and
`bscores.metrics` is a faithful port of `R/loss_functions.R`, tolerance-clipping
behaviour included.

## Licence

GPL-3.0-or-later, as the R package was.
