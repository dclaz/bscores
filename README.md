# bscores

**Rate and forecast pairwise contests using the eigenvector centrality of the
network of results.**

`bscores` gives you a rating for every competitor in a competition, a calibrated
probability for any match you care to ask about, and the tooling to check
whether those probabilities are any good — back-testing, hyperparameter search,
diagnostics, season simulation and plots. It works on anything decided by
head-to-head results: sports leagues, tournaments, chess, esports, A/B contests.

**The method.** Every result becomes an arrow in a directed network, pointing
from the loser to the winner and weighted by how recently the match was played.
A competitor's rating — its *B-score* — is its entry in the principal
eigenvector of that network, which makes the rating recursive: your rating is
high when the competitors you have beaten are themselves highly rated. Because
every rating depends on every other, a single result updates the entire
competition, including competitors who did not play. Probabilities come from a
logistic regression on the two ratings.

This is a Python implementation of:

> Arcagni, A., Candila, V. & Grassi, R. (2023). *A new model for predicting the
> winner in tennis based on the eigenvector centrality.* Annals of Operations
> Research 325, 615–632. <https://doi.org/10.1007/s10479-022-04594-7>

Not yet on PyPI — install from the repository:

```bash
pip install "git+https://github.com/dclaz/bscores@claude/bscores-python-impl-dsbd3i"
```

```python
from bscores import BScoreModel
from bscores.datasets import load_afl

afl = load_afl(as_frame=False)                      # 3533 AFL matches, 2009-2026
model = BScoreModel(alpha=120.0, kernel="exponential")
model.fit(afl.home_team, afl.away_team, afl.outcome, afl.date)

model.leaderboard(top=3)
# [Rating(name='Fremantle', score=0.4059, matches=395, rank=1),
#  Rating(name='Geelong',   score=0.3346, matches=419, rank=2),
#  Rating(name='Hawthorn',  score=0.3126, matches=406, rank=3)]

model.predict_win([["Fremantle"], ["Richmond"]])    # [0.944, 0.056]
```

New to the method? **[TUTORIAL.md](TUTORIAL.md)** walks through the whole
package end to end — rating, forecasting, tuning, diagnostics and every plot —
against the bundled data.

---

## The idea in more detail

Most rating systems treat a match as a private transaction: two competitors
play, their two ratings move, everyone else's stay put. B-scores treat the whole
competition as one object.

Formally, the network at time $t$ collects every past result, each decayed by
its age (Eq. 1 of the paper). Writing $L_s$ for the matrix of losses recorded at
time $s$, and $f$ for the decay kernel:

$$W_t = \sum_{s \le t} f(s, t, \alpha) \, L_s$$

The ratings are then the principal eigenvector of its transpose (Eq. 11), where
$\rho$ is the spectral radius:

$$x = \frac{1}{\rho} \, W_t' \, x, \qquad \lVert x \rVert_2 = 1$$

Reading that equation on this network — where an arrow points from loser to
winner — says a competitor scores highly when the competitors *pointing at* it
score highly, which is to say when it beats strong opponents.

The recursion has a consequence worth pausing on. Because every rating depends
on every other, **a new result moves everybody** — including competitors who
were nowhere near the match:

```python
model = BScoreModel(alpha=365.0)
for winner, loser, when in [("a", "b", "2021-03-01"), ("b", "c", "2021-03-08"),
                            ("c", "a", "2021-03-15"), ("a", "c", "2021-03-22"),
                            ("b", "a", "2021-03-29")]:
    model.rate_result(winner, loser, at=when)

model.rating("c").score        # 0.3979
model.rate_result("b", "a", at="2021-04-05")   # c is not in this match
model.rating("c").score        # 0.2963
```

`c`'s rating fell by a quarter without taking the field. Its one win was over
`a`, and `a` just lost again — so that win is now worth less. Elo, Glicko and
Bradley-Terry would all have left `c` untouched. This is the property the method
exists for, and it is why ratings are computed by solving the network rather
than by updating a pair of numbers.

---

## A tour of the module

### Rating

The API follows [openskill.py](https://github.com/vivekjoshy/openskill.py), so
if you have code built on a rating system you can usually swap the import.

```python
from bscores import BScoreModel

model = BScoreModel(alpha=365.0)          # a result halves in weight after a year

model.rate([["Geelong"], ["Carlton"]], at="2021-03-18")   # teams, best first
model.rate_result("Carlton", "Essendon", at="2021-03-25") # or winner/loser
model.add_matches(home, away, outcome, dates)             # or a whole fixture list

model.rating("Geelong")            # Rating(name='Geelong', score=..., matches=2)
model.leaderboard(top=5)
model.predict_win([["Geelong"], ["Carlton"]])
model.predict_rank([["Geelong"], ["Carlton"], ["Essendon"]])
```

A `Rating` carries `.score` — an entry in a unit-norm vector, so always in
$[0, 1]$ — plus `.matches`, `.wins` and `.ordinal()` for a friendlier display
scale. Multi-member teams, explicit `ranks`, draws and per-result weights are
all supported; the docstrings have the details.

### Forecasting

The paper does not read a probability straight off two centralities. It fits a
logit on them (Eq. 3):

$$p_{i,j} = \frac{\exp(\beta_0 + \beta_1 x_i + \beta_2 x_j)}
{1 + \exp(\beta_0 + \beta_1 x_i + \beta_2 x_j)}$$

Leaving $\beta_1$ and $\beta_2$ free, rather than forcing
$\beta_2 = -\beta_1$, is what lets the intercept carry home advantage. On AFL
data $\sigma(\beta_0) \approx 0.54$ against a 0.572 home win rate — most of the
edge, with the remainder coming from the slopes not quite cancelling.

```python
model.fit(afl.home_team, afl.away_team, afl.outcome, afl.date)
model.predict_win([["Melbourne"], ["Carlton"]])    # calibrated probability
```

`fit` ingests the fixtures and calibrates in one pass. Every feature it
calibrates on is computed **causally** — each match sees only results that
happened strictly before it — so ingesting first cannot leak future information
backwards.

### Evaluating

`rolling_forecast` runs the paper's protocol: fit on everything up to a cut-off,
forecast the next block, fold it in, refit, repeat.

```python
from bscores import rolling_forecast

result = rolling_forecast(
    afl.home_team, afl.away_team, afl.outcome, afl.date,
    model=BScoreModel(alpha=120.0, kernel="exponential"),
    initial_train="2023-01-01", refit_every=300,
)
result.metrics()     # {'log_loss': 0.5910, 'brier_score': 0.2011, 'accuracy': 0.6606, ...}
result.to_frame()    # per-match forecasts, ratings and training-set sizes
```

### Tuning

`alpha` — how fast results are forgotten — is the parameter that matters most,
and the right value depends on how fast form turns over in your competition.
The paper uses 365 days to mirror the 52-week ATP/WTA ranking window; an AFL
season is 22 rounds, and the data prefers something different.

`grid_search` searches on a validation window and `refit_best` reports on a test
window the search never saw, so the number you quote is not the number you
optimised:

```python
from bscores.tuning import grid_search, refit_best

search = grid_search(
    afl.home_team, afl.away_team, afl.outcome, afl.date,
    grid={"alpha": [30.0, 60.0, 120.0, 365.0], "kernel": ["hyperbolic", "exponential"]},
    validation_start="2019-01-01", validation_end="2023-01-01",
)
search.best_params            # {'alpha': 120.0, 'kernel': 'exponential'}
search.sensitivity("alpha")   # how much does this knob actually matter?

held_out = refit_best(
    afl.home_team, afl.away_team, afl.outcome, afl.date,
    search.best_params, test_start="2023-01-01",
)
```

```
2009-06 .. 2018-09   warm-up      1922 matches, the model builds a history
2019-03 .. 2022-09   validation    783 matches, the search runs here
2023-03 .. 2026-08   test          828 matches, reported once at the end
```

Searching is cheaper than it looks: settings that differ only in how the logit
is fitted share one causal rating pass with their siblings, so a 480-point grid
costs 120 rating sweeps rather than 480.

### Understanding a rating

Because a rating *is* the weighted sum of the ratings pointing at it, it
decomposes exactly — no attribution heuristic involved:

```python
from bscores.diagnostics import explain_rating

for part in explain_rating(model, "Fremantle", top=4):
    print(f"{part.opponent:<18} {part.share:>6.1%}  (they rate {part.opponent_score:.3f})")

# Western Bulldogs   16.9%  (they rate 0.245)
# Sydney             10.7%  (they rate 0.312)
# Geelong            10.0%  (they rate 0.335)
# Hawthorn            8.6%  (they rate 0.313)
```

Each row is a competitor Fremantle has beaten, weighted by how recently and by
how highly that competitor is itself rated — so a win over Geelong contributes
more per match than one over the Bulldogs, even though the Bulldogs have been
beaten more often. The shares sum to 1 across all opponents.

Other diagnostics answer the questions that usually come next:

| question | function |
| --- | --- |
| Are the probabilities honest? | `calibration_curve`, `reliability_table` |
| Does the model ever commit? | `sharpness`, `upset_rate` |
| Is the network dense enough to rate on? | `network_summary` |
| How much does the order move? | `rating_churn` |
| Who has the wood on whom? | `head_to_head` |

### Simulating

```python
from bscores.simulation import simulate_season

season = simulate_season(model, home_fixtures, away_fixtures, n_simulations=20_000)
season.top_n_probability(4)     # {'Fremantle': 0.867, 'Sydney': 0.728, ...}
season.expected_points()
season.position_distribution("Geelong")
```

### Plotting

Installing with the `plot` extra (see below) adds `plot_ratings`, `plot_calibration`,
`plot_tuning`, `plot_network`, `plot_backtest` and `plot_decay`. Each takes an
optional `ax` and returns it, so they compose into a dashboard.

```python
from bscores.plotting import plot_ratings

plot_ratings(history, top=6, x="round", schedule=afl)
```

`x="round"` is worth knowing about for a seasonal competition. On a calendar
axis every summer is a flat line through an off-season in which nothing
happened, and across seventeen seasons those gaps take up more width than the
matches do. Plotting against playing rounds puts the seasons side by side.
`bscores.schedule` recovers the rounds from the fixture list itself — a round is
a maximal run of matches in which no competitor plays twice, which handles byes,
split rounds and finals without a fixture template.

### From the command line

```bash
bscores info
bscores rate     --data afl --alpha 120 --kernel exponential --top 10
bscores forecast --data results.csv --initial-train 2023-01-01
bscores tune     --data afl --validation-start 2019-01-01 --validation-end 2023-01-01
bscores simulate --data afl --simulations 20000
```

`--data afl` uses the bundled archive; anything else is read as a CSV
(`--home-col`, `--away-col`, `--outcome-col`, `--date-col` if your headers
differ). `--format json` on any subcommand gives machine-readable output.

---

## A worked example, end to end

`python examples/afl_tuning.py` runs the search above on the bundled AFL
archive and scores the winner once on 2023–2026:

| model | log-loss | Brier | accuracy |
| --- | ---: | ---: | ---: |
| Elo, tuned | **0.5817** | **0.1987** | **0.6848** |
| B-score, tuned | 0.5853 | 0.1993 | 0.6703 |
| Elo, paper defaults | 0.6012 | 0.2062 | 0.6570 |
| B-score, paper defaults (α = 365, hyperbolic) | 0.6348 | 0.2201 | 0.6244 |
| home-ground base rate | 0.6808 | 0.2417 | 0.5749 |

Both tuned models chose their hyperparameters on the 2019–2022 validation
window and were scored once here. **Tuning matters more than the choice of
method:** either system, tuned, beats either system's defaults by a wide
margin, while the gap between the two tuned models is well inside the noise.

The tuned B-score settings —

```python
{"alpha": 120.0, "kernel": "exponential", "transform": "sqrt",
 "regularization": 0.03, "weights": "mov"}      # bscores.tuning.AFL_TUNED
```

— were also what an earlier search picked on the shorter 2009–2022 archive with
a *disjoint* validation window, which is a reassuring sign they describe the
competition rather than the sample. Elo's came from `tune_elo` over the same
window: `home_advantage=45, k_scale=400, k_power=0.4`.

Giving Elo a home-advantage term is what makes the comparison fair rather than
generous. A B-score model picks up home advantage for free — the calibrating
logit fits an intercept, and on a home/away competition that intercept *is* the
home edge — whereas the paper's Elo (Eqs. 4–5) has no such term and must be told.
Comparing a searched B-score model against a default Elo measures the search,
not the rating method:

```python
from bscores import Elo, tune_elo

params = tune_elo(afl.home_team, afl.away_team, afl.outcome, afl.date,
                  validation_start="2019-01-01", validation_end="2023-01-01")
Elo(**params).run(afl.home_team, afl.away_team, afl.outcome)
```

What the search learned about each knob, by best achievable validation
log-loss:

| knob | best | worst tried | how much it matters |
| --- | ---: | ---: | --- |
| `alpha` | 0.6246 @ 120 d | 0.6773 @ 3650 d | most of the available gain |
| `weights` | 0.6191 margin-linear | 0.6282 finals-weighted | second-biggest lever |
| `kernel` | 0.6246 exponential | 0.6352 hyperbolic | worth choosing deliberately |
| `transform` | 0.6246 sqrt | 0.6274 log | small, and free |
| `regularization` | 0.6177 @ 0.03 | 0.6191 @ 0 | marginal |
| `draw_weight` | 0.6191 @ 0.5 | 0.6192 @ 1.0 | indistinguishable |
| `symmetric` | 0.6191 False | 0.6194 True | indistinguishable |
| `refit_every` | 0.6241 @ 828 | 0.6246 @ 300 | indistinguishable |

Three things this brings out:

**The decay kernel's tail matters more than its shape.** The paper's hyperbolic
kernel is heavy-tailed: at α = 365 a decade-old result still carries weight
0.09, and a long archive holds thousands of them. Truncating it —
`Hyperbolic(60)` with `max_age=730` — scores 0.6232, essentially matching the
exponential kernel's 0.6246, where the untruncated version manages 0.6357.
Shortening `alpha` alone does not substitute, because that also discards useful
recent history. `plot_decay` makes the difference visible.

**How much a result counts is a real modelling choice.** The network carries a
weight per arc, and using it for margin of victory buys about as much as the
kernel choice does. `bscores.weights` provides `margin_weight` and
`importance_weight`; the search treats them as another grid dimension.

**The gains are worth testing for significance.** A Diebold-Mariano test[^dm]
against the tuned B-score model returns −5.24 versus the paper's B-score
defaults (p &lt; 0.0001), −1.98 versus a default Elo (p = 0.048), and −7.43
versus the base rate (p &lt; 0.0001). Against a *tuned* Elo it returns +0.59
(p = 0.55) — nominally behind, and nowhere near separable on 828 matches.

So the defensible claim on this data is not that B-scores beat Elo. It is that
B-scores reach Elo-class accuracy from a completely different construction, and
that tuning the memory parameter is worth far more than choosing between them.

[^dm]: The [Diebold-Mariano test](https://doi.org/10.1080/07350015.1995.10524599)
    asks whether two forecasters differ in accuracy by more than sampling noise.
    Rather than compare two summary numbers, it works with the *per-match* loss
    difference $d_i = L(\text{model A}_i) - L(\text{model B}_i)$ over the same
    test set and tests $H_0: \mathbb{E}[d] = 0$. The statistic is
    $\bar{d} / \mathrm{se}(\bar{d})$, asymptotically standard normal, with
    negative values favouring model A. The standard error comes from the
    long-run variance of $d$, which for the one-step-ahead forecasts used here
    needs no autocovariance lags; multi-step forecasts add them, and
    `bscores.metrics.diebold_mariano` takes a `horizon` argument for that. It
    also applies the Harvey-Leybourne-Newbold small-sample correction by
    default. Pairing the losses match by match is what gives the test its
    power: both models face identical fixtures, so the common difficulty of any
    given match cancels. That matters here because 828 matches is not many — a
    0.016 log-loss gap can easily be luck, and this is what separates the two
    cases. `BacktestResult.losses()` produces the per-match series it consumes.

Two more scripts round out the tour: `examples/afl_explore.py` walks through the
diagnostics and writes the figures, and `examples/benchmark.py` measures
throughput.

---

## What is in the box

| module | what it holds |
| --- | --- |
| `bscores.models` | `BScoreModel`, `Rating`, `RatingHistory` — the API you use |
| `bscores.network` | `LossNetwork`: dated results in, $W_t$ out (Eq. 1) |
| `bscores.centrality` | `bonacich_centrality` (Eq. 11), solvers, `neumann_centrality` |
| `bscores.decay` | `Hyperbolic` (Eq. 2), `Exponential`, `Uniform`, `Window` |
| `bscores.weights` | `margin_weight`, `importance_weight` |
| `bscores.calibration` | `LogitCalibrator` (Eq. 3), `fit_logistic` — IRLS, numpy only |
| `bscores.metrics` | `log_loss`, `brier_score`, `accuracy`, `diebold_mariano` |
| `bscores.backtest` | `rolling_forecast`, `walk_forward` |
| `bscores.tuning` | `grid_search`, `refit_best`, `AFL_TUNED` |
| `bscores.diagnostics` | `explain_rating`, calibration, network health, churn |
| `bscores.simulation` | `simulate_season` |
| `bscores.schedule` | `infer_seasons`, `infer_rounds` |
| `bscores.plotting` | matplotlib figures (optional extra) |
| `bscores.baselines` | `Elo` and `tune_elo`, for a fair comparison |
| `bscores.datasets` | `load_afl` — 3533 AFL matches, 2009–2026, bundled |

`numpy` is the only hard requirement. `pandas` powers the DataFrame adapters,
`scipy` the sparse path for thousands of competitors, `matplotlib` the figures —
all optional, all imported lazily.

```bash
REPO="git+https://github.com/dclaz/bscores@claude/bscores-python-impl-dsbd3i"

pip install "$REPO"                  # numpy only
pip install "bscores[all] @ $REPO"   # + pandas, scipy, matplotlib
pip install "bscores[dev] @ $REPO"   # + pytest, ruff, mypy
```

Or clone and work in place, which is what the development commands below
assume:

```bash
git clone -b claude/bscores-python-impl-dsbd3i https://github.com/dclaz/bscores
cd bscores && pip install -e ".[all]"
```

Once the package is released to PyPI these all become plain
`pip install bscores`; the release workflow that publishes it is already in
`.github/workflows/release.yml`, waiting on a `v*` tag.

---

## Implementation notes

**Causality is enforced.** `score_history`, `match_scores`, `predict_proba`,
`rolling_forecast` and `grid_search` all evaluate the network with
`inclusive=False`: a rating used to forecast a match at time *t* is built only
from results strictly before *t*, so fixtures in the same round cannot see each
other. The test suite rewrites the second half of a fixture list and asserts the
first half's forecasts do not move.

**Degenerate networks have a defined answer.** Before anyone has beaten someone
who beat someone else in a loop, the loss matrix is nilpotent: its spectral
radius is zero and the eigenvector equation has no solution to find. Power
iteration would crawl towards an arbitrary basis vector. `bonacich_centrality`
tests for a cycle up front — one sparsity sweep, essentially free once a cycle
exists — and falls back to a finite Neumann series that grades the same "beat
strong opponents" idea and terminates in at most *n* steps. Which route ran is
reported in `EigenResult.method`. Passing `regularization=ε` keeps the solve on
the eigenvector throughout.

**Speed.** `python examples/benchmark.py`:

| competitors | matches | epochs | seconds |
| ---: | ---: | ---: | ---: |
| 18 | 2 500 | 1 795 | 0.9 |
| 64 | 10 000 | 3 404 | 1.8 |
| 128 | 25 000 | 3 648 | 2.9 |
| 256 | 50 000 | 3 650 | 6.5 |

An epoch is one distinct match date, and one centrality solve. The full AFL
back-test — 3533 matches, roughly 2000 causal solves, six logit refits — takes
about three quarters of a second. What makes that work:

- $W_t$ is assembled with a scatter-add over a pre-sorted event array, so an
  epoch costs one pass over history rather than a Python loop over it.
- Repeated times are solved once and shared.
- Each solve warm-starts from the previous epoch's eigenvector; consecutive
  networks barely differ.
- The power iteration runs shifted, so the bipartite-ish networks of an opening
  round converge instead of oscillating.
- Memoryless kernels (`Exponential`, `Uniform`) age the accumulator in place
  rather than rebuilding it — exact, and roughly 30% faster on long histories.
- `max_age` bounds the per-epoch work when a heavy kernel tail is not worth
  paying for.
- Networks above `dense_max_nodes` competitors switch to SciPy CSR.

---

## Development

```bash
pip install -e ".[dev]"
pytest                                    # 532 tests
ruff check src tests scripts examples
mypy
python examples/afl_tuning.py             # hyperparameter search
python examples/afl_explore.py --plot out/
python scripts/build_afl_dataset.py       # rebuild the bundled data
```

See [TUTORIAL.md](TUTORIAL.md) for a guided tour of the API,
[CONTRIBUTING.md](CONTRIBUTING.md) for the conventions worth knowing, and
[CHANGELOG.md](CHANGELOG.md) for what has changed.

## Licence

GPL-3.0-or-later.
