"""Turning B-scores into win probabilities.

The paper does not read a probability straight off the centralities.  It feeds
them to a logit (Eq. 3):

.. math::

    p_{i,j,t+1} = \\frac{\\exp(\\beta_0 + \\beta_1 CR_{i,t} + \\beta_2 CR_{j,t})}
                       {1 + \\exp(\\beta_0 + \\beta_1 CR_{i,t} + \\beta_2 CR_{j,t})}

Keeping :math:`\\beta_1` and :math:`\\beta_2` separate (rather than forcing
:math:`\\beta_2 = -\\beta_1`) matters for a sport with a fixed home/away
asymmetry: the intercept absorbs home advantage and the two slopes are free to
differ.  Pass ``symmetric=True`` for a venue-neutral sport.

The fit is Newton/IRLS with an L2 penalty and step halving — no SciPy or
scikit-learn required, and it accepts fractional outcomes so a draw can enter
as ``0.5``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

__all__ = ["LogisticFit", "fit_logistic", "sigmoid", "LogitCalibrator"]

Transform = Literal["identity", "log", "sqrt"]


def sigmoid(z: Any) -> np.ndarray:
    """Numerically stable logistic function."""
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def _log_likelihood(design: np.ndarray, y: np.ndarray, w: np.ndarray, beta: np.ndarray) -> float:
    z = design @ beta
    # log(sigmoid(z)) == -log1p(exp(-z)); the max/abs form keeps it stable.
    log_p = -np.logaddexp(0.0, -z)
    log_q = -np.logaddexp(0.0, z)
    return float(np.sum(w * (y * log_p + (1.0 - y) * log_q)))


@dataclass(frozen=True)
class LogisticFit:
    """Result of :func:`fit_logistic`."""

    beta: np.ndarray
    """Coefficients, one per design column."""

    n_iter: int
    """Newton steps taken."""

    converged: bool
    """Whether the gradient tolerance was met."""

    log_likelihood: float
    """Weighted log-likelihood at the optimum (penalty excluded)."""

    n_obs: int = 0
    """Number of observations used."""

    penalty: float = 0.0
    """L2 penalty strength applied."""


def fit_logistic(
    design: np.ndarray,
    y: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
    penalize: np.ndarray | None = None,
    ridge: float = 1e-6,
    max_iter: int = 100,
    tol: float = 1e-10,
    beta0: np.ndarray | None = None,
) -> LogisticFit:
    """Fit a logistic regression by penalised IRLS.

    Parameters
    ----------
    design
        ``(n_obs, n_features)`` design matrix, intercept column included by the
        caller if wanted.
    y
        Targets in ``[0, 1]``.  Fractional values are valid: they are read as
        the Bernoulli mean, which is how a draw scored ``0.5`` enters.
    sample_weight
        Optional non-negative observation weights.
    penalize
        Boolean mask of columns the ridge applies to.  Defaults to every column
        (pass an explicit mask to leave an intercept unpenalised).
    ridge
        L2 strength.  A small non-zero default keeps the fit finite under
        complete separation, which happens whenever an early training window
        contains only lopsided results.
    max_iter, tol
        Newton iteration budget and gradient-infinity-norm tolerance.
    beta0
        Warm start.

    Returns
    -------
    LogisticFit
    """
    X = np.asarray(design, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"design must be 2-d, got shape {X.shape}")
    target = np.asarray(y, dtype=np.float64).ravel()
    if target.shape[0] != X.shape[0]:
        raise ValueError(f"design/target length mismatch: {X.shape[0]} vs {target.shape[0]}")
    if target.size == 0:
        raise ValueError("cannot fit a logistic model to zero observations")
    if np.any(target < 0.0) or np.any(target > 1.0):
        raise ValueError("targets must lie in [0, 1]")
    if not np.all(np.isfinite(X)):
        raise ValueError("design contains non-finite values")

    n_obs, n_features = X.shape
    if sample_weight is None:
        w = np.ones(n_obs, dtype=np.float64)
    else:
        w = np.asarray(sample_weight, dtype=np.float64).ravel()
        if w.shape != target.shape:
            raise ValueError(f"sample_weight length mismatch: {w.shape} vs {target.shape}")
        if np.any(w < 0.0):
            raise ValueError("sample_weight must be non-negative")

    if penalize is None:
        mask = np.ones(n_features, dtype=bool)
    else:
        mask = np.asarray(penalize, dtype=bool).ravel()
        if mask.shape != (n_features,):
            raise ValueError(f"penalize must have shape ({n_features},), got {mask.shape}")
    ridge_diag = ridge * mask.astype(np.float64)

    beta: np.ndarray = (
        np.zeros(n_features) if beta0 is None else np.array(beta0, dtype=np.float64).ravel()
    )
    if beta.shape != (n_features,):
        raise ValueError(f"beta0 must have shape ({n_features},)")

    loglik = _log_likelihood(X, target, w, beta) - 0.5 * float(ridge_diag @ (beta * beta))
    converged = False
    iteration = 0

    for iteration in range(1, max_iter + 1):  # noqa: B007 - reported as n_iter
        p = sigmoid(X @ beta)
        residual = w * (target - p)
        gradient = X.T @ residual - ridge_diag * beta
        if np.max(np.abs(gradient)) <= tol:
            converged = True
            break

        hess_w = w * p * (1.0 - p)
        hessian = (X.T * hess_w) @ X
        hessian[np.diag_indices(n_features)] += ridge_diag
        # A tiny jitter keeps the solve well posed when a feature is constant
        # or two are collinear in the current window.
        hessian[np.diag_indices(n_features)] += 1e-12

        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:  # pragma: no cover - jitter makes this rare
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]

        scale = 1.0
        for _ in range(30):
            candidate = beta + scale * step
            value = _log_likelihood(X, target, w, candidate) - 0.5 * float(
                ridge_diag @ (candidate * candidate)
            )
            if np.isfinite(value) and value >= loglik:
                break
            scale *= 0.5
        else:  # no improvement even at 2**-30: we are at the optimum
            converged = True
            break

        shift = np.max(np.abs(candidate - beta))
        beta, loglik = candidate, value
        if shift <= tol:
            converged = True
            break

    return LogisticFit(
        beta=beta,
        n_iter=iteration,
        converged=converged,
        log_likelihood=_log_likelihood(X, target, w, beta),
        n_obs=n_obs,
        penalty=ridge,
    )


def _transform(values: np.ndarray, kind: Transform, epsilon: float) -> np.ndarray:
    if kind == "identity":
        return values
    if kind == "log":
        return np.log(np.maximum(values, 0.0) + epsilon)
    if kind == "sqrt":
        return np.sqrt(np.maximum(values, 0.0))
    raise ValueError(f"unknown transform {kind!r}")


@dataclass
class LogitCalibrator:
    """Eq. 3: map a pair of B-scores onto a win probability.

    Parameters
    ----------
    fit_intercept
        Include :math:`\\beta_0`.  Keep it on for home/away sports — it is
        where home advantage lands.
    symmetric
        Constrain :math:`\\beta_2 = -\\beta_1`, i.e. regress on the score
        difference alone.  Appropriate when the two slots are interchangeable.
    transform
        Applied to both scores before the linear model.  ``"identity"`` is the
        paper's specification; ``"log"`` turns the model into a Bradley-Terry
        style comparison of score ratios.
    ridge
        L2 penalty, not applied to the intercept.
    epsilon
        Floor applied inside ``log`` so an unrated competitor (score 0) stays
        finite.  ``None`` (the default) resolves it at fit time to 1% of the
        mean positive training score, which keeps the floor on the same scale as
        the data.  A fixed tiny value like ``1e-12`` is a trap here: it maps
        every unrated competitor to roughly ``-28``, an outlier that swamps the
        fit.
    """

    fit_intercept: bool = True
    symmetric: bool = False
    transform: Transform = "identity"
    ridge: float = 1e-6
    epsilon: float | None = None
    max_iter: int = 100
    tol: float = 1e-10

    epsilon_: float = field(default=1e-12, init=False)

    beta_: np.ndarray | None = field(default=None, init=False)
    n_iter_: int = field(default=0, init=False)
    converged_: bool = field(default=False, init=False)
    log_likelihood_: float = field(default=float("nan"), init=False)
    n_obs_: int = field(default=0, init=False)
    n_extra_: int = field(default=0, init=False)

    # ------------------------------------------------------------------
    def design(
        self,
        home: Any,
        away: Any,
        extra: np.ndarray | None = None,
    ) -> np.ndarray:
        """Build the design matrix for a batch of matchups."""
        h = _transform(np.asarray(home, dtype=np.float64).ravel(), self.transform, self.epsilon_)
        a = _transform(np.asarray(away, dtype=np.float64).ravel(), self.transform, self.epsilon_)
        if h.shape != a.shape:
            raise ValueError(f"home/away length mismatch: {h.shape} vs {a.shape}")

        columns = []
        if self.fit_intercept:
            columns.append(np.ones_like(h))
        if self.symmetric:
            columns.append(h - a)
        else:
            columns.append(h)
            columns.append(a)
        if extra is not None:
            block = np.asarray(extra, dtype=np.float64)
            if block.ndim == 1:
                block = block[:, None]
            if block.shape[0] != h.shape[0]:
                raise ValueError(f"extra length mismatch: {block.shape[0]} vs {h.shape[0]}")
            columns.extend(block.T)
        return np.column_stack(columns)

    def _resolve_epsilon(self, home: Any, away: Any) -> float:
        """Pick the ``log`` floor, from the data when it was left unset."""
        if self.epsilon is not None:
            return float(self.epsilon)
        if self.transform != "log":
            return 1e-12
        scores = np.concatenate(
            [
                np.asarray(home, dtype=np.float64).ravel(),
                np.asarray(away, dtype=np.float64).ravel(),
            ]
        )
        positive = scores[scores > 0.0]
        if positive.size == 0:
            return 1e-12
        return max(1e-12, 0.01 * float(positive.mean()))

    @property
    def _penalize_mask(self) -> np.ndarray:
        n_slopes = 1 if self.symmetric else 2
        mask: np.ndarray = np.ones(n_slopes + self.n_extra_, dtype=bool)
        if self.fit_intercept:
            mask = np.concatenate([[False], mask])
        return mask

    def fit(
        self,
        home: Any,
        away: Any,
        outcome: Any,
        *,
        sample_weight: Any = None,
        extra: np.ndarray | None = None,
    ) -> LogitCalibrator:
        """Fit the coefficients.

        ``outcome`` is the probability-scale result for the *home* slot: 1 for a
        home win, 0 for an away win, 0.5 for a draw.
        """
        self.n_extra_ = 0 if extra is None else np.atleast_2d(np.asarray(extra).T).shape[0]
        self.epsilon_ = self._resolve_epsilon(home, away)
        X = self.design(home, away, extra)
        y = np.asarray(outcome, dtype=np.float64).ravel()
        result = fit_logistic(
            X,
            y,
            sample_weight=None if sample_weight is None else np.asarray(sample_weight),
            penalize=self._penalize_mask,
            ridge=self.ridge,
            max_iter=self.max_iter,
            tol=self.tol,
        )
        self.beta_ = result.beta
        self.n_iter_ = result.n_iter
        self.converged_ = result.converged
        self.log_likelihood_ = result.log_likelihood
        self.n_obs_ = result.n_obs
        return self

    def predict_proba(self, home: Any, away: Any, *, extra: np.ndarray | None = None) -> np.ndarray:
        """Probability that the *home* slot wins."""
        if self.beta_ is None:
            raise RuntimeError("calibrator is not fitted; call fit() first")
        X = self.design(home, away, extra)
        if X.shape[1] != self.beta_.shape[0]:
            raise ValueError(
                f"design has {X.shape[1]} columns but the fit has "
                f"{self.beta_.shape[0]} coefficients"
            )
        return sigmoid(X @ self.beta_)

    # ------------------------------------------------------------------
    @property
    def is_fitted(self) -> bool:
        return self.beta_ is not None

    @property
    def intercept_(self) -> float:
        """:math:`\\beta_0`, or 0 when the intercept is off."""
        if self.beta_ is None:
            raise RuntimeError("calibrator is not fitted; call fit() first")
        return float(self.beta_[0]) if self.fit_intercept else 0.0

    @property
    def coef_(self) -> np.ndarray:
        """Slopes, excluding the intercept."""
        if self.beta_ is None:
            raise RuntimeError("calibrator is not fitted; call fit() first")
        return self.beta_[1:] if self.fit_intercept else self.beta_

    def __repr__(self) -> str:
        if self.beta_ is None:
            return (
                f"LogitCalibrator(fit_intercept={self.fit_intercept}, "
                f"symmetric={self.symmetric}, transform={self.transform!r}, unfitted)"
            )
        coefficients = ", ".join(f"{b:.4g}" for b in self.beta_)
        return f"LogitCalibrator(beta=[{coefficients}], n_obs={self.n_obs_})"
