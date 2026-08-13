"""
Implied-volatility surface fitting and interpolation.

Pure calculation, like the rest of ratecraft: you bring a normalized option
chain as a frame, this module fits a surface and evaluates it. Nothing here
reads a file, opens a socket, or knows where the quotes came from.

Two smile models are available behind one interface:

- **SVI** (``model="svi"``) — the raw parameterization of Gatheral's stochastic
  volatility inspired form, fitted per expiry in (log-moneyness, total
  variance). The default; the better fit for equity-style chains.
- **SABR** (``model="sabr"``) — Hagan's lognormal expansion with ``beta`` fixed,
  fitted per expiry. The market convention for interest-rate options, and the
  language the swaption/futures-option literature is written in.

Across expiries the surface interpolates **linearly in total variance**, not in
volatility, for both models. That choice is not cosmetic: interpolating in
volatility is the classic way to manufacture calendar arbitrage between two
individually clean expiries, whereas linear-in-total-variance between two
ordered slices is non-decreasing in tenor by construction. Outside the fitted
tenor range the surface extrapolates at constant volatility (total variance
proportional to tenor), which is likewise non-decreasing.

Both arbitrage conditions are *checkable*, which is why they are the design
constraint:

- **Butterfly** — Durrleman's condition ``g(k) >= 0``, evaluated from the
  slice's total variance and its first two derivatives (analytic for SVI,
  central differences otherwise).
- **Calendar** — total variance non-decreasing along the expiry axis at fixed
  log-moneyness.

:func:`VolSurface.check_arbitrage` returns the violations it finds, so a fit
either satisfies these or it does not and a test can say which.

Conventions
-----------
Log-moneyness is ``k = log(strike / forward)``; total variance is
``w(k, T) = iv(k, T)**2 * T``. Option prices, where the module touches them at
all, are Black-76 (forward-based) — the natural convention when the chain
already carries a forward.

There is deliberately no caching: compute on read. If a *measured* latency
problem ever justifies one, key it with :data:`FITTER_VERSION` (also carried on
every fitted surface) so a change to the fitter cannot silently serve a stale
surface.

Scope: the fit, the interpolation, the arbitrage checks. Acquiring chains,
storing surfaces and serving them belong to whichever service consumes this.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy.optimize import brentq, least_squares
from scipy.stats import norm

logger = logging.getLogger(__name__)

__all__ = [
    "FITTER_VERSION",
    "ArbitrageViolation",
    "SABRParams",
    "SABRSlice",
    "SVIParams",
    "SVISlice",
    "VolSlice",
    "VolSurface",
    "black76_price",
    "durrleman_g",
    "fit_surface",
    "implied_vol_black76",
    "sabr_lognormal_vol",
    "svi_total_variance",
]

#: Bumped whenever a change to the fitters can move a fitted surface. Carried on
#: every :class:`VolSurface`; the intended cache key if caching is ever added.
FITTER_VERSION = 1

#: Slack allowed on the arbitrage inequalities, to keep floating-point noise on
#: a numerically clean fit from reading as a violation.
ARBITRAGE_TOL = 1e-8

_MODELS = ("svi", "sabr")

# Free parameters per model, hence the minimum quotes needed for a determined
# per-expiry fit: SVI has (a, b, rho, m, sigma); SABR has (alpha, rho, nu) with
# beta held fixed, since beta is not identifiable from a single smile.
_MIN_QUOTES = {"svi": 5, "sabr": 3}


# ---------------------------------------------------------------------------
# Black-76
# ---------------------------------------------------------------------------


def _is_call(option_type) -> np.ndarray:
    """Normalize an option-type label (``"C"``, ``"call"``, ``"P"``, ...) to a bool."""
    kinds = np.asarray(option_type, dtype=object)
    flat = np.array([str(x).strip().upper()[:1] if x is not None else "" for x in kinds.ravel()])
    if not np.isin(flat, ("C", "P")).all():
        bad = sorted(set(flat[~np.isin(flat, ("C", "P"))]))
        raise ValueError(f"option_type must be call/put (got {bad})")
    return (flat == "C").reshape(kinds.shape)


def black76_price(forward, strike, tenor, vol, option_type="C", discount=1.0):
    """
    Black-76 price of a European option on a forward.

    Args:
        forward: Forward (or futures) price of the underlying.
        strike: Strike.
        tenor: Time to expiry in years.
        vol: Lognormal (Black) volatility.
        option_type: ``"C"``/``"call"`` or ``"P"``/``"put"``.
        discount: Discount factor from expiry to the valuation date. Pass 1.0
            for a forward premium (undiscounted) quote.

    Returns:
        The option price; a float for scalar inputs, otherwise a numpy array
        broadcast over the inputs.
    """
    f = np.asarray(forward, dtype=float)
    k = np.asarray(strike, dtype=float)
    t = np.asarray(tenor, dtype=float)
    v = np.asarray(vol, dtype=float)
    d = np.asarray(discount, dtype=float)
    if np.any(f <= 0) or np.any(k <= 0):
        raise ValueError("forward and strike must be positive")

    # Floor the total standard deviation rather than branching: as it goes to
    # zero d1/d2 run off to +/-inf and the price collapses to the intrinsic.
    std = np.maximum(v * np.sqrt(np.maximum(t, 0.0)), 1e-300)
    d1 = (np.log(f / k) + 0.5 * std**2) / std
    d2 = d1 - std
    call = d * (f * norm.cdf(d1) - k * norm.cdf(d2))
    put = d * (k * norm.cdf(-d2) - f * norm.cdf(-d1))

    price = np.where(_is_call(option_type), call, put)
    return float(price) if price.ndim == 0 else price


def implied_vol_black76(
    price,
    forward,
    strike,
    tenor,
    option_type="C",
    discount=1.0,
    *,
    vol_bounds: tuple[float, float] = (1e-6, 5.0),
) -> float:
    """
    Back out the Black-76 implied volatility of a single option quote.

    Scalar only — inverting a frame is a row-wise job and doing it here would
    hide which row failed.

    Args:
        price: Observed option price, on the same discounting basis as
            ``discount``.
        forward: Forward price of the underlying.
        strike: Strike.
        tenor: Time to expiry in years.
        option_type: ``"C"``/``"call"`` or ``"P"``/``"put"``.
        discount: Discount factor from expiry to the valuation date.
        vol_bounds: Bracket searched for the root.

    Returns:
        The implied volatility, or ``nan`` when the quote is not invertible —
        expiry in the past, or a price outside the no-arbitrage bounds implied
        by ``vol_bounds``. Returning ``nan`` rather than raising lets a caller
        drop the handful of stale quotes every real chain carries;
        :func:`fit_surface` does exactly that.
    """
    price = float(price)
    tenor = float(tenor)
    if not np.isfinite(price) or tenor <= 0:
        return float("nan")

    lo, hi = vol_bounds

    def objective(v: float) -> float:
        return float(black76_price(forward, strike, tenor, v, option_type, discount)) - price

    f_lo, f_hi = objective(lo), objective(hi)
    if f_lo > 0 or f_hi < 0:  # below intrinsic, or richer than vol_bounds[1] allows
        return float("nan")
    return float(brentq(objective, lo, hi, xtol=1e-12, rtol=1e-10, maxiter=200))


# ---------------------------------------------------------------------------
# Smile models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SVIParams:
    """Raw-SVI parameters: ``w(k) = a + b * (rho * (k - m) + sqrt((k - m)**2 + sigma**2))``."""

    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def __post_init__(self) -> None:
        if self.b < 0:
            raise ValueError(f"SVI b must be non-negative (got {self.b})")
        if not -1 < self.rho < 1:
            raise ValueError(f"SVI rho must lie in (-1, 1) (got {self.rho})")
        if self.sigma <= 0:
            raise ValueError(f"SVI sigma must be positive (got {self.sigma})")
        if self.min_total_variance < -ARBITRAGE_TOL:
            raise ValueError(f"SVI parameters imply negative total variance ({self.min_total_variance:.6g})")

    @property
    def min_total_variance(self) -> float:
        """The smile's minimum total variance, attained at ``m - rho*sigma/sqrt(1-rho**2)``."""
        return self.a + self.b * self.sigma * np.sqrt(1.0 - self.rho**2)


@dataclass(frozen=True)
class SABRParams:
    """SABR parameters under Hagan's lognormal expansion. ``beta`` is held fixed when fitting."""

    alpha: float
    beta: float
    rho: float
    nu: float

    def __post_init__(self) -> None:
        if self.alpha <= 0:
            raise ValueError(f"SABR alpha must be positive (got {self.alpha})")
        if not 0 <= self.beta <= 1:
            raise ValueError(f"SABR beta must lie in [0, 1] (got {self.beta})")
        if not -1 < self.rho < 1:
            raise ValueError(f"SABR rho must lie in (-1, 1) (got {self.rho})")
        if self.nu < 0:
            raise ValueError(f"SABR nu must be non-negative (got {self.nu})")


def svi_total_variance(log_moneyness, params: SVIParams):
    """Total variance ``w(k)`` of a raw-SVI smile."""
    k = np.asarray(log_moneyness, dtype=float)
    x = k - params.m
    return params.a + params.b * (params.rho * x + np.sqrt(x**2 + params.sigma**2))


def svi_derivatives(log_moneyness, params: SVIParams):
    """``(w, dw/dk, d2w/dk2)`` of a raw-SVI smile, analytically."""
    k = np.asarray(log_moneyness, dtype=float)
    x = k - params.m
    root = np.sqrt(x**2 + params.sigma**2)
    w = params.a + params.b * (params.rho * x + root)
    dw = params.b * (params.rho + x / root)
    d2w = params.b * params.sigma**2 / root**3
    return w, dw, d2w


def sabr_lognormal_vol(forward, strike, tenor, params: SABRParams):
    """
    Hagan's lognormal (Black) implied volatility for SABR.

    The usual second-order expansion. The ``z / x(z)`` factor is taken to its
    limit of 1 near the money, which also recovers Hagan's ATM formula.
    """
    f = np.asarray(forward, dtype=float)
    k = np.asarray(strike, dtype=float)
    t = np.asarray(tenor, dtype=float)
    if np.any(f <= 0) or np.any(k <= 0):
        raise ValueError("forward and strike must be positive")

    alpha, beta, rho, nu = params.alpha, params.beta, params.rho, params.nu
    one_minus_beta = 1.0 - beta
    fk = f * k
    fk_mid = fk ** (one_minus_beta / 2.0)
    log_fk = np.log(f / k)

    z = (nu / alpha) * fk_mid * log_fk
    near_atm = np.abs(z) < 1e-9
    x_of_z = np.log((np.sqrt(1.0 - 2.0 * rho * z + z**2) + z - rho) / (1.0 - rho))
    ratio = np.where(near_atm, 1.0, z / np.where(near_atm, 1.0, x_of_z))

    denominator = fk_mid * (1.0 + one_minus_beta**2 / 24.0 * log_fk**2 + one_minus_beta**4 / 1920.0 * log_fk**4)
    correction = 1.0 + (
        one_minus_beta**2 / 24.0 * alpha**2 / fk**one_minus_beta
        + rho * beta * nu * alpha / (4.0 * fk_mid)
        + (2.0 - 3.0 * rho**2) / 24.0 * nu**2
    ) * t
    return alpha / denominator * ratio * correction


# ---------------------------------------------------------------------------
# Slices
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VolSlice:
    """
    One expiry's fitted smile.

    Attributes:
        tenor: Time to expiry in years.
        forward: Forward level the smile is quoted against.
        n_quotes: Number of quotes the fit consumed.
        rmse: Root-mean-square fit error, in volatility points.
        k_min: Lowest fitted log-moneyness (the quoted range, used to pick a
            sensible default grid for the arbitrage checks).
        k_max: Highest fitted log-moneyness.
    """

    tenor: float
    forward: float
    n_quotes: int
    rmse: float
    k_min: float
    k_max: float

    def total_variance(self, log_moneyness):
        """Total variance ``w(k)`` at this expiry."""
        raise NotImplementedError

    def derivatives(self, log_moneyness, h: float = 1e-3):
        """``(w, dw/dk, d2w/dk2)``; central differences unless a subclass knows better."""
        k = np.asarray(log_moneyness, dtype=float)
        w = self.total_variance(k)
        up = self.total_variance(k + h)
        down = self.total_variance(k - h)
        return w, (up - down) / (2.0 * h), (up - 2.0 * w + down) / h**2

    def iv(self, log_moneyness):
        """Implied volatility at this expiry."""
        w = np.maximum(self.total_variance(log_moneyness), 0.0)
        return np.sqrt(w / self.tenor)


@dataclass(frozen=True)
class SVISlice(VolSlice):
    """A raw-SVI smile."""

    params: SVIParams = SVIParams(a=0.0, b=0.0, rho=0.0, m=0.0, sigma=1.0)

    def total_variance(self, log_moneyness):
        return svi_total_variance(log_moneyness, self.params)

    def derivatives(self, log_moneyness, h: float = 1e-3):
        return svi_derivatives(log_moneyness, self.params)


@dataclass(frozen=True)
class SABRSlice(VolSlice):
    """A SABR smile. Strikes are reconstructed as ``forward * exp(k)``."""

    params: SABRParams = SABRParams(alpha=0.2, beta=0.5, rho=0.0, nu=0.0)

    def total_variance(self, log_moneyness):
        k = np.asarray(log_moneyness, dtype=float)
        vol = sabr_lognormal_vol(self.forward, self.forward * np.exp(k), self.tenor, self.params)
        return vol**2 * self.tenor


# ---------------------------------------------------------------------------
# Arbitrage
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArbitrageViolation:
    """
    One arbitrage condition failing at one point of the surface.

    Attributes:
        kind: ``"butterfly"`` or ``"calendar"``.
        tenor: Where it fails. For a calendar violation, the *later* of the two
            tenors compared.
        log_moneyness: Where it fails.
        value: How badly, signed and negative: Durrleman's ``g(k)`` for a
            butterfly violation, the total-variance decrement for a calendar
            one.
    """

    kind: str
    tenor: float
    log_moneyness: float
    value: float


def durrleman_g(log_moneyness, w, dw, d2w):
    """
    Durrleman's function ``g(k)``, whose non-negativity is the butterfly
    condition.

    Up to a positive factor it *is* the risk-neutral density implied by the
    smile, so ``g(k) < 0`` means a negative density — and a butterfly spread
    that prices below zero.

    Args:
        log_moneyness: Where it is evaluated.
        w: Total variance there.
        dw: ``dw/dk`` there.
        d2w: ``d2w/dk2`` there.
    """
    k = np.asarray(log_moneyness, dtype=float)
    w = np.asarray(w, dtype=float)
    return (1.0 - k * dw / (2.0 * w)) ** 2 - (dw**2 / 4.0) * (1.0 / w + 0.25) + d2w / 2.0


# ---------------------------------------------------------------------------
# Surface
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VolSurface:
    """
    A fitted implied-volatility surface: per-expiry smiles plus an
    interpolation rule that is linear in total variance across expiries.

    Attributes:
        slices: The fitted smiles, ascending in tenor.
        model: ``"svi"`` or ``"sabr"``.
        fitter_version: :data:`FITTER_VERSION` at fit time.
    """

    slices: tuple[VolSlice, ...]
    model: str
    fitter_version: int = FITTER_VERSION

    def __post_init__(self) -> None:
        if not self.slices:
            raise ValueError("a surface needs at least one fitted expiry")
        tenors = [s.tenor for s in self.slices]
        if any(t <= 0 for t in tenors):
            raise ValueError("slice tenors must be positive")
        if any(b <= a for a, b in zip(tenors, tenors[1:], strict=False)):
            raise ValueError("slices must be strictly ascending in tenor")

    @property
    def tenors(self) -> np.ndarray:
        """The fitted tenors, ascending."""
        return np.array([s.tenor for s in self.slices], dtype=float)

    def total_variance(self, log_moneyness, tenor):
        """
        Total variance ``w(k, T)``.

        Between fitted expiries this is linear in total variance; outside them
        it holds volatility flat, so ``w`` stays proportional to ``T``. Both
        rules are non-decreasing in ``T``, which is exactly the calendar
        condition.
        """
        k_in = np.asarray(log_moneyness, dtype=float)
        t_in = np.asarray(tenor, dtype=float)
        k, t = np.broadcast_arrays(k_in, t_in)
        shape = k.shape
        scalar = shape == ()
        k, t = np.atleast_1d(k).ravel(), np.atleast_1d(t).ravel()
        if np.any(t <= 0):
            raise ValueError("tenor must be positive")

        # Every slice evaluated at every point: the slice count is the number of
        # quoted expiries, so this is cheap and keeps the indexing legible.
        by_slice = np.array([s.total_variance(k) for s in self.slices], dtype=float)
        ts = self.tenors
        cols = np.arange(k.size)

        if len(ts) == 1:
            out = by_slice[0] * (t / ts[0])
        else:
            i = np.clip(np.searchsorted(ts, t, side="right") - 1, 0, len(ts) - 2)
            lo, hi = by_slice[i, cols], by_slice[i + 1, cols]
            frac = (t - ts[i]) / (ts[i + 1] - ts[i])
            out = lo + frac * (hi - lo)
            out = np.where(t < ts[0], by_slice[0, cols] * (t / ts[0]), out)
            out = np.where(t > ts[-1], by_slice[-1, cols] * (t / ts[-1]), out)

        out = out.reshape(shape)
        return float(out) if scalar else out

    def iv(self, log_moneyness, tenor):
        """
        Implied volatility at a log-moneyness and tenor.

        Args:
            log_moneyness: ``log(strike / forward)``; scalar or array.
            tenor: Time to expiry in years; scalar or array broadcastable
                against ``log_moneyness``.

        Returns:
            The implied volatility, a float for scalar inputs.
        """
        w = self.total_variance(log_moneyness, tenor)
        return np.sqrt(np.maximum(w, 0.0) / np.asarray(tenor, dtype=float))

    def forward(self, tenor) -> float:
        """The forward of the nearest fitted expiry — what ``log_moneyness`` is measured against."""
        idx = int(np.argmin(np.abs(self.tenors - float(tenor))))
        return float(self.slices[idx].forward)

    def check_arbitrage(
        self,
        log_moneyness=None,
        tenors=None,
        *,
        tol: float = ARBITRAGE_TOL,
    ) -> list[ArbitrageViolation]:
        """
        Check the surface for butterfly and calendar arbitrage.

        Butterfly is Durrleman's ``g(k) >= 0``, evaluated at every tenor in
        ``tenors``; the default grid is the fitted tenors *and their midpoints*,
        because a linear-in-total-variance blend of two butterfly-free slices is
        not itself guaranteed butterfly-free. Calendar is total variance
        non-decreasing in tenor, checked on consecutive fitted slices — which is
        sufficient for the whole surface, since the interpolation and
        extrapolation rules are both monotone in tenor.

        Args:
            log_moneyness: Grid to check. Defaults to 101 points spanning the
                fitted quote range, widened by 10% at each end.
            tenors: Tenors for the butterfly check. Defaults as described above.
            tol: Slack on both inequalities, to absorb floating-point noise.

        Returns:
            The violations found, worst first. An empty list means the surface
            is clean on the grid checked — which is a statement about the grid,
            not a proof over all of R.
        """
        k = np.asarray(log_moneyness, dtype=float).ravel() if log_moneyness is not None else self._default_k_grid()
        violations: list[ArbitrageViolation] = []

        check_tenors = self._default_tenor_grid() if tenors is None else np.asarray(tenors, dtype=float).ravel()
        for t in check_tenors:
            g = durrleman_g(k, *self._derivatives(k, float(t)))
            for idx in np.flatnonzero(g < -tol):
                violations.append(ArbitrageViolation("butterfly", float(t), float(k[idx]), float(g[idx])))

        for earlier, later in zip(self.slices, self.slices[1:], strict=False):
            gap = later.total_variance(k) - earlier.total_variance(k)
            for idx in np.flatnonzero(gap < -tol):
                violations.append(ArbitrageViolation("calendar", float(later.tenor), float(k[idx]), float(gap[idx])))

        return sorted(violations, key=lambda v: v.value)

    def _derivatives(self, k, tenor: float, h: float = 1e-3):
        """``(w, dw/dk, d2w/dk2)`` at a tenor — exactly from the slice where one sits, else numerically."""
        for candidate in self.slices:
            if candidate.tenor == tenor:
                return candidate.derivatives(k)
        w = self.total_variance(k, tenor)
        up = self.total_variance(k + h, tenor)
        down = self.total_variance(k - h, tenor)
        return w, (up - down) / (2.0 * h), (up - 2.0 * w + down) / h**2

    def _default_k_grid(self) -> np.ndarray:
        lo = min(s.k_min for s in self.slices)
        hi = max(s.k_max for s in self.slices)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = -0.5, 0.5
        pad = 0.1 * (hi - lo)
        return np.linspace(lo - pad, hi + pad, 101)

    def _default_tenor_grid(self) -> np.ndarray:
        ts = self.tenors
        midpoints = 0.5 * (ts[:-1] + ts[1:])
        return np.unique(np.concatenate([ts, midpoints]))


# ---------------------------------------------------------------------------
# Chain normalization
# ---------------------------------------------------------------------------


def _tenor_years(frame: pd.DataFrame, asof, day_count: float) -> pd.Series:
    if "tenor" in frame.columns:
        return pd.to_numeric(frame["tenor"], errors="coerce").astype(float)
    if "expiry" not in frame.columns:
        raise ValueError("chain needs a 'tenor' column (years) or an 'expiry' column with asof=")
    if asof is None:
        raise ValueError("chain has 'expiry' dates but no asof= to measure them from")

    expiry = pd.to_datetime(frame["expiry"])
    anchor = pd.Timestamp(asof)
    tz = getattr(expiry.dt, "tz", None)
    if tz is not None and anchor.tz is None:
        anchor = anchor.tz_localize(tz)
    elif tz is None and anchor.tz is not None:
        anchor = anchor.tz_localize(None)
    return (expiry - anchor).dt.total_seconds() / (86400.0 * day_count)


def _implied_vols(frame: pd.DataFrame) -> pd.Series:
    if "iv" in frame.columns:
        return pd.to_numeric(frame["iv"], errors="coerce").astype(float)
    if "price" not in frame.columns:
        raise ValueError("chain needs an 'iv' column, or 'price' + 'option_type' to imply one")
    if "option_type" not in frame.columns:
        raise ValueError("implying vol from 'price' also needs an 'option_type' column")

    discount = frame["discount"].to_numpy() if "discount" in frame.columns else np.ones(len(frame))
    quotes = zip(
        frame["price"].to_numpy(),
        frame["forward"].to_numpy(),
        frame["strike"].to_numpy(),
        frame["tenor"].to_numpy(),
        frame["option_type"].to_numpy(),
        discount,
        strict=True,
    )
    return pd.Series([implied_vol_black76(*quote) for quote in quotes], index=frame.index, dtype=float)


def _normalize_chain(chain: pd.DataFrame, asof, day_count: float) -> pd.DataFrame:
    """Reduce a caller's chain to the columns the fitters need: tenor, forward, k, iv, weight."""
    if not isinstance(chain, pd.DataFrame):
        raise TypeError(f"chain must be a pandas DataFrame (got {type(chain).__name__})")

    frame = chain.copy()
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    missing = [c for c in ("strike", "forward") if c not in frame.columns]
    if missing:
        raise ValueError(f"chain is missing required column(s): {missing}")

    frame["strike"] = pd.to_numeric(frame["strike"], errors="coerce").astype(float)
    frame["forward"] = pd.to_numeric(frame["forward"], errors="coerce").astype(float)
    frame["tenor"] = _tenor_years(frame, asof, day_count)
    frame["iv"] = _implied_vols(frame)
    frame["weight"] = (
        pd.to_numeric(frame["weight"], errors="coerce").astype(float) if "weight" in frame.columns else 1.0
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        frame["k"] = np.log(frame["strike"] / frame["forward"])

    usable = (
        np.isfinite(frame[["tenor", "forward", "strike", "iv", "k", "weight"]]).all(axis=1)
        & (frame["tenor"] > 0)
        & (frame["iv"] > 0)
        & (frame["weight"] > 0)
    )
    dropped = int((~usable).sum())
    if dropped:
        logger.warning("dropping %d of %d chain rows: not invertible, non-positive, or expired", dropped, len(frame))
    frame = frame.loc[usable, ["tenor", "forward", "strike", "k", "iv", "weight"]]
    if frame.empty:
        raise ValueError("chain has no usable quotes after cleaning")
    return frame.sort_values(["tenor", "k"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def _svi_from_free(free: np.ndarray) -> SVIParams:
    """
    Rebuild SVI parameters from the fitting parameterization.

    We fit ``(w_min, b, rho, m, sigma)`` rather than ``(a, b, rho, m, sigma)``
    and recover ``a`` from them, because the smile's minimum total variance is
    ``a + b*sigma*sqrt(1-rho^2)``: bounding ``w_min >= 0`` is a plain box
    constraint, whereas the same requirement on ``a`` is a nonlinear one that
    ``least_squares`` cannot take.
    """
    w_min, b, rho, m, sigma = free
    return SVIParams(a=w_min - b * sigma * np.sqrt(1.0 - rho**2), b=b, rho=rho, m=m, sigma=sigma)


def _fit_svi(k: np.ndarray, w: np.ndarray, weights: np.ndarray) -> SVIParams:
    span = max(float(k.max() - k.min()), 1e-3)
    bounds = (
        [0.0, 0.0, -0.999, float(k.min()) - span, 1e-4],
        [10.0, 100.0, 0.999, float(k.max()) + span, 10.0],
    )
    w_min0 = float(np.clip(w.min(), 0.0, 10.0))
    b0 = float(np.clip((w.max() - w.min()) / span, 1e-3, 100.0))
    m0 = float(k[int(np.argmin(w))])
    root_weights = np.sqrt(weights)

    def residuals(free: np.ndarray) -> np.ndarray:
        w_min, b, rho, m, sigma = free
        a = w_min - b * sigma * np.sqrt(1.0 - rho**2)
        model = a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma**2))
        return root_weights * (model - w)

    # A few starts: the SVI objective has local minima, and rho in particular
    # will sit in whichever sign of skew it was started near.
    best = None
    for rho0 in (-0.6, 0.0, 0.6):
        for sigma0 in (0.1, 0.5):
            start = np.clip([w_min0, b0, rho0, m0, sigma0], bounds[0], bounds[1])
            fit = least_squares(residuals, start, bounds=bounds, method="trf", max_nfev=5000)
            if best is None or fit.cost < best.cost:
                best = fit
    return _svi_from_free(best.x)


def _fit_sabr(strikes: np.ndarray, ivs: np.ndarray, forward: float, tenor: float, beta: float, weights) -> SABRParams:
    atm = float(ivs[int(np.argmin(np.abs(strikes - forward)))])
    alpha0 = float(np.clip(atm * forward ** (1.0 - beta), 1e-4, 10.0))
    bounds = ([1e-6, -0.999, 1e-6], [10.0, 0.999, 10.0])
    root_weights = np.sqrt(weights)

    def residuals(free: np.ndarray) -> np.ndarray:
        alpha, rho, nu = free
        model = sabr_lognormal_vol(forward, strikes, tenor, SABRParams(alpha=alpha, beta=beta, rho=rho, nu=nu))
        return root_weights * (model - ivs)

    best = None
    for rho0 in (-0.3, 0.0, 0.3):
        for nu0 in (0.2, 0.8):
            start = np.clip([alpha0, rho0, nu0], bounds[0], bounds[1])
            fit = least_squares(residuals, start, bounds=bounds, method="trf", max_nfev=5000)
            if best is None or fit.cost < best.cost:
                best = fit
    alpha, rho, nu = best.x
    return SABRParams(alpha=alpha, beta=beta, rho=rho, nu=nu)


def _fit_slice(group: pd.DataFrame, model: str, beta: float) -> VolSlice:
    tenor = float(group["tenor"].iloc[0])
    if len(group) < _MIN_QUOTES[model]:
        raise ValueError(
            f"expiry T={tenor:.4f} has {len(group)} quote(s); {model} needs at least {_MIN_QUOTES[model]}"
        )

    forwards = group["forward"].to_numpy(dtype=float)
    forward = float(forwards.mean())
    if np.ptp(forwards) > 1e-6 * max(abs(forward), 1.0):
        logger.warning("expiry T=%.4f quotes %d distinct forwards; using their mean %.6g", tenor, len(set(forwards)), forward)

    k = group["k"].to_numpy(dtype=float)
    ivs = group["iv"].to_numpy(dtype=float)
    weights = group["weight"].to_numpy(dtype=float)

    common = {
        "tenor": tenor,
        "forward": forward,
        "n_quotes": len(group),
        "rmse": float("nan"),
        "k_min": float(k.min()),
        "k_max": float(k.max()),
    }
    if model == "svi":
        fitted: VolSlice = SVISlice(params=_fit_svi(k, ivs**2 * tenor, weights), **common)
    else:
        strikes = group["strike"].to_numpy(dtype=float)
        fitted = SABRSlice(params=_fit_sabr(strikes, ivs, forward, tenor, beta, weights), **common)

    residual = np.asarray(fitted.iv(k), dtype=float) - ivs
    return replace(fitted, rmse=float(np.sqrt(np.mean(residual**2))))


def fit_surface(
    chain: pd.DataFrame,
    model: str = "svi",
    *,
    asof: dt.date | dt.datetime | pd.Timestamp | None = None,
    beta: float = 0.5,
    day_count: float = 365.0,
) -> VolSurface:
    """
    Fit an implied-volatility surface to an option chain.

    Each expiry is fitted independently; the surface then interpolates across
    expiries linearly in total variance (see the module docstring for why).

    The chain is a plain frame — callers already have frames, so there is no
    bespoke chain class to construct. Column names are matched
    case-insensitively:

    ==================  ==========================================================
    ``strike``          required
    ``forward``         required; the forward this expiry's smile is quoted against
    ``tenor``           years to expiry — or ``expiry`` (dates) together with ``asof=``
    ``iv``              implied volatility — or ``price`` + ``option_type``
                        (and optionally ``discount``) to imply one via Black-76
    ``weight``          optional relative fit weight, default 1 (e.g. vega)
    ==================  ==========================================================

    Rows that cannot be used — an expiry in the past, a price outside the
    no-arbitrage bounds, a non-positive vol — are dropped with a logged warning
    rather than failing the fit.

    Args:
        chain: The option chain, as described above.
        model: ``"svi"`` (default) or ``"sabr"``.
        asof: Valuation date, required only when the chain carries ``expiry``
            dates rather than a ``tenor``.
        beta: SABR's CEV exponent, held fixed during the fit because a single
            smile does not identify it. Ignored for SVI.
        day_count: Days per year used to convert ``expiry`` dates to tenors.

    Returns:
        The fitted :class:`VolSurface`.

    Raises:
        ValueError: If the chain lacks the columns needed, has no usable quotes,
            or has an expiry with fewer quotes than the model has free
            parameters.
    """
    model = str(model).strip().lower()
    if model not in _MODELS:
        raise ValueError(f"model must be one of {_MODELS} (got {model!r})")

    frame = _normalize_chain(chain, asof, day_count)
    slices = tuple(_fit_slice(group, model, beta) for _, group in frame.groupby("tenor", sort=True))
    return VolSurface(slices=slices, model=model, fitter_version=FITTER_VERSION)
