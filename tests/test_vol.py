"""Tests for ratecraft.vol — surface fit, interpolation, and the arbitrage checks."""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from ratecraft.vol import (
    FITTER_VERSION,
    SABRParams,
    SABRSlice,
    SVIParams,
    SVISlice,
    VolSurface,
    black76_price,
    durrleman_g,
    fit_surface,
    implied_vol_black76,
    sabr_lognormal_vol,
    svi_derivatives,
    svi_total_variance,
)

# A calendar-clean set of SVI smiles: `a` and `b` both rise with tenor, so total
# variance is non-decreasing along the expiry axis at every log-moneyness.
SVI_TRUTH = {
    0.25: SVIParams(a=0.010, b=0.050, rho=-0.35, m=0.02, sigma=0.15),
    0.50: SVIParams(a=0.020, b=0.071, rho=-0.35, m=0.02, sigma=0.15),
    1.00: SVIParams(a=0.042, b=0.100, rho=-0.35, m=0.02, sigma=0.15),
}

# Axel Vogt's parameters, the standard worked example of an SVI slice that fits
# a smile nicely and still admits butterfly arbitrage. Its density goes negative
# over roughly k in [0.64, 1.26], so the checks below quote a wing wide enough to
# reach it.
VOGT = SVIParams(a=-0.0410, b=0.1331, rho=0.3060, m=0.3586, sigma=0.4153)
VOGT_K_RANGE = {"k_min": -0.6, "k_max": 1.3}


def svi_chain(forward=100.0, n_strikes=15, k_range=0.4, truth=None):
    """A synthetic chain generated from known SVI parameters, one block per expiry."""
    truth = SVI_TRUTH if truth is None else truth
    rows = []
    for tenor, params in truth.items():
        for k in np.linspace(-k_range, k_range, n_strikes):
            rows.append(
                {
                    "strike": forward * np.exp(k),
                    "forward": forward,
                    "tenor": tenor,
                    "iv": float(np.sqrt(svi_total_variance(k, params) / tenor)),
                }
            )
    return pd.DataFrame(rows)


def sabr_chain(forward=0.04, params=SABRParams(alpha=0.03, beta=0.5, rho=-0.25, nu=0.45), tenors=(0.5, 1.0, 2.0)):
    """A synthetic rate-option chain generated from known SABR parameters."""
    rows = []
    for tenor in tenors:
        for k in np.linspace(-0.3, 0.3, 11):
            strike = forward * np.exp(k)
            rows.append(
                {
                    "strike": strike,
                    "forward": forward,
                    "tenor": tenor,
                    "iv": float(sabr_lognormal_vol(forward, strike, tenor, params)),
                }
            )
    return pd.DataFrame(rows)


class TestBlack76:
    def test_call_put_parity(self):
        call = black76_price(100.0, 105.0, 1.5, 0.22, "C", 0.95)
        put = black76_price(100.0, 105.0, 1.5, 0.22, "P", 0.95)
        assert call - put == pytest.approx(0.95 * (100.0 - 105.0))

    def test_zero_vol_is_intrinsic(self):
        assert black76_price(100.0, 90.0, 1.0, 0.0, "C") == pytest.approx(10.0)
        assert black76_price(100.0, 90.0, 1.0, 0.0, "P") == pytest.approx(0.0)

    def test_broadcasts_over_strikes(self):
        prices = black76_price(100.0, np.array([90.0, 100.0, 110.0]), 1.0, 0.2)
        assert prices.shape == (3,)
        assert np.all(np.diff(prices) < 0)  # calls fall as the strike rises

    def test_option_type_labels_are_flexible(self):
        assert black76_price(100.0, 100.0, 1.0, 0.2, "call") == black76_price(100.0, 100.0, 1.0, 0.2, "C")
        assert black76_price(100.0, 100.0, 1.0, 0.2, "put") == black76_price(100.0, 100.0, 1.0, 0.2, "P")

    def test_rejects_unknown_option_type(self):
        with pytest.raises(ValueError, match="call/put"):
            black76_price(100.0, 100.0, 1.0, 0.2, "straddle")


class TestImpliedVol:
    @pytest.mark.parametrize("option_type", ["C", "P"])
    @pytest.mark.parametrize("strike", [80.0, 100.0, 130.0])
    def test_round_trips(self, strike, option_type):
        price = black76_price(100.0, strike, 1.25, 0.23, option_type, 0.9)
        assert implied_vol_black76(price, 100.0, strike, 1.25, option_type, 0.9) == pytest.approx(0.23)

    def test_below_intrinsic_is_nan(self):
        assert np.isnan(implied_vol_black76(1.0, 100.0, 50.0, 1.0, "C"))

    def test_expired_is_nan(self):
        assert np.isnan(implied_vol_black76(5.0, 100.0, 100.0, 0.0, "C"))

    def test_richer_than_bracket_is_nan(self):
        assert np.isnan(implied_vol_black76(99.0, 100.0, 100.0, 1.0, "C"))


class TestSmileModels:
    def test_svi_derivatives_match_finite_differences(self):
        params = SVI_TRUTH[0.5]
        k = np.linspace(-0.5, 0.5, 21)
        _, dw, d2w = svi_derivatives(k, params)
        h = 1e-5
        up, down, mid = (
            svi_total_variance(k + h, params),
            svi_total_variance(k - h, params),
            svi_total_variance(k, params),
        )
        assert dw == pytest.approx((up - down) / (2 * h), abs=1e-7)
        assert d2w == pytest.approx((up - 2 * mid + down) / h**2, abs=1e-3)

    def test_svi_minimum_total_variance(self):
        params = SVI_TRUTH[1.0]
        k = np.linspace(-5.0, 5.0, 20001)
        assert svi_total_variance(k, params).min() == pytest.approx(params.min_total_variance, abs=1e-6)

    def test_svi_rejects_negative_variance(self):
        with pytest.raises(ValueError, match="negative total variance"):
            SVIParams(a=-1.0, b=0.1, rho=0.0, m=0.0, sigma=0.1)

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"b": -0.1}, "b must be non-negative"),
            ({"rho": 1.0}, "rho must lie"),
            ({"sigma": 0.0}, "sigma must be positive"),
        ],
    )
    def test_svi_parameter_bounds(self, kwargs, match):
        base = {"a": 0.04, "b": 0.1, "rho": 0.0, "m": 0.0, "sigma": 0.1}
        with pytest.raises(ValueError, match=match):
            SVIParams(**{**base, **kwargs})

    def test_sabr_atm_matches_hagan_closed_form(self):
        params = SABRParams(alpha=0.03, beta=0.5, rho=-0.25, nu=0.45)
        f, t = 0.04, 2.0
        expected = (
            params.alpha
            / f ** (1 - params.beta)
            * (
                1
                + (
                    (1 - params.beta) ** 2 / 24 * params.alpha**2 / f ** (2 - 2 * params.beta)
                    + params.rho * params.beta * params.nu * params.alpha / (4 * f ** (1 - params.beta))
                    + (2 - 3 * params.rho**2) / 24 * params.nu**2
                )
                * t
            )
        )
        assert sabr_lognormal_vol(f, f, t, params) == pytest.approx(expected)

    def test_sabr_rho_sets_the_skew(self):
        f, t = 0.04, 1.0
        strikes = np.array([0.03, 0.05])
        down_skew = sabr_lognormal_vol(f, strikes, t, SABRParams(0.03, 0.5, -0.6, 0.45))
        up_skew = sabr_lognormal_vol(f, strikes, t, SABRParams(0.03, 0.5, 0.6, 0.45))
        assert down_skew[0] > down_skew[1]
        assert up_skew[0] < up_skew[1]

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"alpha": 0.0}, "alpha must be positive"),
            ({"beta": 1.5}, "beta must lie"),
            ({"rho": -1.0}, "rho must lie"),
            ({"nu": -0.1}, "nu must be non-negative"),
        ],
    )
    def test_sabr_parameter_bounds(self, kwargs, match):
        base = {"alpha": 0.03, "beta": 0.5, "rho": 0.0, "nu": 0.4}
        with pytest.raises(ValueError, match=match):
            SABRParams(**{**base, **kwargs})


class TestFitSurface:
    def test_svi_recovers_known_parameters(self):
        surface = fit_surface(svi_chain(), "svi")
        assert surface.model == "svi"
        assert surface.fitter_version == FITTER_VERSION
        assert surface.tenors == pytest.approx([0.25, 0.5, 1.0])
        for fitted in surface.slices:
            truth = SVI_TRUTH[fitted.tenor]
            assert fitted.rmse < 1e-8
            assert fitted.params.b == pytest.approx(truth.b, abs=1e-4)
            assert fitted.params.rho == pytest.approx(truth.rho, abs=1e-3)
            assert fitted.params.sigma == pytest.approx(truth.sigma, abs=1e-3)

    def test_sabr_recovers_known_parameters(self):
        truth = SABRParams(alpha=0.03, beta=0.5, rho=-0.25, nu=0.45)
        surface = fit_surface(sabr_chain(params=truth), "sabr", beta=0.5)
        assert surface.model == "sabr"
        for fitted in surface.slices:
            assert fitted.rmse < 1e-8
            assert fitted.params.alpha == pytest.approx(truth.alpha, abs=1e-5)
            assert fitted.params.rho == pytest.approx(truth.rho, abs=1e-3)
            assert fitted.params.nu == pytest.approx(truth.nu, abs=1e-3)

    def test_fit_reproduces_the_quoted_smile(self):
        chain = svi_chain()
        surface = fit_surface(chain, "svi")
        block = chain[chain["tenor"] == 0.5]
        k = np.log(block["strike"] / block["forward"]).to_numpy()
        assert surface.iv(k, 0.5) == pytest.approx(block["iv"].to_numpy(), abs=1e-6)

    def test_implies_vol_from_prices(self):
        rows = []
        for k in np.linspace(-0.3, 0.3, 11):
            strike = 100.0 * np.exp(k)
            vol = 0.20 + 0.5 * k**2
            option_type = "C" if k >= 0 else "P"  # the liquid side of each wing
            rows.append(
                {
                    "strike": strike,
                    "forward": 100.0,
                    "tenor": 1.0,
                    "price": float(black76_price(100.0, strike, 1.0, vol, option_type, 0.97)),
                    "option_type": option_type,
                    "discount": 0.97,
                }
            )
        surface = fit_surface(pd.DataFrame(rows), "svi")
        assert surface.iv(0.0, 1.0) == pytest.approx(0.20, abs=2e-3)

    def test_expiry_dates_with_asof(self):
        rows = [
            {"strike": strike, "forward": 100.0, "expiry": expiry, "iv": iv}
            for expiry, iv in [(dt.date(2027, 1, 1), 0.20), (dt.date(2028, 1, 1), 0.22)]
            for strike in np.linspace(80.0, 120.0, 9)
        ]
        surface = fit_surface(pd.DataFrame(rows), asof=dt.date(2026, 1, 1))
        assert surface.tenors == pytest.approx([1.0, 2.0], abs=1e-9)

    def test_tz_aware_expiries(self):
        rows = [
            {"strike": strike, "forward": 100.0, "expiry": pd.Timestamp("2027-01-01", tz="UTC"), "iv": 0.20}
            for strike in np.linspace(80.0, 120.0, 9)
        ]
        surface = fit_surface(pd.DataFrame(rows), asof=dt.date(2026, 1, 1))
        assert surface.tenors == pytest.approx([1.0], abs=1e-9)

    def test_column_names_are_case_insensitive(self):
        chain = svi_chain().rename(columns={"strike": "Strike", "forward": "FORWARD", "iv": " IV "})
        assert fit_surface(chain, "svi").tenors == pytest.approx([0.25, 0.5, 1.0])

    def test_unusable_rows_are_dropped_not_fatal(self, caplog):
        chain = svi_chain()
        junk = chain.iloc[[0, 1]].copy()
        junk["iv"] = [np.nan, -0.1]
        junk["tenor"] = [0.25, -1.0]
        with caplog.at_level("WARNING"):
            surface = fit_surface(pd.concat([chain, junk], ignore_index=True), "svi")
        assert "dropping 2 of" in caplog.text
        assert surface.tenors == pytest.approx([0.25, 0.5, 1.0])

    def test_weights_pull_the_fit(self):
        chain = svi_chain()
        block = chain["tenor"] == 0.5
        outlier = chain.index[block][3]
        chain.loc[outlier, "iv"] += 0.05

        unweighted = fit_surface(chain, "svi")
        weighted = fit_surface(chain.assign(weight=np.where(chain.index == outlier, 500.0, 1.0)), "svi")
        k = float(np.log(chain.loc[outlier, "strike"] / chain.loc[outlier, "forward"]))
        target = float(chain.loc[outlier, "iv"])
        assert abs(weighted.iv(k, 0.5) - target) < abs(unweighted.iv(k, 0.5) - target)

    @pytest.mark.parametrize(
        "mutate, match",
        [
            (lambda c: c.drop(columns=["strike"]), "missing required column"),
            (lambda c: c.drop(columns=["iv"]), "'iv' column"),
            (lambda c: c.drop(columns=["tenor"]), "'tenor' column"),
            (lambda c: c.drop(columns=["iv"]).assign(price=1.0), "option_type"),
            (lambda c: c.assign(iv=np.nan), "no usable quotes"),
            (lambda c: c.head(3), "needs at least 5"),
        ],
    )
    def test_rejects_unusable_chains(self, mutate, match):
        with pytest.raises(ValueError, match=match):
            fit_surface(mutate(svi_chain()), "svi")

    def test_expiry_without_asof(self):
        chain = svi_chain().drop(columns=["tenor"]).assign(expiry=dt.date(2027, 1, 1))
        with pytest.raises(ValueError, match="no asof"):
            fit_surface(chain, "svi")

    def test_rejects_unknown_model(self):
        with pytest.raises(ValueError, match="model must be one of"):
            fit_surface(svi_chain(), "heston")

    def test_rejects_non_frame(self):
        with pytest.raises(TypeError, match="must be a pandas DataFrame"):
            fit_surface({"strike": [100.0]}, "svi")

    def test_sabr_needs_fewer_quotes_than_svi(self):
        thin = sabr_chain(tenors=(1.0,)).head(4)
        assert fit_surface(thin, "sabr").slices[0].n_quotes == 4
        with pytest.raises(ValueError, match="needs at least 5"):
            fit_surface(thin, "svi")


class TestInterpolation:
    def test_linear_in_total_variance_not_in_vol(self):
        """The design decision, asserted: the midpoint blends variance, not vol."""
        surface = fit_surface(svi_chain(), "svi")
        early, late = surface.slices[1], surface.slices[2]
        mid_tenor = 0.5 * (early.tenor + late.tenor)
        k = 0.1

        expected_w = 0.5 * (early.total_variance(k) + late.total_variance(k))
        assert surface.total_variance(k, mid_tenor) == pytest.approx(expected_w)

        vol_interpolated = 0.5 * (early.iv(k) + late.iv(k))
        assert surface.iv(k, mid_tenor) != pytest.approx(vol_interpolated, abs=1e-6)

    def test_total_variance_is_non_decreasing_in_tenor(self):
        surface = fit_surface(svi_chain(), "svi")
        tenors = np.linspace(0.02, 4.0, 500)
        for k in (-0.35, -0.1, 0.0, 0.1, 0.35):
            w = surface.total_variance(np.full_like(tenors, k), tenors)
            assert np.all(np.diff(w) >= -1e-12)

    def test_hits_the_fitted_slices_exactly(self):
        surface = fit_surface(svi_chain(), "svi")
        for fitted in surface.slices:
            assert surface.total_variance(0.15, fitted.tenor) == pytest.approx(fitted.total_variance(0.15))

    def test_extrapolates_at_constant_vol(self):
        surface = fit_surface(svi_chain(), "svi")
        first, last = surface.slices[0], surface.slices[-1]
        assert surface.iv(0.2, 0.01) == pytest.approx(first.iv(0.2))
        assert surface.iv(0.2, 10.0) == pytest.approx(last.iv(0.2))

    def test_single_expiry_surface_extrapolates_both_ways(self):
        surface = fit_surface(svi_chain(truth={1.0: SVI_TRUTH[1.0]}), "svi")
        assert len(surface.slices) == 1
        assert surface.iv(0.1, 0.25) == pytest.approx(surface.iv(0.1, 4.0))

    def test_broadcasts(self):
        surface = fit_surface(svi_chain(), "svi")
        assert isinstance(surface.iv(0.0, 0.5), float)
        assert surface.iv(np.linspace(-0.2, 0.2, 7), 0.5).shape == (7,)
        assert surface.iv(0.0, np.array([0.3, 0.6, 0.9])).shape == (3,)
        grid = surface.iv(np.linspace(-0.2, 0.2, 4).reshape(4, 1), np.array([[0.3, 0.9]]))
        assert grid.shape == (4, 2)

    def test_rejects_non_positive_tenor(self):
        surface = fit_surface(svi_chain(), "svi")
        with pytest.raises(ValueError, match="tenor must be positive"):
            surface.iv(0.0, 0.0)

    def test_forward_lookup(self):
        surface = fit_surface(svi_chain(forward=250.0), "svi")
        assert surface.forward(0.6) == pytest.approx(250.0)

    def test_rejects_out_of_order_slices(self):
        slices = fit_surface(svi_chain(), "svi").slices
        with pytest.raises(ValueError, match="ascending in tenor"):
            VolSurface(slices=(slices[2], slices[0]), model="svi")

    def test_rejects_empty_surface(self):
        with pytest.raises(ValueError, match="at least one fitted expiry"):
            VolSurface(slices=(), model="svi")


class TestArbitrage:
    def test_durrleman_g_of_a_flat_smile_is_one(self):
        assert durrleman_g(np.linspace(-1, 1, 5), 0.04, 0.0, 0.0) == pytest.approx(1.0)

    def test_clean_svi_surface_has_no_violations(self):
        assert fit_surface(svi_chain(), "svi").check_arbitrage() == []

    def test_clean_sabr_surface_has_no_violations(self):
        assert fit_surface(sabr_chain(), "sabr").check_arbitrage() == []

    def test_detects_butterfly_arbitrage(self):
        """Vogt's parameters: a plausible-looking smile with a negative density."""
        slice_ = SVISlice(tenor=1.0, forward=1.0, n_quotes=9, rmse=0.0, params=VOGT, **VOGT_K_RANGE)
        violations = VolSurface(slices=(slice_,), model="svi").check_arbitrage()
        assert violations
        assert {v.kind for v in violations} == {"butterfly"}
        assert all(v.value < 0 for v in violations)
        assert all(0.6 < v.log_moneyness < 1.3 for v in violations)
        assert min(v.value for v in violations) == violations[0].value  # worst first

    def test_butterfly_check_covers_interpolated_tenors(self):
        """
        A blend of two slices is not the same smile as either, so the default
        grid checks the midpoints between fitted expiries as well as the
        expiries themselves.
        """
        doubled = SVIParams(a=2 * VOGT.a, b=2 * VOGT.b, rho=VOGT.rho, m=VOGT.m, sigma=VOGT.sigma)
        surface = VolSurface(
            slices=(
                SVISlice(tenor=1.0, forward=1.0, n_quotes=9, rmse=0.0, params=VOGT, **VOGT_K_RANGE),
                SVISlice(tenor=2.0, forward=1.0, n_quotes=9, rmse=0.0, params=doubled, **VOGT_K_RANGE),
            ),
            model="svi",
        )
        checked = {v.tenor for v in surface.check_arbitrage() if v.kind == "butterfly"}
        assert checked == {1.0, 1.5, 2.0}

    def test_detects_calendar_arbitrage(self):
        """Total variance falling with tenor: 30 vol at 6m against 20 vol at 1y."""
        rows = [
            {"strike": 100.0 * np.exp(k), "forward": 100.0, "tenor": tenor, "iv": base + 0.3 * k**2}
            for tenor, base in [(0.5, 0.30), (1.0, 0.20)]
            for k in np.linspace(-0.3, 0.3, 9)
        ]
        violations = fit_surface(pd.DataFrame(rows), "svi").check_arbitrage()
        calendar = [v for v in violations if v.kind == "calendar"]
        assert calendar
        assert all(v.tenor == 1.0 for v in calendar)
        assert all(v.value < 0 for v in calendar)

    def test_interpolating_in_vol_would_have_hidden_it(self):
        """
        The reason the surface interpolates in total variance.

        Two individually clean slices whose *vols* are ordered but whose total
        variances are not: a vol interpolant looks fine at the midpoint while
        total variance has already fallen.
        """
        early = SVISlice(
            tenor=0.5,
            forward=100.0,
            n_quotes=9,
            rmse=0.0,
            k_min=-0.3,
            k_max=0.3,
            params=SVIParams(a=0.045, b=0.02, rho=0.0, m=0.0, sigma=0.1),
        )
        late = SVISlice(
            tenor=1.0,
            forward=100.0,
            n_quotes=9,
            rmse=0.0,
            k_min=-0.3,
            k_max=0.3,
            params=SVIParams(a=0.040, b=0.02, rho=0.0, m=0.0, sigma=0.1),
        )
        assert late.iv(0.0) < early.iv(0.0)  # a vol-space monotonicity check passes
        assert late.total_variance(0.0) < early.total_variance(0.0)  # total variance does not
        surface = VolSurface(slices=(early, late), model="svi")
        assert any(v.kind == "calendar" for v in surface.check_arbitrage())

    def test_accepts_an_explicit_grid(self):
        surface = fit_surface(svi_chain(), "svi")
        assert surface.check_arbitrage(log_moneyness=np.linspace(-1.5, 1.5, 41), tenors=[0.3, 0.7]) == []

    def test_tolerance_absorbs_a_marginal_violation(self):
        rows = [
            {"strike": 100.0 * np.exp(k), "forward": 100.0, "tenor": tenor, "iv": iv}
            for tenor, iv in [(1.0, 0.200000), (2.0, 0.1414208)]  # w: 0.0400000 then 0.0399999
            for k in np.linspace(-0.2, 0.2, 9)
        ]
        surface = fit_surface(pd.DataFrame(rows), "svi")
        assert any(v.kind == "calendar" for v in surface.check_arbitrage(tol=1e-12))
        assert surface.check_arbitrage(tol=1e-3) == []


class TestSlices:
    def test_numeric_derivatives_match_analytic_svi(self):
        """SABRSlice leans on the base class's central differences; check the machinery."""
        params = SVI_TRUTH[1.0]
        analytic = SVISlice(tenor=1.0, forward=100.0, n_quotes=9, rmse=0.0, k_min=-0.3, k_max=0.3, params=params)
        k = np.linspace(-0.3, 0.3, 13)
        numeric = super(SVISlice, analytic).derivatives(k)
        # The base class differences at h=1e-3, so the second derivative carries
        # O(h^2) truncation error -- a few parts in 1e5 here.
        for exact, approx in zip(analytic.derivatives(k), numeric, strict=True):
            assert exact == pytest.approx(approx, rel=1e-4, abs=1e-8)

    def test_sabr_slice_reconstructs_strikes_from_log_moneyness(self):
        params = SABRParams(alpha=0.03, beta=0.5, rho=-0.25, nu=0.45)
        slice_ = SABRSlice(tenor=1.0, forward=0.04, n_quotes=11, rmse=0.0, k_min=-0.3, k_max=0.3, params=params)
        k = 0.2
        assert slice_.iv(k) == pytest.approx(sabr_lognormal_vol(0.04, 0.04 * np.exp(k), 1.0, params))

    def test_slice_iv_and_total_variance_agree(self):
        slice_ = fit_surface(svi_chain(), "svi").slices[0]
        k = np.linspace(-0.3, 0.3, 7)
        assert slice_.iv(k) ** 2 * slice_.tenor == pytest.approx(slice_.total_variance(k))
