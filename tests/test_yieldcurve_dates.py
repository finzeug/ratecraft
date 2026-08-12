"""Date handling in YieldCurve, exercised by building actual curves.

finzeug/ratecraft#17. The sibling file test_yieldcurve_tz.py pins the shapes of
the fix by reading the source; these tests build curves and ask them what they
computed, which is what actually matters.

The root was that `d0` carried a time of day (16:00 ET arrives as 21:00 UTC)
while every `maturity_date` sat at midnight UTC. Everything below follows from
that mismatch:

* day counts came out one day short, because `n days 03:00:00` truncates to n-1;
* a bond maturing ON the price date sorted BEFORE the interpolation anchor, at
  `d_to == -1`, and was bootstrapped over a negative interval;
* a bond maturing the NEXT day landed on the anchor's own `d_to == 0` with
  `days == 0`;
* `.loc[d0 + relativedelta(days=1):]` dropped real maturities at 00:00 UTC on
  d0+1 and raised KeyError instead of returning fewer rows.
"""

import logging

import numpy as np
import pandas as pd
import pytest

from ratecraft import yieldcurve
from ratecraft.yieldcurve import YieldCurve

# 16:00 ET, the shape a US Treasury close arrives in.
PRICE_DATE = pd.Timestamp("2024-01-02 16:00")
D0 = pd.Timestamp("2024-01-02 21:00", tz="UTC")

# Maturities far enough out that no ex-coupon boundary is in play.
BASE = [
    ("2024-04-15", 0.0275, 99.1),
    ("2024-06-30", 0.0450, 100.3),
    ("2025-06-30", 0.0375, 98.7),
    ("2027-06-30", 0.0400, 101.2),
    ("2034-06-30", 0.0425, 97.4),
]

# Maturities that survive even ex_coupon_days=200 (cutoff 2024-07-20), so the two
# curves in the ex-coupon tests have the same node set and stay comparable.
FAR = BASE[2:]


def frame(rows, price_date=PRICE_DATE):
    """A hog-shaped price frame: tz-naive price_date, midnight maturities."""
    return pd.DataFrame(
        [
            {
                "cusip": f"C{i:04d}",
                "buy": buy,
                "sell": buy + 0.1,
                "rate": rate,
                "maturity_date": pd.Timestamp(mat),
                "price_date": price_date,
                "sectype": "NOTE",
            }
            for i, (mat, rate, buy) in enumerate(rows)
        ]
    )


def maturities(yc):
    """The curve's surviving maturities, tz intact.

    Deliberately not `yc.p["maturity_date"].values` -- that returns a tz-NAIVE
    datetime64 array, so `pd.Timestamp(..., tz="UTC") in ...` is always False and
    a membership assertion written that way passes for the wrong reason.
    """
    return set(yc.p["maturity_date"])


class TestD0Normalisation:
    def test_time_of_day_does_not_change_the_curve(self):
        """Same trading date, two stamping times -> one curve.

        The curve is a function of the trading date. Before normalising, a frame
        stamped at 16:00 ET and the same frame stamped at 09:00 ET produced
        different day counts and therefore different discount factors.
        """
        morning = YieldCurve(pd.Timestamp("2024-01-02 09:00", tz="UTC"), frame(BASE))
        close = YieldCurve(D0, frame(BASE))

        assert morning.d0 == close.d0 == pd.Timestamp("2024-01-02", tz="UTC")
        pd.testing.assert_series_equal(morning.rates["d_to"], close.rates["d_to"])
        pd.testing.assert_series_equal(
            morning.rates["force_cumul"], close.rates["force_cumul"]
        )

    def test_day_counts_are_true_date_differences(self):
        """2024-01-02 to 2024-06-30 is 180 days, not 179.

        Every discount period used to be one day short. duration.py already
        counts days as `(maturity - as_of_date).days`, so the curve's node
        placement disagreed with the consumer's own day count by one.
        """
        yc = YieldCurve(D0, frame(BASE))
        for mat, _rate, _buy in BASE:
            expected = (pd.Timestamp(mat).date() - pd.Timestamp("2024-01-02").date()).days
            assert yc.rates.loc[pd.Timestamp(mat, tz="UTC"), "d_to"] == expected

        assert yc.rates.loc[pd.Timestamp("2024-06-30", tz="UTC"), "d_to"] == 180

    def test_d0_is_midnight_even_from_a_tz_aware_et_timestamp(self):
        yc = YieldCurve(pd.Timestamp("2024-01-02 16:00", tz="America/New_York"), frame(BASE))
        assert yc.d0 == pd.Timestamp("2024-01-02", tz="UTC")


class TestSpentSecuritiesAreExcluded:
    """Symptom 2: a bond with no cash flow left must not become a curve node."""

    def test_same_day_maturity_leaves_the_curve_untouched(self):
        clean = YieldCurve(D0, frame(BASE))
        with_matured = YieldCurve(D0, frame([("2024-01-02", 0.0175, 99.98)] + BASE))

        assert pd.Timestamp("2024-01-02", tz="UTC") not in maturities(with_matured)
        pd.testing.assert_series_equal(
            clean.rates["force_cumul"], with_matured.rates["force_cumul"]
        )
        pd.testing.assert_series_equal(clean.rates["z"], with_matured.rates["z"])

    def test_no_node_sorts_before_the_anchor(self):
        """`d_to == -1` was how the matured bond got in front of the anchor."""
        yc = YieldCurve(D0, frame([("2024-01-02", 0.0175, 99.98)] + BASE))
        assert (yc.rates["d_to"] >= 0).all()
        assert yc.rates.index[0] == yc.d0

    def test_no_zero_or_negative_length_bootstrap_period(self):
        """`days <= 0` means a period's force was solved over no time at all."""
        yc = YieldCurve(D0, frame([("2024-01-02", 0.0175, 99.98), ("2024-01-03", 0.02, 99.99)] + BASE))
        days = yc.rates["days"].dropna()
        assert (days > 0).all(), f"non-positive period lengths: {days[days <= 0]}"

    def test_annualised_force_is_never_nonsense(self):
        """The matured node reported force_annual == -3.07, i.e. -307%/yr."""
        yc = YieldCurve(D0, frame([("2024-01-02", 0.0175, 99.98)] + BASE))
        fa = yc.rates["force_annual"].dropna()
        assert (fa.abs() < 1).all(), f"implausible annualised forces: {fa[fa.abs() >= 1]}"

    def test_the_drop_is_logged_with_the_cusip(self, caplog):
        """A silently discarded priced security is the failure mode of the issue."""
        with caplog.at_level(logging.WARNING, logger="ratecraft.yieldcurve"):
            YieldCurve(D0, frame([("2024-01-02", 0.0175, 99.98)] + BASE))
        assert any(
            "C0000" in r.getMessage() and "2024-01-02" in r.getMessage()
            for r in caplog.records
            if r.levelno >= logging.WARNING
        ), caplog.text

    def test_ex_coupon_window_is_the_cutoff(self):
        """`coupon_dates` is empty exactly when maturity <= d0 + ex_coupon_days.

        Such a bond reports no payments at all -- it loses even its principal
        repayment -- so `Bond.payments()` fails on an empty frame. The curve must
        not admit one.
        """
        yc = YieldCurve(D0, frame([("2024-01-03", 0.02, 99.99)] + BASE), ex_coupon_days=1)
        assert pd.Timestamp("2024-01-03", tz="UTC") not in maturities(yc)

        # ex_coupon_days=0 moves the boundary, so the same bond is admissible.
        yc0 = YieldCurve(D0, frame([("2024-01-03", 0.02, 99.99)] + BASE), ex_coupon_days=0)
        assert pd.Timestamp("2024-01-03", tz="UTC") in maturities(yc0)

    def test_nothing_left_to_bootstrap_says_so(self):
        with pytest.raises(ValueError, match="nothing to bootstrap"):
            YieldCurve(D0, frame([("2023-11-15", 0.02, 99.9), ("2024-01-02", 0.02, 99.99)]))


class TestPaymentDatesSkipOnlyTheAnchor:
    """Symptom 3: the intent was "skip the anchor row", not "skip a day"."""

    def test_maturities_on_price_date_and_next_day_do_not_raise(self):
        """The exact reproduction: KeyError: Timestamp('2024-01-03 00:00:00+0000')."""
        yc = YieldCurve(D0, frame([("2024-01-02", 0.0175, 99.98), ("2024-01-03", 0.02, 99.99)] + BASE))
        assert yc.payment_dates is not None
        assert not yc.payment_dates.empty

    def test_every_curve_node_gets_payment_dates_and_the_anchor_gets_none(self):
        yc = YieldCurve(D0, frame(BASE))
        covered = set(yc.payment_dates.index.get_level_values("maturity_date"))
        assert covered == {pd.Timestamp(m, tz="UTC") for m, _r, _b in BASE}
        assert yc.d0 not in covered

    def test_a_maturity_one_day_out_is_kept_not_dropped(self):
        """`.loc[d0 + 1 day:]` was excluding this row as well as the anchor."""
        yc = YieldCurve(D0, frame([("2024-01-03", 0.02, 99.99)] + BASE), ex_coupon_days=0)
        covered = set(yc.payment_dates.index.get_level_values("maturity_date"))
        assert pd.Timestamp("2024-01-03", tz="UTC") in covered
        assert yc.d0 not in covered


class TestYieldsActuallySolve:
    """Symptom 1: the tz mismatch made every yield a swallowed NaN."""

    def test_ytm_is_finite_for_every_bond_the_curve_built(self):
        yc = YieldCurve(D0, frame(BASE))
        ytms = list(yc.p["bond"].bnd.ytm())
        assert len(ytms) == len(BASE)
        assert all(np.isfinite(y) for y in ytms), ytms
        assert all(0 < y < 0.2 for y in ytms), ytms

    def test_price_date_column_is_tz_aware(self):
        yc = YieldCurve(D0, frame(BASE))
        assert yc.p["price_date"].dt.tz is not None
        # The subtraction that used to raise TypeError.
        assert (yc.p["maturity_date"] - yc.p["price_date"]).dt.days.gt(0).all()

    def test_income_and_const_rate_pv_run(self):
        yc = YieldCurve(D0, frame(BASE))
        bond = yc.p["bond"].iloc[-1]
        assert np.isfinite(bond.const_rate_pv(0.04))
        assert not bond.income().empty

    def test_standard_yield_curve_is_populated(self):
        yc = YieldCurve(D0, frame(BASE))
        syc = yc.standard_yield_curve()
        assert syc.notna().all(), syc


class TestExCouponDaysIsHonoured:
    """Symptom 4: the parameter was accepted, stored nowhere, forwarded nowhere."""

    def test_it_changes_the_accrued_interest_factors(self):
        tight = YieldCurve(D0, frame(FAR), ex_coupon_days=1)
        wide = YieldCurve(D0, frame(FAR), ex_coupon_days=200)
        # Same node set, so the factors line up index-for-index.
        assert list(tight.rates.index) == list(wide.rates.index)
        assert not np.allclose(
            tight.rates["accrued_interest_factor"].values,
            wide.rates["accrued_interest_factor"].values,
        )

    def test_it_reaches_the_bonds_on_the_frame(self):
        wide = YieldCurve(D0, frame(FAR), ex_coupon_days=200)
        assert all(b.ex_coupon_days == 200 for b in wide.p["bond"])

    def test_it_reaches_the_payment_schedule(self):
        """This Bond is built inside _prep_payment_dates, which omitted it.

        The payment dates it produces feed every discount factor, so pinning them
        to the module default made the parameter cosmetic where it mattered most.
        """
        tight = YieldCurve(D0, frame(FAR), ex_coupon_days=1)
        wide = YieldCurve(D0, frame(FAR), ex_coupon_days=200)
        assert len(wide.payment_dates) < len(tight.payment_dates)

    def test_it_reaches_yield_rate(self, monkeypatch):
        """Asserted on the constructor call, not on the resulting number.

        Comparing `tight.yield_rate(t) != wide.yield_rate(t)` proves nothing here:
        the two curves already have different `rates` (accrued interest feeds the
        bootstrap), so that comparison passes even when yield_rate's own Bond is
        built with the module default. Watch the argument instead.
        """
        yc = YieldCurve(D0, frame(FAR), ex_coupon_days=200)

        real_bond = yieldcurve.Bond
        seen = []

        def spy(p, **kw):
            seen.append(kw.get("ex_coupon_days", "OMITTED"))
            return real_bond(p, **kw)

        monkeypatch.setattr(yieldcurve, "Bond", spy)
        yc.yield_rate(pd.Timestamp("2029-01-02", tz="UTC"))
        assert seen == [200]
