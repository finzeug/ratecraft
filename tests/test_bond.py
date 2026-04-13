"""Basic tests for ratecraft.bond module."""

import datetime as dt

import pandas as pd

from ratecraft.bond import Bond, TIPS, prior_coupon_date, accrued_interest_factor


def _make_bond_series(**overrides):
    """Helper to build a minimal bond price Series."""
    defaults = {
        "cusip": "912828ZZ0",
        "rate": 0.025,  # 2.5% coupon
        "maturity_date": dt.datetime(2030, 5, 15),
        "price_date": dt.datetime(2025, 4, 13),
        "buy": 0.98,
        "sell": 0.97,
        "end_of_day": 0.975,
    }
    defaults.update(overrides)
    return pd.Series(defaults)


class TestPriorCouponDate:
    def test_between_coupons(self):
        """Price date between two coupon dates returns the earlier one."""
        mat = dt.datetime(2030, 5, 15)
        date = dt.datetime(2025, 3, 1)
        pcd = prior_coupon_date(date, mat)
        assert pcd == dt.datetime(2024, 11, 15)  # most recent semianniversary before price date

    def test_on_coupon_date(self):
        """Price date on a coupon date returns that date."""
        mat = dt.datetime(2030, 5, 15)
        date = dt.datetime(2025, 5, 15)
        pcd = prior_coupon_date(date, mat)
        assert pcd == dt.datetime(2025, 5, 15)


class TestBond:
    def test_basic_construction(self):
        b = Bond(_make_bond_series())
        assert b.p["rate"] == 0.025
        assert len(b.coupon_dates) > 0
        assert b.accrued_interest_factor >= 0

    def test_payments_sum(self):
        """Last payment should include principal."""
        b = Bond(_make_bond_series())
        pmts = b.payments()
        assert pmts.iloc[-1]["principal"] == 1.0
        assert all(pmts["coupon"] == 0.025 / 2)

    def test_ytm_reasonable(self):
        """YTM should be a reasonable rate for a near-par bond."""
        b = Bond(_make_bond_series())
        y = b.ytm(basis="sell")
        assert 0 < y < 0.20  # between 0% and 20%


class TestTIPS:
    def test_index_ratio(self):
        s = _make_bond_series(IndexRatio=1.05)
        t = TIPS(s)
        assert t.index_ratio == 1.05
        assert t.price > 0


class TestAccruedInterestFactor:
    def test_positive(self):
        mat = dt.datetime(2030, 5, 15)
        date = dt.datetime(2025, 3, 1)
        aif = accrued_interest_factor(date, mat)
        assert 0 < aif < 0.5  # less than one semiannual period
