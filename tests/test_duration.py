"""Basic tests for ratecraft.duration module."""

import datetime as dt

import pytest

from ratecraft.duration import (
    parse_zero_maturity,
    parse_actuarial_maturity,
    parse_bond_maturity,
    is_nominal_zero,
    is_real_zero,
    is_actuarial,
    zero_yield_from_price,
    zero_duration,
    calculate_breakeven_inflation,
    years_to_maturity,
    get_duration,
    load_etf_durations,
)


class TestParsers:
    def test_parse_zero_nominal(self):
        assert parse_zero_maturity("zn_2030-06-30") == dt.date(2030, 6, 30)

    def test_parse_zero_real(self):
        assert parse_zero_maturity("zr_2025-12-31") == dt.date(2025, 12, 31)

    def test_parse_zero_invalid(self):
        assert parse_zero_maturity("SVIX") is None

    def test_parse_actuarial(self):
        assert parse_actuarial_maturity("lr_2086-12-31_BDH") == dt.date(2086, 12, 31)
        assert parse_actuarial_maturity("dn_2045-09-30_EVH") == dt.date(2045, 9, 30)

    def test_parse_bond_maturity_full(self):
        assert parse_bond_maturity("US Treasury TIP 0.25% 02/15/2050") == dt.date(2050, 2, 15)

    def test_parse_bond_maturity_due(self):
        assert parse_bond_maturity("UST INFL IDX 1.625%10/27INFL INDEX DUE 10/15/27") == dt.date(2027, 10, 15)


class TestClassifiers:
    def test_nominal_zero(self):
        assert is_nominal_zero("zn_2030-06-30")
        assert not is_nominal_zero("zr_2030-06-30")

    def test_real_zero(self):
        assert is_real_zero("zr_2030-06-30")
        assert not is_real_zero("zn_2030-06-30")

    def test_actuarial(self):
        assert is_actuarial("lr_2086-12-31_BDH")
        assert is_actuarial("dn_2045-09-30_EVH")
        assert not is_actuarial("zn_2030-06-30")


class TestCalculations:
    def test_zero_yield(self):
        # A zero at 0.85 with 10 years: yield ~ 1.6%
        y = zero_yield_from_price(0.85, 10)
        assert 0.01 < y < 0.03

    def test_zero_yield_edge(self):
        assert zero_yield_from_price(0, 10) == 0.0
        assert zero_yield_from_price(0.85, 0) == 0.0

    def test_zero_duration(self):
        result = zero_duration("zn_2035-04-13", 0.85, dt.date(2025, 4, 13))
        assert result is not None
        assert result["years_to_maturity"] == pytest.approx(10.0, abs=0.1)
        assert result["macaulay_dur"] == result["years_to_maturity"]
        assert result["modified_dur"] < result["macaulay_dur"]

    def test_breakeven_inflation(self):
        # Real zero more expensive → positive breakeven
        be = calculate_breakeven_inflation(0.75, 0.80, 10)
        assert be > 0

    def test_years_to_maturity(self):
        y = years_to_maturity("zn_2035-04-13", dt.date(2025, 4, 13))
        assert y == pytest.approx(10.0, abs=0.1)

    def test_years_to_maturity_actuarial(self):
        y = years_to_maturity("lr_2086-12-31_BDH", dt.date(2025, 4, 13))
        assert y is not None
        assert y > 60


class TestGetDuration:
    def test_zero(self):
        result = get_duration("zn_2035-04-13", 0.85, dt.date(2025, 4, 13))
        assert result is not None
        assert result["source"] == "calculated"

    def test_etf_from_config(self):
        result = get_duration("TLT", 90.0, dt.date(2025, 4, 13))
        if result is not None:  # config file must be present
            assert result["source"] == "config"
            assert result["modified_dur"] > 10

    def test_unknown(self):
        result = get_duration("UNKNOWN_TICKER", 100.0, dt.date(2025, 4, 13))
        assert result is None


class TestLoadEtfDurations:
    def test_loads(self):
        d = load_etf_durations()
        assert isinstance(d, dict)
        if d:  # config present
            assert "TLT" in d
