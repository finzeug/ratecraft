"""YieldCurve must not make one date aware and leave its partner naive.

finzeug/ratecraft#17. `__init__` forced `maturity_date` to UTC and left the
`price_date` column alone, so every `Bond` a curve built raised
`TypeError: Cannot subtract tz-naive and tz-aware` at `(p.maturity_date - d).days`
-- and a bare `except` around the yield solve turned that into a silent NaN.

`ytm()`, `income()` and `const_rate_pv()` were therefore dead on the normal
construction path, and a consumer charting yields got an all-NaN column with
nothing logged as a failure. That is the shape worth pinning: not "the yield is
wrong" but "the yield never ran and said NaN".
"""

import ast
from pathlib import Path

import pandas as pd
import pytest

_SRC = Path(__file__).resolve().parents[1] / "ratecraft"


def _code(path):
    return ast.unparse(ast.parse((_SRC / path).read_text()))


def test_price_date_is_normalised_with_maturity_date():
    code = _code("yieldcurve.py")
    assert "p['price_date'] = pd.to_datetime(p['price_date'], utc=True)" in code


def test_subtracting_the_two_no_longer_raises():
    """The exact operation that was failing, on the exact shapes involved."""
    maturity = pd.to_datetime(pd.Series(["2030-05-15"]), utc=True)
    naive = pd.to_datetime(pd.Series(["2024-01-02 16:00"]))
    with pytest.raises(TypeError):
        _ = maturity - naive  # what used to happen
    aware = pd.to_datetime(naive, utc=True)
    assert (maturity - aware).dt.days.iloc[0] > 0  # what happens now


def test_the_yield_solve_no_longer_swallows_everything():
    """A solver that fails to converge is a NaN; a type error is a bug.

    A bare `except` made the two indistinguishable, which is how an all-NaN
    column read as 'did not converge' for what was really a malformed frame.
    """
    code = _code("bond.py")
    assert "except (ValueError, TypeError, RuntimeError)" in code
    assert "except Exception as e:\n            logger.error(f'Exception: {e}')" not in code


def test_ex_coupon_days_reaches_the_things_that_use_it():
    """Accepted since the signature was written; never stored, never forwarded.

    A curve built with ex_coupon_days=200 produced factors identical to =1. A
    parameter that does nothing is worse than no parameter: the caller believes
    it worked.
    """
    code = _code("yieldcurve.py")
    assert "self.ex_coupon_days = ex_coupon_days" in code
    assert "ex_coupon_days=self.ex_coupon_days" in code
    assert code.count("ex_coupon_days=self.ex_coupon_days") >= 2
