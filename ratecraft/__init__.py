"""
ratecraft — fixed income math: bonds, yield curves, duration, and inflation.

Pure calculation library with no I/O or data fetching.
"""

__version__ = "0.1.1"

__all__ = [
    "Bond",
    "TIPS",
    "BondAccessor",
    "prior_coupon_date",
    "accrued_interest_factor",
    "ex_coupon_days",
    "YieldCurve",
    "cpi_factors",
    "zero_duration",
    "zero_yield_from_price",
    "calculate_breakeven_inflation",
    "calculate_dollar_duration",
    "get_duration",
    "get_matching_zeros",
    "load_etf_durations",
]

from .bond import Bond, TIPS, BondAccessor, prior_coupon_date, accrued_interest_factor, ex_coupon_days  # noqa: F401
from .yieldcurve import YieldCurve, cpi_factors  # noqa: F401
from .duration import (  # noqa: F401
    zero_duration,
    zero_yield_from_price,
    calculate_breakeven_inflation,
    calculate_dollar_duration,
    get_duration,
    get_matching_zeros,
    load_etf_durations,
)
