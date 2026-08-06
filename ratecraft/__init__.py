"""
ratecraft — fixed income math: bonds, yield curves, duration, and inflation.

Pure calculation library with no I/O or data fetching.
"""

__version__ = "0.1.5"

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

from .bond import TIPS, Bond, BondAccessor, accrued_interest_factor, ex_coupon_days, prior_coupon_date  # noqa: F401
from .duration import (  # noqa: F401
    calculate_breakeven_inflation,
    calculate_dollar_duration,
    get_duration,
    get_matching_zeros,
    load_etf_durations,
    zero_duration,
    zero_yield_from_price,
)
from .yieldcurve import YieldCurve, cpi_factors  # noqa: F401
