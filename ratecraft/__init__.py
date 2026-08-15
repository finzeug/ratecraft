"""
ratecraft — fixed income math: bonds, yield curves, duration, inflation, and
implied-volatility surfaces.

Pure calculation library with no I/O or data fetching.
"""

__version__ = "0.1.15"

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
    "fit_surface",
    "VolSurface",
    "VolSlice",
    "SVISlice",
    "SABRSlice",
    "SVIParams",
    "SABRParams",
    "ArbitrageViolation",
    "svi_total_variance",
    "svi_derivatives",
    "sabr_lognormal_vol",
    "durrleman_g",
    "black76_price",
    "implied_vol_black76",
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
from .vol import (  # noqa: F401
    ArbitrageViolation,
    SABRParams,
    SABRSlice,
    SVIParams,
    SVISlice,
    VolSlice,
    VolSurface,
    black76_price,
    durrleman_g,
    fit_surface,
    implied_vol_black76,
    sabr_lognormal_vol,
    svi_derivatives,
    svi_total_variance,
)
from .yieldcurve import YieldCurve, cpi_factors  # noqa: F401
