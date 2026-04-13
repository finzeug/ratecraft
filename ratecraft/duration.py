"""
Duration calculation utilities for fixed income instruments.

Provides functions to calculate Macaulay duration, modified duration, and dollar duration
for zero-coupon bonds and to load ETF durations from configuration.
"""

import datetime as dt
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

import logging

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def load_etf_durations() -> dict[str, float]:
    """
    Load ETF durations from YAML config.

    Returns:
        Dictionary mapping ETF mnemonic to duration in years.
    """
    config_path = Path(__file__).parent / "data" / "duration_config.yaml"
    if not config_path.exists():
        logger.warning(f"Duration config file not found: {config_path}")
        return {}

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config.get("etf_durations", {})


def parse_zero_maturity(mnemonic: str) -> Optional[dt.date]:
    """
    Extract maturity date from zn_YYYY-MM-DD or zr_YYYY-MM-DD mnemonic.

    Args:
        mnemonic: The commodity mnemonic (e.g., "zn_2030-06-30", "zr_2025-12-31")

    Returns:
        The maturity date, or None if mnemonic doesn't match the zero pattern.
    """
    match = re.match(r"^z[nr]_(\d{4}-\d{2}-\d{2})$", mnemonic)
    if not match:
        return None

    try:
        return dt.datetime.strptime(match.group(1), "%Y-%m-%d").date()
    except ValueError:
        logger.warning(f"Invalid date in mnemonic: {mnemonic}")
        return None


def parse_actuarial_maturity(mnemonic: str) -> Optional[dt.date]:
    """
    Extract maturity date from actuarial mnemonics.

    Actuarial mnemonics have the format: {type}_{YYYY-MM-DD}_{suffix}
    where type is lr, ln, dr, or dn.

    Examples:
        - lr_2086-12-31_BDH (life contingent real)
        - ln_2050-06-30_EVH (life contingent nominal)
        - dr_2040-03-31_BDH (death contingent real)
        - dn_2045-09-30_EVH (death contingent nominal)

    Args:
        mnemonic: The actuarial commodity mnemonic

    Returns:
        The maturity date, or None if mnemonic doesn't match the actuarial pattern.
    """
    match = re.match(r"^[ld][nr]_(\d{4}-\d{2}-\d{2})_", mnemonic)
    if not match:
        return None

    try:
        return dt.datetime.strptime(match.group(1), "%Y-%m-%d").date()
    except ValueError:
        logger.warning(f"Invalid date in actuarial mnemonic: {mnemonic}")
        return None


def parse_bond_maturity(fullname: str) -> Optional[dt.date]:
    """
    Extract maturity date from bond fullname.

    Handles various formats:
        - "US Treasury TIP 0.25% 02/15/2050"
        - "UST INFL IDX 1.625%10/27INFL INDEX DUE 10/15/27"
        - "UST INFL IDX 0.375%07/27INFL INDEX DUE 07/15/27"

    Args:
        fullname: The bond's full name/description

    Returns:
        The maturity date, or None if no date pattern found.
    """
    if not fullname:
        return None

    # Pattern 1: MM/DD/YYYY (e.g., "02/15/2050")
    match = re.search(r'(\d{2})/(\d{2})/(\d{4})', fullname)
    if match:
        try:
            return dt.date(int(match.group(3)), int(match.group(1)), int(match.group(2)))
        except ValueError:
            pass

    # Pattern 2: "DUE MM/DD/YY" (e.g., "DUE 10/15/27")
    match = re.search(r'DUE\s+(\d{1,2})/(\d{1,2})/(\d{2})', fullname)
    if match:
        try:
            year = int(match.group(3))
            # Assume 20xx for 2-digit years
            year = 2000 + year if year < 100 else year
            return dt.date(year, int(match.group(1)), int(match.group(2)))
        except ValueError:
            pass

    # Pattern 3: percentage followed by MM/YY (e.g., "1.625%10/27")
    match = re.search(r'%\s*(\d{1,2})/(\d{2})', fullname)
    if match:
        try:
            year = int(match.group(2))
            year = 2000 + year if year < 100 else year
            month = int(match.group(1))
            # Assume 15th of month for bonds
            return dt.date(year, month, 15)
        except ValueError:
            pass

    return None


def is_tips_bond(mnemonic: str, namespace: Optional[str] = None) -> bool:
    """
    Check if a security is a TIPS bond based on mnemonic/namespace.

    TIPS bonds typically have CUSIPs starting with 912 and are in BOND namespace.
    """
    if namespace == "BOND":
        return True
    if mnemonic and mnemonic.startswith("912"):
        return True
    return False


def is_nominal_zero(mnemonic: str) -> bool:
    """Check if mnemonic represents a nominal zero (zn_*)."""
    return mnemonic.startswith("zn_")


def is_real_zero(mnemonic: str) -> bool:
    """Check if mnemonic represents a real (inflation-indexed) zero (zr_*)."""
    return mnemonic.startswith("zr_")


def is_actuarial_nominal(mnemonic: str) -> bool:
    """
    Check if mnemonic represents an actuarial nominal liability (ln_* or dn_*).

    These are treated as nominal zeros for duration purposes.
    """
    return mnemonic.startswith("ln_") or mnemonic.startswith("dn_")


def is_actuarial_real(mnemonic: str) -> bool:
    """
    Check if mnemonic represents an actuarial real liability (lr_* or dr_*).

    These are treated as real zeros for duration purposes.
    """
    return mnemonic.startswith("lr_") or mnemonic.startswith("dr_")


def is_actuarial(mnemonic: str) -> bool:
    """Check if mnemonic represents any actuarial liability."""
    return is_actuarial_nominal(mnemonic) or is_actuarial_real(mnemonic)


def is_nominal_type(mnemonic: str) -> bool:
    """Check if mnemonic is nominal (zn_* or ln_* or dn_*)."""
    return is_nominal_zero(mnemonic) or is_actuarial_nominal(mnemonic)


def is_real_type(mnemonic: str) -> bool:
    """Check if mnemonic is real/inflation-indexed (zr_* or lr_* or dr_*)."""
    return is_real_zero(mnemonic) or is_actuarial_real(mnemonic)


def years_to_maturity(mnemonic: str, as_of_date: dt.date) -> Optional[float]:
    """
    Calculate years to maturity for a zero-coupon bond or actuarial liability.

    Args:
        mnemonic: The mnemonic (zn_*, zr_*, lr_*, ln_*, dr_*, dn_*)
        as_of_date: The date from which to calculate time to maturity

    Returns:
        Years to maturity as a float, or None if not a valid mnemonic.
    """
    # Try zero-coupon bond format first
    maturity = parse_zero_maturity(mnemonic)

    # Try actuarial format if zero format didn't match
    if maturity is None:
        maturity = parse_actuarial_maturity(mnemonic)

    if maturity is None:
        return None

    days = (maturity - as_of_date).days
    if days <= 0:
        return 0.0

    return days / 365.25


def zero_yield_from_price(price: float, years: float) -> float:
    """
    Derive yield from zero-coupon bond price.

    For a zero-coupon bond: price = face_value / (1 + yield)^years
    Assuming face_value = 1, we get: yield = (1/price)^(1/years) - 1

    Args:
        price: Current price (as a decimal, e.g., 0.85 for 85%)
        years: Years to maturity

    Returns:
        Annual yield as a decimal.
    """
    if years <= 0 or price <= 0:
        return 0.0

    return (1.0 / price) ** (1.0 / years) - 1.0


def zero_duration(
    mnemonic: str, price: float, as_of_date: dt.date
) -> Optional[dict]:
    """
    Calculate duration metrics for a zero-coupon bond or actuarial liability.

    For a zero-coupon bond (or actuarial liability treated as zero):
    - Macaulay Duration = time to maturity (exact for zeros)
    - Modified Duration = Macaulay Duration / (1 + yield)
    - Dollar Duration = Modified Duration * Market Value / 100

    Args:
        mnemonic: The mnemonic (zn_*, zr_*, lr_*, ln_*, dr_*, dn_*)
        price: Current price (as a decimal, e.g., 0.85 for 85%)
        as_of_date: The valuation date

    Returns:
        Dictionary with duration metrics, or None if not a valid mnemonic.
        Keys: years_to_maturity, yield_pct, macaulay_dur, modified_dur
    """
    years = years_to_maturity(mnemonic, as_of_date)
    if years is None:
        return None

    if years <= 0:
        return {
            "years_to_maturity": 0.0,
            "yield_pct": 0.0,
            "macaulay_dur": 0.0,
            "modified_dur": 0.0,
        }

    yld = zero_yield_from_price(price, years)
    macaulay_dur = years  # Exact for zeros
    modified_dur = macaulay_dur / (1.0 + yld) if (1.0 + yld) > 0 else macaulay_dur

    return {
        "years_to_maturity": years,
        "yield_pct": yld * 100,  # Convert to percentage
        "macaulay_dur": macaulay_dur,
        "modified_dur": modified_dur,
    }


def calculate_breakeven_inflation(
    zn_price: float, zr_price: float, years: float
) -> float:
    """
    Calculate implied breakeven inflation from nominal vs real zero prices.

    The breakeven inflation is derived from:
    (1 + nominal_yield) = (1 + real_yield) * (1 + breakeven_inflation)

    For zero-coupon bonds: yield = (1/price)^(1/years) - 1
    Therefore: (zn_price / zr_price)^(1/years) = 1 / (1 + breakeven)
    So: breakeven = (zr_price / zn_price)^(1/years) - 1

    Args:
        zn_price: Nominal zero price (e.g., 0.75)
        zr_price: Real zero price (e.g., 0.80)
        years: Years to maturity

    Returns:
        Annualized breakeven inflation rate as a decimal.
    """
    if years <= 0 or zn_price <= 0 or zr_price <= 0:
        return 0.0

    # If real zero is more expensive, breakeven is positive (expected inflation)
    return (zr_price / zn_price) ** (1.0 / years) - 1.0


def adjust_real_value_for_inflation(
    value: float, breakeven: float, years: float
) -> float:
    """
    Adjust real zero value for expected inflation.

    Real zeros are priced in terms of real (inflation-adjusted) dollars.
    To get nominal value, multiply by expected inflation factor.

    Args:
        value: Current market value in real terms
        breakeven: Annualized breakeven inflation rate as a decimal
        years: Years to maturity

    Returns:
        Adjusted nominal value.
    """
    if years <= 0:
        return value

    return value * ((1.0 + breakeven) ** years)


def get_duration(
    mnemonic: str,
    price: float,
    as_of_date: dt.date,
    etf_durations: Optional[dict[str, float]] = None,
    fullname: Optional[str] = None,
    namespace: Optional[str] = None,
) -> Optional[dict]:
    """
    Get duration for any instrument (zeros/actuarial calculated, ETFs from config).

    Handles:
    - Zero-coupon bonds: zn_* (nominal), zr_* (real)
    - Actuarial liabilities: lr_*, dr_* (real), ln_*, dn_* (nominal)
    - TIPS bonds: individual bonds with maturity in fullname
    - ETFs/funds: from config file

    Args:
        mnemonic: The commodity mnemonic
        price: Current price
        as_of_date: Valuation date
        etf_durations: Optional dict of ETF durations (loaded if not provided)
        fullname: Optional fullname for parsing bond maturity dates
        namespace: Optional namespace to help identify bond type

    Returns:
        Dictionary with duration metrics:
        - modified_dur: Modified duration in years
        - source: "calculated" for zeros/actuarial/bonds, "config" for ETFs
        Returns None if no duration data available.
    """
    # Check if it's a zero-coupon bond or actuarial liability
    if (is_nominal_zero(mnemonic) or is_real_zero(mnemonic) or
            is_actuarial_nominal(mnemonic) or is_actuarial_real(mnemonic)):
        metrics = zero_duration(mnemonic, price, as_of_date)
        if metrics is not None:
            return {
                "modified_dur": metrics["modified_dur"],
                "macaulay_dur": metrics["macaulay_dur"],
                "years_to_maturity": metrics["years_to_maturity"],
                "yield_pct": metrics["yield_pct"],
                "source": "calculated",
            }

    # Check if it's a TIPS or other bond with maturity in fullname
    if is_tips_bond(mnemonic, namespace) and fullname:
        maturity = parse_bond_maturity(fullname)
        if maturity:
            days = (maturity - as_of_date).days
            if days > 0:
                years = days / 365.25
                # For coupon bonds, modified duration < maturity
                # Use approximate formula: mod_dur ≈ maturity * 0.9 for low coupon TIPS
                # This is a simplification - could be enhanced with actual coupon info
                modified_dur = years * 0.92  # Approximate for TIPS with low coupons
                return {
                    "modified_dur": modified_dur,
                    "macaulay_dur": years,
                    "years_to_maturity": years,
                    "yield_pct": None,
                    "source": "calculated",
                }

    # Check ETF config
    if etf_durations is None:
        etf_durations = load_etf_durations()

    if mnemonic in etf_durations:
        dur = etf_durations[mnemonic]
        return {
            "modified_dur": dur,
            "macaulay_dur": dur,  # Approximate for ETFs
            "years_to_maturity": None,
            "yield_pct": None,
            "source": "config",
        }

    return None


def calculate_dollar_duration(modified_dur: float, market_value: float) -> float:
    """
    Calculate dollar duration.

    Dollar duration represents the dollar change in value for a 1% (100bp)
    change in interest rates.

    Dollar Duration = Modified Duration * Market Value / 100

    Args:
        modified_dur: Modified duration in years
        market_value: Market value in dollars

    Returns:
        Dollar duration.
    """
    return modified_dur * market_value / 100.0


def get_matching_zeros(
    price_df: pd.DataFrame,
    commodity_df: pd.DataFrame,
    as_of_date: dt.date,
) -> pd.DataFrame:
    """
    Get pairs of nominal and real zeros with matching maturities for breakeven calculation.

    Args:
        price_df: DataFrame with columns ['commodity_guid', 'value'] for latest prices
        commodity_df: DataFrame with mnemonic info, indexed by guid
        as_of_date: The valuation date

    Returns:
        DataFrame with columns: maturity_date, zn_mnemonic, zr_mnemonic,
                               zn_price, zr_price, years, breakeven
    """
    # Build mnemonic -> (guid, price) lookup
    merged = price_df.join(commodity_df[["mnemonic"]], on="commodity_guid")

    zn_data = {}
    zr_data = {}

    for _, row in merged.iterrows():
        mn = row.get("mnemonic")
        if not isinstance(mn, str):
            continue

        maturity = parse_zero_maturity(mn)
        if maturity is None:
            continue

        if is_nominal_zero(mn):
            zn_data[maturity] = {"mnemonic": mn, "price": row["value"]}
        elif is_real_zero(mn):
            zr_data[maturity] = {"mnemonic": mn, "price": row["value"]}

    # Find matching pairs
    results = []
    for maturity in sorted(set(zn_data.keys()) & set(zr_data.keys())):
        zn = zn_data[maturity]
        zr = zr_data[maturity]

        years = (maturity - as_of_date).days / 365.25
        if years <= 0:
            continue

        breakeven = calculate_breakeven_inflation(zn["price"], zr["price"], years)

        results.append({
            "maturity_date": maturity,
            "zn_mnemonic": zn["mnemonic"],
            "zr_mnemonic": zr["mnemonic"],
            "zn_price": zn["price"],
            "zr_price": zr["price"],
            "years": years,
            "breakeven": breakeven,
            "breakeven_pct": breakeven * 100,
        })

    return pd.DataFrame(results)
