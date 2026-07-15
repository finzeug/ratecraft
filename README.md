# ratecraft

Fixed income math: bonds, yield curves, duration, and inflation.

`ratecraft` is a **pure calculation library** — no I/O, no data fetching, no
service surface. You bring the prices and dates; it does the fixed-income
arithmetic: pricing bonds and TIPS, bootstrapping a yield curve, deriving zero
yields and durations, and backing out breakeven inflation. Data acquisition and
presentation live in the consuming applications; this library is just the math.

## Install

```bash
pip install .            # from a checkout
# or, as vendored in the stack:
#   the consuming repo carries ratecraft as a git submodule (e.g. lib/ratecraft)
#   and installs it in its build.
```

Requires Python >= 3.11. Runtime dependencies: `numpy`, `pandas`, `scipy`,
`python-dateutil`, `pyyaml`.

## What's inside

The public API (see `ratecraft.__all__`) is organized into three modules:

### `ratecraft.bond` — instruments
- **`Bond`** — a coupon bond built from a price record (a pandas `Series` with
  `price_date`, `maturity_date`, `rate`, and optional price columns); put on a
  $1-principal basis.
- **`TIPS`** — inflation-linked bond (a `Bond` subclass).
- **`BondAccessor`** — accessor for working with bonds off a frame.
- **`prior_coupon_date`**, **`accrued_interest_factor`**, **`ex_coupon_days`** —
  coupon-schedule and accrued-interest helpers.

### `ratecraft.yieldcurve` — the curve
- **`YieldCurve`** — bootstraps a curve from dated security prices, assuming
  constant forces of interest between quotes. Built from a date `d0` and a
  prices `DataFrame`.
- **`cpi_factors`** — the CPI factor series implied by a real and a nominal
  yield curve.

### `ratecraft.duration` — duration & inflation
- **`zero_duration`** — duration metrics for a zero-coupon bond or actuarial
  liability.
- **`zero_yield_from_price`** — yield implied by a zero's price and term.
- **`calculate_breakeven_inflation`** — breakeven inflation from nominal vs.
  real zero prices.
- **`calculate_dollar_duration`**, **`get_duration`**, **`get_matching_zeros`**,
  **`load_etf_durations`** — dollar duration, duration lookup by instrument
  mnemonic, zero-matching, and ETF duration config loading.

## Quick example

The scalar helpers take and return plain floats:

```python
from ratecraft import (
    zero_yield_from_price,
    calculate_breakeven_inflation,
    calculate_dollar_duration,
)

# Yield of a 10-year zero trading at 74.41 (per 100 face):
y = zero_yield_from_price(price=0.7441, years=10)      # ~0.0300 (3.0%)

# Breakeven inflation from a nominal vs. a real 10-year zero price:
be = calculate_breakeven_inflation(zn_price=0.7441, zr_price=0.8203, years=10)

# Dollar duration from a modified duration and a market value:
dd = calculate_dollar_duration(modified_dur=7.2, market_value=1_000_000)
```

The richer `Bond` / `TIPS` / `YieldCurve` types consume pandas price records and
frames — see the docstrings on each for the exact expected columns.

## Development

```bash
pip install -e '.[dev]'   # pytest + ruff
pytest                    # run the test suite (tests/)
ruff check .              # lint
```

## Role in the stack

`ratecraft` is a public, dependency-light math library vendored by the
analytics app (panoptikon) as a git submodule and installed at build time — no
credentials needed to fetch it. It carries the shared `common` (public) and
`stack-common` (private, dev tooling only) submodules for consistent dev
scaffolding; those are **not** needed to use the library — a non-recursive
clone or a plain `pip install` gets you the math with no auth.
