# ratecraft

Fixed income math: bonds, yield curves, duration, inflation, and volatility
surfaces.

`ratecraft` is a **pure calculation library** — no I/O, no data fetching, no
service surface. You bring the prices and dates; it does the fixed-income
arithmetic: pricing bonds and TIPS, bootstrapping a yield curve, deriving zero
yields and durations, backing out breakeven inflation, and fitting an
implied-volatility surface to an option chain. Data acquisition and
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

The public API (see `ratecraft.__all__`) is organized into four modules:

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

### `ratecraft.vol` — the volatility surface
- **`fit_surface`** — fits an implied-volatility surface to a normalized option
  chain (a plain frame). Each expiry is fitted independently; the surface then
  interpolates across expiries **linearly in total variance**.
- **`VolSurface`** — the fitted surface: `.iv(log_moneyness, tenor)` to
  evaluate, `.check_arbitrage()` to test it.
- **`SVIParams`** / **`SABRParams`**, **`SVISlice`** / **`SABRSlice`** — the two
  smile models and their per-expiry fits.
- **`svi_total_variance`**, **`svi_derivatives`**, **`sabr_lognormal_vol`**,
  **`durrleman_g`** — the underlying model functions.
- **`black76_price`**, **`implied_vol_black76`** — forward-based option pricing
  and its inversion, used when the chain carries prices rather than vols.

Two design choices worth knowing about:

**Interpolation is linear in total variance, not in volatility.** Interpolating
in vol is the classic way to introduce calendar arbitrage between two
individually clean expiries; linear-in-total-variance between two ordered
slices is non-decreasing in tenor by construction. Outside the fitted tenor
range the surface extrapolates at constant volatility, which is likewise
non-decreasing.

**The arbitrage conditions are checkable, and that is the point.**
`check_arbitrage()` returns the violations it finds — butterfly via
Durrleman's `g(k) >= 0`, calendar via total variance non-decreasing along the
expiry axis — so a fit either satisfies them or it does not, and a test can say
which rather than an eye on a chart.

There is no caching: compute on read. If a *measured* latency problem ever
justifies one, key it with `ratecraft.vol.FITTER_VERSION` (carried on every
fitted surface) so a change to the fitter cannot silently serve a stale surface.

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

The surface takes a chain frame with `strike`, `forward`, a `tenor` in years (or
`expiry` dates plus `asof=`), and either an `iv` column or `price` +
`option_type` to imply one:

```python
from ratecraft import fit_surface

surface = fit_surface(chain, model="svi")   # or model="sabr" for rate products
surface.iv(log_moneyness=0.0, tenor=0.75)   # ATM vol three quarters out
surface.check_arbitrage()                   # [] means clean on the grid checked
```

## Development

```bash
pip install -e '.[dev]'   # pytest + ruff
pytest                    # run the test suite (tests/)
ruff check .              # lint
```

## Role in the stack

`ratecraft` is a public, dependency-light math library vendored by the
analytics app (panoptikon) as a git submodule and installed at build time — no
credentials needed to fetch it. Its one submodule, `common` (dev-common), is
public and carries dev scaffolding only; it is **not** needed to use the
library. A recursive clone, a `pip install git+...`, and a plain non-recursive
clone all work with no auth.

## License

MIT — see [LICENSE](LICENSE).
