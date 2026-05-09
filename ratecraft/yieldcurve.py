"""
yieldcurve module

Contains class YieldCurve
"""

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
import datetime as dt
import scipy as sp
import scipy.optimize

from .bond import Bond, accrued_interest_factor, ex_coupon_days

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class YieldCurve:
    """
    Functions for dealing with yield curves
    * deriving
    * transforming
    * applying

    It is set from prices, as of a date.
    """

    def __init__(
        self,
        d0: dt.datetime,
        p: pd.DataFrame,
        basis="mean",
        ex_coupon_days=ex_coupon_days,
    ):
        """
        Initialize yield curve from prices of securities with their own terms and yields.
        Yield curve is assumed to have contant forces of interest between quoted prices.

        d0: date at which prices are given
        p: prices dataframe

            cusip   : must be either a column or the index.
            buy     : the purchase offer price.
            sell    :  the sale offer price.
            Added here:
            "bond": the Bond instance using this record, created here

        basis: 'buy' or 'sell', which column to use in dataframe to determine curve
            if 'mean': use midpoint.
            If one is zero, use nonzero one only.  It would probably be better to omit such securities entirely
                as they are likely too thinly traded to have fair bid / ask offers.
        """

        assert basis in ["mean", "buy", "sell"]
        self.basis = basis
        # Set the initial date: date of the prices
        self.d0 = pd.to_datetime(
            d0, utc=True
        )  # force utc so don't have problems vs utc-naive

        # Misc tidying up:

        # Set cusip to be the index, also have column for cusip
        if p.index.name == "cusip":
            p = p.copy()  # need a copy so do not dchange original.
        else:
            p = p.set_index("cusip")  # setting index will change the
        p["cusip"] = p.index.to_frame()
        p["buy"] = p["buy"].astype(float)
        p["sell"] = p["sell"].astype(float)

        # Replace 0 in prices with nulls so they won't hurt means
        p = p.replace({c: {0: np.nan} for c in ["buy", "sell"]})

        # Set the price column to be the correct basis
        if basis == "mean":
            p["price"] = p[["buy", "sell"]].mean(axis=1)
        else:
            if basis == "sell":
                a, b = "buy", "sell"
            else:  # basis=='buy'
                a, b = "sell", "buy"
            p["price"] = p[[a, b]].ffill(axis=1)[b]  # fill nulls with the other one.

        # to avoid issues creeping in with utc-aware and utc-naive dates being compared
        p["maturity_date"] = pd.to_datetime(p["maturity_date"], utc=True)
        p = p.sort_values("maturity_date")
        # Day count used in calculations

        # TODO: pick over utc, will only matter for near term for my purposes and those are already odd and
        # begging for review
        p["days"] = (p.maturity_date - self.d0).map(lambda dd: dd.days)

        # Bond instance column. Use a list comprehension instead of
        # p.apply(Bond, axis=1) — pandas 2.x's apply infers from the return
        # type and may expand a Bond instance into a DataFrame (because the
        # instance carries .p, .coupon_dates, etc. that pandas treats as
        # multiple values), which then fails the column assignment with
        # "Cannot set a DataFrame with multiple columns to the single
        # column bond". The comprehension keeps each Bond as one scalar.
        p["bond"] = [Bond(row) for _, row in p.iterrows()]
        self.p = p

        rates = self._prep_rates()  # prepare the attribute "rates"
        self._prep_payment_dates()  # prepare attribute 'payment_dates', facts by (maturity_date, payment_date) pair

        def _msg(*args):
            print("#" * 80, *args, sep="\n")

        # md = List of maturity dates over which will loop to get the force in between.
        # There are periods i.e. gaps in maturity over 180 days so there will be periods
        # in which a coupon is paid as well
        # as principal, unfortunately
        md = list(sorted(p.maturity_date.unique()))  # maturity dates

        # Set the initial force over the period
        # -----------------------------
        d = md[0]
        r = rates.loc[d]  # for convenience
        # Just compute directly, as will be < year.  Note: price does NOT include accrued interest.
        rates.loc[d, "force"] = np.log(
            (1 + r["coupon"]) / (r["price"] + r["accrued_interest_factor"] * r["rate"])
        )
        rates.loc[d, "force_cumul"] = rates.loc[
            d, "force"
        ]  # .cumsum() # it's a scalar, is nothing to accumulate here
        # the discount back to d0 from the start of the period
        rates.loc[d, "z"] = np.exp(-rates.loc[d, "force_cumul"])
        self.payment_dates.loc[(d, 0), "z"] = rates.loc[d, "z"]

        # Do each subsequent period: to the next maturity date.
        a = d  # the prior maturity date
        for d in md[1:]:  # We have prior period maturity date as "a"
            r = rates.loc[d]  # for convenience;

            # For each bond get p0, the price of the cash flows before the final period,
            # which period ends in principal payment: from the last maturity date to the current.

            # payment dates including current period where measuring force for maturity date d
            pmtdates = self.payment_dates.loc[d]

            # coupon pmt dates prior to current period, i.e. through prior maturity date
            prior_per_dates = pmtdates[pmtdates["payment_date"] <= a]
            current_per_dates = pmtdates[pmtdates["payment_date"] > a]

            if len(prior_per_dates):
                # See https://stackoverflow.com/questions/54304423/why-does-a-conversion-from-np-datetime64-to-float-and-back-lead-to-a-time-differ
                # cumulative forces at each payment date: none of these should be null
                # NOTE: converting dates to floats to work in numpy 2.2
                forces = np.interp(
                    prior_per_dates["d_to"],
                    rates["d_to"],
                    [
                        np.nan if x == b"" else x for x in rates["force_cumul"].values
                    ],  # kludge to  deal with numpy 2.2
                )
                z = np.exp(-forces)  # the discount factors
                # add up the present values of all of the payments for securities at this maturity date from payments.
                # Coupons are the same for each point in time, they have to be since the bonds all have the same
                # maturity date.
                p0 = r["coupon"] * z.sum()

                # in the prior period: at or before the date of the prior maturity date of a security
                # save these discount factors for convenience: set z for the payment dates for that maturity for the record and for review
                for i, _z in zip(prior_per_dates.index, z):
                    self.payment_dates.loc[(d, i), "z"] = _z

            else:
                p0 = 0  # no value before current period, as were no coupon payments

            # Set the force over the current period ending at date "d", starting at "a"

            # ... the residual PV at the start of this period: discount at constant force back to this point.
            z0 = rates.loc[
                a, "z"
            ]  # discount from BOP (the prior maturity date) to time 0

            resid_pv = (
                r["price"]
                + r["accrued_interest_factor"]
                * r["rate"]  # because accrued interest is not included in given price
                - p0
            ) / z0
            rates.loc[d, "p0"] = p0  # to keep for error checking
            rates.loc[d, "resid_pv"] = resid_pv  # to keep for error checking

            d_from = rates.loc[d, "d_from"]
            days = rates.loc[d, "days"]
            period_share = (current_per_dates["d_to"] - d_from) / days
            v = sp.optimize.fsolve(
                lambda force: current_per_dates["payment"].dot(
                    np.exp(-force * period_share)
                )
                - resid_pv,
                0.01,  # not a great estimate but prob doesn't matter as is monotonic function
            )
            rates.loc[d, "force"] = v[0]  # will be one root

            # Set the cumulative force, needed in next round
            rates.loc[d, "force_cumul"] = (
                rates.loc[a, "force_cumul"] + rates.loc[d, "force"]
            )
            rates.loc[d, "z"] = np.exp(-rates.loc[d, "force_cumul"])

            # Set the discounts of the (maturity_date, payment_date) pairs
            # self.rates = rates
            # self.present_value(pd.Series(1, index=current_per_dates.index), total=False)
            self.payment_dates.loc[d, "z"] = z0 * np.exp(
                -period_share * rates.loc[d, "force"]
            )

            # Keep this maturity date to use for the next round
            a = d

        rates["force_cumul"] = (
            rates["force_cumul"].replace(b"", np.nan).astype(float)
        )  # TODO why have to do this in numpy 2.2?
        # Run out to 100 years

        d = self.d0 + relativedelta(years=100)
        date_end = d
        prior_date = a
        if prior_date < date_end:
            # Add the ending date, go 100 years, for actuarial calculations.
            # rates.loc[date_end, 'cusip'] = []
            rates.loc[date_end, "num_bonds"] = 0
            rates.loc[date_end, "prior_date"] = prior_date
            rates.loc[date_end, "d_to"] = float((date_end - self.d0).days)
            rates.loc[date_end, "d_from"] = self.rates.loc[prior_date, "d_to"]
            rates.loc[date_end, "days"] = float((date_end - prior_date).days)
            # Get total force over last 2 years or so before final date
            last_2_years = rates.loc[prior_date + relativedelta(years=-2) : prior_date][
                ["days", "force"]
            ].sum()
            rates.loc[date_end, "force"] = rates.loc[date_end, "days"] * (
                last_2_years["force"] / last_2_years["days"]
            )
            rates.loc[date_end, "force_cumul"] = (
                rates.loc[prior_date, "force_cumul"] + rates.loc[date_end, "force"]
            )
            rates.loc[date_end, "z"] = np.exp(-rates.loc[date_end, "force_cumul"])

        rates["force_annual"] = rates["force"] / rates["days"] * 365.25
        self.rates = rates  # in case need to reset

    def yield_rate(self, date):
        """Semiannual yield to given date, after already have the forces
        Note: if date is less than a year out then gross up to a year pro rata

        TODO: duplicate what the Treasury is doing.
        https://www.federalreserve.gov/releases/h15/
        ... somewhere referenced by that.

        """
        # For convenience: to get payment dates
        b = Bond(
            pd.Series(
                {
                    "cusip": "hypothetical",
                    "buy": 100,
                    "sell": 100,
                    "end_of_day": 100,
                    "rate": 1,  # really 100%
                    "maturity_date": date,
                    "price_date": self.d0,
                }
            )
        )

        pmt_dates = b.coupon_dates
        # Days from present
        pmt_days = [(d - self.d0).days for d in pmt_dates]

        # discount factors for those dates
        z = np.exp(
            -np.interp(
                pmt_days,
                self.rates["d_to"],
                [
                    np.nan if x == b"" else x for x in self.rates["force_cumul"].values
                ],  # Kludge trying to work in numpy 2.2 TODO clean up
            )
        )

        a = 0.5 * z.sum()  # annuity factor
        aif = (
            b.accrued_interest_factor
        )  # must pay this times coupon in addition to the price of 1

        p = z[-1]  # principal factor

        # Solve for the rate: what you pay is the value of what you get
        # 1 + r * aif = r * a  + p
        r = (1 - p) / (a - aif)
        return r

    def standard_yield_curve(self):
        """Yield curve at standard points in years :
        1/12, 1/4, 1/2, 1, 2, 3, 5, 7, 10, 20, 30
        """
        return pd.Series(
            {
                m / 12: self.yield_rate(self.d0 + relativedelta(months=m))
                for m in np.concatenate(
                    [[1, 3, 6], 12 * np.array([1, 2, 3, 5, 7, 10, 20, 30])]
                )
            }
        )

    def _prep_rates(self):
        """utility function to prepare the rates attribute for results.
        This attribute is a table of facts by maturity date of bonds being fitted.
        Price, coupon are averages, equally weighted, of all bonds with that maturity date

        """

        d0, p = self.d0, self.p

        # stats by maturity date, and also for holding the computed rates

        rates = p.pivot_table(
            index="maturity_date",
            aggfunc={
                "rate": "mean",  # each bond weighted equally
                "price": "mean",  # each bond weighted equally
                "sectype": len,  # just to get the number of bonds
                "cusip": list,  # so will have a list of the cusips
            },
        )
        rates = rates.rename(columns={"sectype": "num_bonds"})  # since that's the count

        rates["coupon"] = (
            rates["rate"] / 2
        )  # coupon for total bonds with that maturity date
        rates["price"] = rates["price"] / 100  # for $1 of principal for the bonds
        rates["maturity_date"] = rates.index  # for use in computations
        rates["accrued_interest_factor"] = rates.maturity_date.map(
            lambda _: accrued_interest_factor(self.d0, _)
        )
        rates["prior_date"] = rates.maturity_date.shift()
        md = list(rates.index)  # maturity dates

        # initial date: prior date for maturity date 0 is date of yield curve: force constant from that date to the matuirty date
        rates.loc[md[0], "prior_date"] = d0
        rates["d_to"] = (rates["maturity_date"] - d0).dt.days
        rates["d_from"] = rates["d_to"].shift().fillna(0).astype(int)
        rates["days"] = (rates["maturity_date"] - rates["prior_date"]).dt.days

        # Force of interest over the interval:
        rates["force"] = np.nan
        rates["z"] = np.nan  # discount from end of period to d0, the beginning point
        rates["force_annual"] = (
            np.nan
        )  # annualized force over  the interval: set at end of the function, it isn't used in between

        # Set initial value, for interpolation before first maturity:
        rates.loc[
            self.d0,
            [
                "maturity_date",
                "price",
                "num_bonds",
                "d_to",
                "d_from",
                "force_cumul",
                "z",
            ],
        ] = [self.d0, 1, 1, 0, 0, 0, 1]
        rates = rates.sort_index()
        self.rates = rates
        return rates  # for convenience

    def _prep_payment_dates(self):
        """Prepare the facts by (maturity_date, payment_date) pair"""
        # Dates of coupon payments by maturity date, and days from the start:
        d0 = self.d0
        # ignoring initial value for rates that have no payments,
        # it's only there for interpolating.
        # make dataframe with columns for payments 0, -1, -2, etc before maturity, and
        # unstack into a series, drop payments before d0, turn into dataframe
        try:
            # Make list of payment dates to concatenate for each maturity date
            # skip first row, which was added for interpolation
            self.payment_dates = (
                pd.concat(
                    self.rates.loc[d0 + relativedelta(days=1) :].apply(
                        # Get payment dates using the Bond class,
                        # just bare bones attributes to get payments
                        lambda r: Bond(
                            pd.Series(
                                {
                                    "rate": r.rate,
                                    "maturity_date": r.maturity_date,
                                    "price_date": self.d0,
                                }
                            )
                        ),
                        axis=1,
                    )
                    # Dataframe, by maturity date, of payment dates, for each row, to get concatenated
                    .bnd.payments().to_dict(),
                    names=["maturity_date"],
                )["total"]
                .rename("payment")
                .reset_index()
                .rename(columns={"date": "payment_date"})
                .assign(
                    # Number of the coupon payment: 0 is at maturity, -1 just before, etc.
                    # Compute with a month index, divide by 6, will be an integer.
                    i=lambda df: (
                        (
                            (df.payment_date.dt.year * 12 + df.payment_date.dt.month)
                            - (
                                df.maturity_date.dt.year * 12
                                + df.maturity_date.dt.month
                            )
                        )
                        / 6
                    ).astype(int),
                    d_to=lambda df: (df.payment_date - self.d0).dt.days,
                    z=np.nan,
                )
                .set_index(["maturity_date", "i"])
            )
        except Exception as e:
            self.payment_dates = None
            logger.error(f"Could not prepare payment dates; exception {e}")

    def present_value(self, payments: pd.Series, total: bool = True):
        """
        Present value of payments. Payments is a Series of {day: value} pairs.
            THe index of the series may be an integer or a datetime.
            If datetime, the starting point is the date of the yield curve (attribute d0)

        total=True: return the total present value of payments
        total=False: return the individual payments discounted to present value, with the same
        """

        xs = payments.index  # assume works
        if isinstance(payments.index, pd.DatetimeIndex):
            # must take number of days from start of yield curve
            xs = [(i - self.d0).days for i in payments.index]

        forces = np.interp(xs, self.rates["d_to"], self.rates["force_cumul"])

        z = np.exp(-forces)  # the discount factors
        # Add up the payments as needed.

        if total:
            return payments.dot(z)
        else:
            return pd.Series(z, index=payments.index)


def cpi_factors(dates, yield_curve_real: YieldCurve, yield_curve_nominal: YieldCurve):
    """
    Return series of date -> CPI factor

    as implied by the yield curves provided, growing from the initial date of the yield curves.
    Those dates must agree.

    """
    assert yield_curve_real.d0 == yield_curve_nominal.d0
    d = yield_curve_real.d0

    tmp = {
        _: pd.Series({(_ - d).days: 1}) for _ in dates
    }  # small series for pv functions

    return pd.Series(
        {
            _d: yield_curve_real.present_value(_s)
            / yield_curve_nominal.present_value(_s)
            for _d, _s in tmp.items()
        }
    )
