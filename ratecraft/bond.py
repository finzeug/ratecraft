"""
Tools for working with bonds:
    Bond class

"""

import numpy as np
import pandas as pd
import scipy as sp
import scipy.optimize

import types
import datetime as dt
from dateutil.relativedelta import relativedelta

import logging

logger = logging.getLogger(__name__)

# In the U.S. Treasury market, bonds go ex-coupon one business day before the record date
# (which itself is typically one business day before the coupon payment date).
ex_coupon_days = 1


def prior_coupon_date(
    date: dt.datetime, maturity_date: dt.datetime, ex_coupon_days: int = ex_coupon_days
):
    """
    Get the date of the prior coupon payment as of the given date.

    The prior coupon date is the semianniversary that is on or before the date + ex_coupon_days.
    As of the given date, the coupon will not be paid if the semianniversary is ex_coupon_days after the date or before.
    Even if the semianniversary is after the given date it could be the date then.
    """
    # The semianniversary in prior year is definitely before the given date.

    # d0 = (maturity_date + relativedelta(year=date.year - 1)).date()  # to make utc naive I think
    if isinstance(date, dt.date):
        date = dt.datetime.combine(date, dt.time.min)
    dx = date + relativedelta(
        days=ex_coupon_days
    )  # TODO state definition: next coupon?
    dates_before = [
        _
        for _ in [  # [d0 + relativedelta(months=m) for m in [0, 6, 12, 18]]
            maturity_date + relativedelta(months=-6 * i)
            for i in range(2 * 31)  # are at most 30y bonds so just check all
        ]
        if _.date() <= dx.date()
    ]
    if len(dates_before):
        return max(dates_before)
    else:
        logger.error(f"No prior coupon dates for date {date}, maturity_date {maturity_date}, ex_coupon_days {ex_coupon_days}")
        return pd.NaT


def accrued_interest_factor(date, maturity_date, ex_coupon_days=ex_coupon_days):
    """ "
    Get the accrued interest factor to multiply by the annual coupon rate.
    TODO: low: just compute more cleverly.
    """
    # get lastest coupon date on or before the given date.  If today or tomorrow, will be next to no acccrued interested.  Including
    # tomorrow since ex date is tomorrow - not 100% sure that's it but close enough. TODO tie this down
    pcd = prior_coupon_date(date, maturity_date, ex_coupon_days)  #
    return (date.date() - pcd.date()).days / 365


class Bond:
    """
    Generic class for bonds

    """

    # attributes that will be set in __init__

    p = pd.DataFrame()
    coupon_dates = []
    prior_coupon_date = None
    accrued_interest = np.nan
    accrued_interest_factor = np.nan

    def __init__(self, p, ex_coupon_days=ex_coupon_days):
        """
                Initialize the price with a series p including keys from UST.get_prices:

                bond is put on basis of $1 principal value from the $100
                p: a Series of attributes.
                    minimal keys in p, all requrired for basic functions:
                        price_date
                        maturity_date
                        rate, i.e. coupon
                Optional keys in p:
        `           cusip
                    sectype
                    Prices put onto 1 basis from 100:
                        buy
                        sell
                        end_of_day
        """
        self.ex_coupon_days = ex_coupon_days

        p = p.copy()  # the price record from get_prices
        for c in ["end_of_day", "buy", "sell"]:
            if c in p:
                if (
                    p[c] > 10
                ):  # a safety threshold in case did not pass as from get_prices
                    p[c] /= 100  # to put on unitary basis
        if "end_if_day" in p:
            p["end_of_day"] = p["end_of_day"].replace({0: np.nan})
        self.p = p

        # Get coupon dates
        # 1. the last coupon date that the bond had, not the last in the series
        self.prior_coupon_date = prior_coupon_date(
            p["price_date"], p["maturity_date"], ex_coupon_days
        )

        self.coupon_dates = [
            d
            for d in sorted(
                [
                    p.maturity_date + relativedelta(months=-6 * n)
                    for n in range(
                        2 * (p.maturity_date.year - p.price_date.year + 1)
                    )  # Grab maybe too many semiannual coupons and filter
                ]
            )
            # none should be there on price date, so just in case. Also allow one day for ex post date,
            # based on inspecting ytm for reasonability.
            if d > self.prior_coupon_date
        ]

        # negative is goofy, could happen and be small.
        # Note: might be a bit inexact when there are not 365/2 days between coupons,
        # but close enough for me.
        self.accrued_interest_factor = max(
            0,
            (
                pd.to_datetime(
                    self.p.price_date
                ).date()  # kludge because sometimes came through as date already
                - self.prior_coupon_date.date()
            ).days
            / 365,
        )
        self.accrued_interest = self.p["rate"] * self.accrued_interest_factor
        # Put additional facts into series for convenience of use and display:
        self.p["accrued_interest_factor"] = self.accrued_interest_factor
        self.p["accrued_interest"] = self.accrued_interest

    def __repr__(self):
        p = self.p
        return f"mat {p['maturity_date']} rate {p['rate']:0,.2%}"

    def payments(self):
        """
        Bond payments per then-current principal, i.e. not inflated
        """
        res = pd.DataFrame(
            {"coupon": self.p["rate"] / 2, "principal": 0},
            index=pd.Index(self.coupon_dates, name="date"),
        )
        res.iloc[-1, -1] = 1
        res["total"] = res.sum(axis=1)
        return res

    def income(
        self,
        yield_curve_nominal=None,
        yield_curve_real=None,
        basis="buy",
        from_date=None,
    ):
        """
        Income by year for $1 of the bond.
        TIPS need expected_inflation (of CPI) expressed somehow.

        basis: buy|sell|end_of_day   , sometimes end_of_day is not provided in the source and is 0.
        Income components:
        1. Coupons: depend on expected inflation for TIPS
        2. Premium, discount amortization
        3. Principal growth for TIPS

        Yield curve instances have methods for getting the implied discount and therefore inflation for the coupon payment dates.

        expected inflation is nominal - real.  Express as force.
        """

        # Import module inside this function to avoid circular import
        from .yieldcurve import cpi_factors

        p = self.p  # for convenience

        if from_date is None:
            from_date = p["price_date"]
        d = from_date  # for convenience

        days = (p.maturity_date - d).days

        # prices at points we need: just say at coupon dates. Difference is also income.
        # dates we need: current date and coupon dates
        ds = [d] + self.coupon_dates
        res = pd.DataFrame({"date": {_: _ for _ in ds}}).rename_axis(index="date")

        res["y"] = res["date"].dt.year  # helps when need tax year

        res["coupon_pct"] = pd.Series({_: p["rate"] for _ in self.coupon_dates})
        res["coupon_pct"] = res["coupon_pct"].fillna(0)
        res["days"] = [(_ - d).days for _ in ds]
        res["price"] = p[basis] * (1 / p[basis]) ** (
            res["days"] / days
        )  # is accreted income

        if yield_curve_nominal is None or yield_curve_real is None:
            res["cpi_factor"] = 1
        else:
            res["cpi_factor"] = cpi_factors(ds, yield_curve_real, yield_curve_nominal)

        res["face_with_cpi"] = res[
            "cpi_factor"
        ]  # must grow from 100 with the CPI .  Used for coupon.

        # semiannual coupon depends on the CPI.
        # TODO: confirm updated semiannually with CPI not annually.
        res["coupon"] = 0.5 * res["coupon_pct"] * res["face_with_cpi"]

        # Accrued income is accretion of discount or amortization of principal, togetehr with increase in face from cpi
        res["accrued_income"] = (
            (res["price"] + res["face_with_cpi"]).diff().fillna(0)
        )  # Includes both discount accretion and cpi component: taxable, but noncash
        res["taxable_income"] = res["coupon"] + res["accrued_income"]

        res["principal"] = 0.0

        # repayment of principal with CPI
        res.loc[res.index[-1], "principal"] = res.loc[res.index[-1], "face_with_cpi"]
        res["cashflow"] = res["coupon"] + res["principal"]

        return res

    def const_rate_pv(self, rate, basis="sell"):
        """Return pv at a constant rate of interest expressed as semiannual yield

        TODO lo: refactor to make less computationally expensive, do less to get payments
        """
        force = np.log((1 + rate / 2) ** 2)
        inc = self.income(basis=basis)
        return (inc["cashflow"] * np.exp(-force * inc["days"] / 365.25)).sum()

    def ytm(self, basis="sell"):
        """ "YTM for the bond using prices at the given rate"""
        f = (  # noqa: E731
            lambda r: self.const_rate_pv(r, basis)
            - self.p[basis]
            - self.accrued_interest
        )
        try:
            x = sp.optimize.fsolve(f, 0.02)  # Solve for the yield
            return x[0]  # 'x' attribute is an array
        except Exception as e:
            logger.error(f"Exception: {e}")
            return np.nan


class TIPS(Bond):
    """Wrapper for TIPS: also adjusts for index ratio"""

    def __init__(self, p, ex_coupon_days=ex_coupon_days):
        super().__init__(p, ex_coupon_days=ex_coupon_days)
        self.index_ratio = p["IndexRatio"]
        self.price = self._get_price()

    def _get_price(self):
        """Price including accrued interest: cash price, using end-of-day price, for $1 of bond"""
        return self.index_ratio * (
            self.accrued_interest
            +
            # in case end of day not yet populated, e.g. if same day
            self.p[["sell", "end_of_day"]].astype(float).ffill()["end_of_day"]
        )


# register an accessor - merci a...
# https://stackoverflow.com/questions/70609935/is-there-a-pandas-accessor-for-whatever-is-the-underlying-value-in-the-object-in
# ... didn't understand that, had issues implementing, dodgy example, used same var name w/in a loop for a comprehension

# TODO: implement these for all attributes
# https://pandas.pydata.org/docs/development/extending.html#registering-custom-accessors


@pd.api.extensions.register_series_accessor("bnd")
class BondAccessor:
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj
        # Accessors for attributes: not for functions, which would require additional arguments

        b = self._obj.values[0]  # the first one, is a Bond instance
        for attr in b.__dict__:  # for each bond attribute:
            if not attr.startswith("__") and not isinstance(
                getattr(b, attr), types.FunctionType
            ):
                # Set attribute for the accessor : a series of that attribute
                setattr(self, attr, self._obj.map(lambda _b: getattr(_b, attr)))

    @staticmethod
    def _validate(obj):
        # verify required keys are present
        assert all(
            [isinstance(_o, Bond) for _o in obj]
        ), "All objects must be of type Bond to use the accessor"

    def ytm(self, basis="sell"):
        return self._obj.map(lambda b: b.ytm(basis))

    def payments(self):
        return self._obj.map(lambda b: b.payments())
