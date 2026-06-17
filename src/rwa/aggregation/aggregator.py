"""
Capital aggregation -- the *aggregate* stage of the RWA pipeline.

Rolls the per-exposure risk-weighted assets from the assign stage up to
portfolio totals: total RWA, the Pillar 1 minimum capital requirement, RWA
density, and breakdowns by exposure class and by rating.

Approach-agnostic by design: it consumes whatever ``rwa`` column it is given.
Today that is the standardised-approach credit RWA. When an IRB leg and the
CRR3 output floor are added, the floored credit RWA replaces it here with no
change to this module -- the floor is computed upstream, aggregation downstream.

Market and operational risk are accepted as scalar add-ons so the capital ratio
can sit on a full risk-exposure base once those legs exist. Both default to
zero, giving a credit-risk-only view.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


# CRR / Basel III Pillar 1 total own-funds requirement. The capital conservation
# buffer (+2.5%) and other buffers sit on top of this; kept out so the headline
# figure is the hard regulatory minimum, overridable via min_capital_ratio.
MIN_CAPITAL_RATIO = 0.08


@dataclass(frozen=True)
class CapitalResult:
    """Structured output of the aggregate stage. DataFrames carry the cuts."""

    n_exposures: int
    n_priced: int
    n_excluded: int
    total_ead: float
    excluded_ead: float
    credit_rwa: float
    market_rwa: float
    operational_rwa: float
    total_rwa: float
    min_capital_ratio: float
    capital_requirement: float
    density: float
    by_class: pd.DataFrame
    by_rating: pd.DataFrame
    own_funds: float | None = None
    capital_ratio: float | None = None
    capital_surplus: float | None = None

    def headline(self) -> str:
        lines = [
            f"exposures           : {self.n_exposures:,}  "
            f"({self.n_priced:,} priced, {self.n_excluded:,} excluded)",
            f"EAD (priced)        : EUR {self.total_ead:,.0f}",
            f"credit RWA          : EUR {self.credit_rwa:,.0f}",
        ]
        if self.market_rwa or self.operational_rwa:
            lines += [
                f"market RWA          : EUR {self.market_rwa:,.0f}",
                f"operational RWA     : EUR {self.operational_rwa:,.0f}",
                f"total RWA           : EUR {self.total_rwa:,.0f}",
            ]
        lines += [
            f"RWA density         : {self.density:.2%}",
            f"capital req ({self.min_capital_ratio:.1%})  : EUR {self.capital_requirement:,.0f}",
        ]
        if self.own_funds is not None:
            status = "surplus" if (self.capital_surplus or 0) >= 0 else "SHORTFALL"
            lines += [
                f"own funds           : EUR {self.own_funds:,.0f}",
                f"capital ratio       : {self.capital_ratio:.2%}",
                f"{status:20s}: EUR {abs(self.capital_surplus or 0):,.0f}",
            ]
        return "\n".join(lines)


def _breakdown(priced: pd.DataFrame, by: str, rwa_base: float, min_ratio: float) -> pd.DataFrame:
    """Per-group n / EAD / RWA / density / capital / share, sorted by RWA."""
    g = priced.groupby(by, dropna=False)
    out = g.agg(
        n=("rwa", "size"),
        ead=("ead_after_crm", "sum"),
        rwa=("rwa", "sum"),
    )
    out["density"] = out["rwa"] / out["ead"]
    out["capital"] = min_ratio * out["rwa"]
    out["rwa_share"] = out["rwa"] / rwa_base if rwa_base else float("nan")
    return out.sort_values("rwa", ascending=False)


def aggregate(
    df: pd.DataFrame,
    *,
    market_rwa: float = 0.0,
    operational_rwa: float = 0.0,
    own_funds: float | None = None,
    min_capital_ratio: float = MIN_CAPITAL_RATIO,
) -> CapitalResult:
    """Aggregate a priced portfolio into capital figures and breakdowns.

    Expects the columns the assign stage produces (``rwa``, ``ead_after_crm``,
    ``exposure_class``). Rows the validator excluded carry NaN ``rwa`` and are
    reported separately rather than counted into RWA.

    Parameters
    ----------
    market_rwa, operational_rwa
        Optional risk-type add-ons. Default 0 -> credit-only view.
    own_funds
        If supplied, the actual capital ratio and surplus/shortfall vs the
        minimum requirement are computed.
    min_capital_ratio
        Pillar 1 minimum (default 8%).
    """
    if "rwa" not in df.columns:
        raise KeyError("aggregate() needs an 'rwa' column; run assign() first.")

    priced = df[df["rwa"].notna()].copy()
    excluded = df[df["rwa"].isna()]

    credit_rwa = float(priced["rwa"].sum())
    total_rwa = credit_rwa + float(market_rwa) + float(operational_rwa)
    total_ead = float(priced["ead_after_crm"].sum())

    if len(excluded) and "exposure_amount" in excluded.columns:
        excluded_ead = float(
            pd.to_numeric(excluded["exposure_amount"], errors="coerce").clip(lower=0).sum()
        )
    else:
        excluded_ead = 0.0

    capital_requirement = min_capital_ratio * total_rwa
    density = credit_rwa / total_ead if total_ead else float("nan")

    # Shares are expressed against credit RWA so the class/rating cuts sum to 1
    # regardless of any market/op add-ons.
    by_class = _breakdown(priced, "exposure_class", credit_rwa, min_capital_ratio)
    by_rating = _breakdown(priced, "external_rating", credit_rwa, min_capital_ratio)

    capital_ratio = capital_surplus = None
    if own_funds is not None:
        capital_ratio = own_funds / total_rwa if total_rwa else float("nan")
        capital_surplus = own_funds - capital_requirement

    return CapitalResult(
        n_exposures=len(df),
        n_priced=len(priced),
        n_excluded=len(excluded),
        total_ead=total_ead,
        excluded_ead=excluded_ead,
        credit_rwa=credit_rwa,
        market_rwa=float(market_rwa),
        operational_rwa=float(operational_rwa),
        total_rwa=total_rwa,
        min_capital_ratio=min_capital_ratio,
        capital_requirement=capital_requirement,
        density=density,
        by_class=by_class,
        by_rating=by_rating,
        own_funds=own_funds,
        capital_ratio=capital_ratio,
        capital_surplus=capital_surplus,
    )
