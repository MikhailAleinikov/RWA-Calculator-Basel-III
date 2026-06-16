from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PORTFOLIO_CONFIG_PATH = PROJECT_ROOT / "config" / "portfolio.yaml"


def load_thresholds() -> dict[str, Any]:
    with open(PORTFOLIO_CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    return cfg["thresholds"]


THRESHOLDS = load_thresholds()

DEFAULT_DPD_DAYS = THRESHOLDS["default_dpd_days"]
RETAIL_LIMIT_EUR = THRESHOLDS["retail_limit_eur"]
RETAIL_GRANULARITY = THRESHOLDS["retail_granularity"]


def _get(r: pd.Series, *names: str, default: Any = None) -> Any:
    """
    Return the first non-null value among possible column names.
    Useful because generated datasets often change naming slightly.
    """
    for name in names:
        if name in r and pd.notna(r[name]):
            return r[name]
    return default


def _norm(x: Any) -> str:
    """
    Normalise strings for rule checks:
    'Central Gov' -> 'central_gov'
    'commercial-real-estate' -> 'commercial_real_estate'
    """
    if x is None or pd.isna(x):
        return ""

    return str(x).strip().lower().replace("-", "_").replace(" ", "_")


def _truthy(x: Any) -> bool:
    if isinstance(x, bool):
        return x

    if x is None or pd.isna(x):
        return False

    if isinstance(x, (int, float)):
        return x != 0

    return str(x).strip().lower() in {
        "true",
        "t",
        "yes",
        "y",
        "1",
        "default",
        "defaulted",
    }


def is_defaulted(r: pd.Series) -> bool:
    default_flag = _get(
        r,
        "default_flag",
        "is_defaulted",
        "defaulted",
        default=False,
    )

    days_past_due = _get(
        r,
        "days_past_due",
        "dpd",
        default=0,
    )

    try:
        days_past_due = float(days_past_due)
    except (TypeError, ValueError):
        days_past_due = 0.0

    return _truthy(default_flag) or days_past_due > DEFAULT_DPD_DAYS


def is_equity(r: pd.Series) -> bool:
    values = {
        _norm(_get(r, "exposure_type")),
        _norm(_get(r, "asset_class")),
        _norm(_get(r, "instrument_type")),
        _norm(_get(r, "counterparty_type")),
    }

    return bool(values & {"equity", "equities", "share", "shares"})


def is_residential_re(r: pd.Series) -> bool:
    values = {
        _norm(_get(r, "property_type")),
        _norm(_get(r, "real_estate_type")),
        _norm(_get(r, "collateral_property_type")),
        _norm(_get(r, "collateral_type")),
    }

    return bool(
        values
        & {
            "residential",
            "residential_re",
            "residential_real_estate",
            "secured_by_residential_re",
        }
    )


def is_commercial_re(r: pd.Series) -> bool:
    values = {
        _norm(_get(r, "property_type")),
        _norm(_get(r, "real_estate_type")),
        _norm(_get(r, "collateral_property_type")),
        _norm(_get(r, "collateral_type")),
    }

    return bool(
        values
        & {
            "commercial",
            "commercial_re",
            "commercial_real_estate",
            "secured_by_commercial_re",
        }
    )


def is_sovereign(r: pd.Series) -> bool:
    counterparty_type = _norm(
        _get(
            r,
            "counterparty_type",
            "obligor_type",
            "borrower_type",
        )
    )

    return counterparty_type in {
        "sovereign",
        "central_government",
        "central_gov",
        "central_bank",
        "central_gov_central_bank",
    }


def is_institution(r: pd.Series) -> bool:
    counterparty_type = _norm(
        _get(
            r,
            "counterparty_type",
            "obligor_type",
            "borrower_type",
        )
    )

    return counterparty_type in {
        "institution",
        "institutions",
        "bank",
        "credit_institution",
        "financial_institution",
    }


def is_retail(r: pd.Series) -> bool:
    counterparty_type = _norm(
        _get(
            r,
            "counterparty_type",
            "obligor_type",
            "borrower_type",
        )
    )

    if counterparty_type not in {
        "retail",
        "individual",
        "natural_person",
        "person",
        "sme",
    }:
        return False

    exposure_amount = _get(
        r,
        "aggregate_obligor_exposure",
        "total_obligor_exposure",
        "exposure_amount",
        "ead",
        default=0,
    )

    try:
        exposure_amount = float(exposure_amount)
    except (TypeError, ValueError):
        return False

    return exposure_amount <= RETAIL_LIMIT_EUR


def is_corporate(r: pd.Series) -> bool:
    counterparty_type = _norm(
        _get(
            r,
            "counterparty_type",
            "obligor_type",
            "borrower_type",
        )
    )

    return counterparty_type in {
        "corporate",
        "corporates",
        "company",
        "large_corporate",
    }


@dataclass(frozen=True)
class Rule:
    name: str
    predicate: Callable[[pd.Series], bool]
    exposure_class: str


# ORDER = PRECEDENCE. First match wins.
RULES = [
    Rule("defaulted",      is_defaulted,      "exposures_in_default"),
    Rule("equity",         is_equity,         "equity"),
    Rule("residential_re", is_residential_re, "secured_by_residential_re"),
    Rule("commercial_re",  is_commercial_re,  "secured_by_commercial_re"),
    Rule("sovereign",      is_sovereign,      "central_gov_central_bank"),
    Rule("institution",    is_institution,    "institutions"),
    Rule("retail",         is_retail,         "retail"),
    Rule("corporate",      is_corporate,      "corporates"),
]


def classify_row(r: pd.Series) -> tuple[str, str]:
    for rule in RULES:
        if rule.predicate(r):
            return rule.exposure_class, rule.name

    return "unclassified", "no_rule_matched"


def classify(df: pd.DataFrame) -> pd.DataFrame:
    out = df.apply(classify_row, axis=1, result_type="expand")

    df = df.copy()
    df["exposure_class"] = out[0]
    df["classification_rule"] = out[1]

    return df