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


# ---------------------------------------------------------------------------
# Counterparty-type vocabularies. Kept as module-level sets so the precedence
# rules below read declaratively and the retail/corporate spillover logic has a
# single source of truth.
# ---------------------------------------------------------------------------
SOVEREIGN_TYPES = {
    "sovereign",
    "central_government",
    "central_gov",
    "central_bank",
    "central_gov_central_bank",
}

INSTITUTION_TYPES = {
    "institution",
    "institutions",
    "bank",
    "credit_institution",
    "financial_institution",
}

# Regulatory-retail counterparties (SMEs / small business). Subject to the
# retail size cap; an obligor above the cap spills over into `corporate`.
REGULATORY_RETAIL_TYPES = {
    "retail",
    "sme",
    "small_business",
}

# Natural persons. These map to the YAML `individual` class when they are not
# secured by real estate (those are caught earlier) and not in default.
INDIVIDUAL_TYPES = {
    "individual",
    "natural_person",
    "person",
}

CORPORATE_TYPES = {
    "corporate",
    "corporates",
    "company",
    "large_corporate",
}


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


def _counterparty(r: pd.Series) -> str:
    return _norm(_get(r, "counterparty_type", "obligor_type", "borrower_type"))


def _obligor_exposure(r: pd.Series) -> float:
    """Aggregate-to-one-obligor exposure used for the retail size cap.

    Falls back to the single-exposure amount when no aggregated figure is
    present (the synthetic generator produces one row per obligor).
    """
    amount = _get(
        r,
        "aggregate_obligor_exposure",
        "total_obligor_exposure",
        "exposure_amount",
        "ead",
        default=0,
    )
    try:
        return float(amount)
    except (TypeError, ValueError):
        return 0.0


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
    return _counterparty(r) in SOVEREIGN_TYPES


def is_institution(r: pd.Series) -> bool:
    return _counterparty(r) in INSTITUTION_TYPES


def is_retail(r: pd.Series) -> bool:
    """Regulatory retail: a retail-type obligor at or below the size cap.

    A retail/SME obligor *above* the cap is not regulatory retail; it is picked
    up by ``is_corporate`` further down the precedence list (CRR3 spillover).
    """
    if _counterparty(r) not in REGULATORY_RETAIL_TYPES:
        return False

    return _obligor_exposure(r) <= RETAIL_LIMIT_EUR


def is_corporate(r: pd.Series) -> bool:
    """Corporate counterparties, plus retail/SME obligors above the retail cap.

    The second clause encodes the CRR3 scoping rule: an SME that fails the
    regulatory-retail size test is treated under the corporate class rather
    than retail.
    """
    cp = _counterparty(r)
    if cp in CORPORATE_TYPES:
        return True

    if cp in REGULATORY_RETAIL_TYPES and _obligor_exposure(r) > RETAIL_LIMIT_EUR:
        return True

    return False


def is_individual(r: pd.Series) -> bool:
    """Natural persons not otherwise classified (not RE-secured, not defaulted)."""
    return _counterparty(r) in INDIVIDUAL_TYPES


@dataclass(frozen=True)
class Rule:
    name: str
    predicate: Callable[[pd.Series], bool]
    exposure_class: str


# ORDER = PRECEDENCE. First match wins.
#
# The `exposure_class` strings are the vocabulary of config/risk_weights.yaml,
# so the assignment stage can look up a weight directly with no translation
# layer. Precedence follows CRR3: default status overrides everything; real
# estate security overrides the counterparty class; then the counterparty
# hierarchy (sovereign -> institution -> retail -> corporate -> individual).
RULES = [
    Rule("defaulted",      is_defaulted,      "defaulted"),
    Rule("residential_re", is_residential_re, "residential_real_estate"),
    Rule("commercial_re",  is_commercial_re,  "commercial_real_estate"),
    Rule("sovereign",      is_sovereign,      "sovereign"),
    Rule("institution",    is_institution,    "institution"),
    Rule("retail",         is_retail,         "retail"),
    Rule("corporate",      is_corporate,      "corporate"),
    Rule("individual",     is_individual,     "individual"),
]

# Fallback bucket for rows no rule matches. Aligned with the YAML `other` class
# (conservative 100% weight). Equity and other CRR3 classes are intentionally
# out of scope for this synthetic project and land here if ever encountered.
FALLBACK_CLASS = "other"
FALLBACK_RULE = "no_rule_matched"


def classify_row(r: pd.Series) -> tuple[str, str]:
    for rule in RULES:
        if rule.predicate(r):
            return rule.exposure_class, rule.name

    return FALLBACK_CLASS, FALLBACK_RULE


def classify(df: pd.DataFrame) -> pd.DataFrame:
    out = df.apply(classify_row, axis=1, result_type="expand")

    df = df.copy()
    df["exposure_class"] = out[0]
    df["classification_rule"] = out[1]

    return df
