"""
Risk-weight assignment -- the *assign* stage of the RWA pipeline.

Reads config/risk_weights.yaml (the declarative CRR3 standardised-approach
rule-book) and stamps each classified exposure with a standardised-approach
risk weight, then RWA = EAD_after_CRM * risk_weight.

Design rule (mirrors the YAML): this module is the ENGINE, the YAML is the
RULES. No risk-weight numbers live here. The classifier has already resolved
each row to an ``exposure_class`` drawn from the YAML vocabulary; this module
only has to pick the right lookup *strategy* per class:

    sovereign / institution / corporate / retail / individual / other
        -> by_class_rating[class][rating_bucket]          (rating axis)
    residential_real_estate / commercial_real_estate
        -> by_class_ltv[class][ipre|non_ipre][ltv_bucket] (LTV axis)
    defaulted
        -> defaulted[<collateral route>][risk_weight]     (collateral axis)

Every row gains audit columns so the choice is fully traceable:
    risk_weight          the decimal SA weight applied (0.20 = 20%)
    risk_weight_source   which table/branch/bucket produced it
    ead_after_crm        EAD entering the RWA product (no unfunded CRM modelled)
    rwa                  ead_after_crm * risk_weight
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

# Reuse the classifier's RE detectors so "is this defaulted loan RE-secured?"
# has exactly one definition across the pipeline.
from .classifier import is_residential_re, is_commercial_re, _get, _norm


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RISK_WEIGHTS_CONFIG_PATH = PROJECT_ROOT / "config" / "risk_weights.yaml"


def load_risk_weights(path: str | Path | None = None) -> dict[str, Any]:
    """Load risk_weights.yaml (defaults to the repo's config/risk_weights.yaml)."""
    with open(path or RISK_WEIGHTS_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Dimension extractors
# ---------------------------------------------------------------------------
def _rating_bucket(r: pd.Series, valid_buckets: set[str]) -> str:
    """Resolve external_rating to one of the YAML rating buckets.

    Anything missing, blank, or unrecognised -> 'unrated' (the conservative SA
    default; a rated obligor whose rating was lost in the data still gets the
    unrated treatment rather than silently inheriting a favourable weight).
    """
    raw = _get(r, "external_rating", "rating", "ecai_rating")
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return "unrated"

    norm = str(raw).strip().upper().replace("-", "_")
    if norm in valid_buckets:
        return norm
    if norm in {"NR", "UNRATED", ""}:
        return "unrated"
    return "unrated"


def _ltv(r: pd.Series) -> float | None:
    """Loan-to-value for a real-estate row, or None if not computable."""
    loan = _get(r, "loan_amount", "ead", "exposure_amount")
    value = _get(r, "property_value", "collateral_value")
    try:
        loan = float(loan)
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not value or value <= 0 or math.isnan(loan) or math.isnan(value):
        return None
    return loan / value


def _is_ipre(r: pd.Series) -> bool:
    """Income-producing real estate flag.

    The synthetic generator emits no IPRE field, so this defaults to False
    (non-IPRE). The YAML carries both branches, so the engine supports IPRE the
    moment an ``ipre`` / ``income_producing`` column appears upstream.
    """
    flag = _get(r, "ipre", "is_ipre", "income_producing", "income_producing_re")
    if flag is None:
        return False
    if isinstance(flag, bool):
        return flag
    return _norm(flag) in {"true", "t", "yes", "y", "1", "ipre", "income_producing"}


# ---------------------------------------------------------------------------
# Per-strategy lookups. Each returns (risk_weight, source_label).
# ---------------------------------------------------------------------------
def _lookup_rating(table: dict, cls: str, r: pd.Series, valid_buckets: set[str]):
    sub = table["by_class_rating"][cls]
    bucket = _rating_bucket(r, valid_buckets)
    if bucket in sub:
        return float(sub[bucket]), f"by_class_rating[{cls}][{bucket}]"
    if "unrated" in sub:
        return float(sub["unrated"]), f"by_class_rating[{cls}][unrated:fallback]"
    return float(sub["fallback"]), f"by_class_rating[{cls}][fallback]"


def _lookup_ltv(table: dict, cls: str, r: pd.Series):
    block = table["by_class_ltv"][cls]
    ltv = _ltv(r)
    if ltv is None:
        return float(block["fallback_if_ltv_missing"]), f"by_class_ltv[{cls}][ltv_missing]"

    branch = "ipre" if _is_ipre(r) else "non_ipre"
    for b in block[branch]["buckets"]:
        cap = b["ltv_max"]
        if cap is None or ltv <= float(cap):
            edge = "inf" if cap is None else f"{float(cap):.2f}"
            return float(b["risk_weight"]), f"by_class_ltv[{cls}][{branch}][ltv<={edge}]"

    # Unreachable while the last bucket has ltv_max: null, but stay safe.
    return float(block["fallback_if_ltv_missing"]), f"by_class_ltv[{cls}][no_bucket]"


def _lookup_defaulted(table: dict, r: pd.Series):
    d = table["defaulted"]
    if is_residential_re(r):
        return float(d["residential_real_estate"]["risk_weight"]), "defaulted[residential_real_estate]"
    if is_commercial_re(r):
        return float(d["commercial_real_estate"]["risk_weight"]), "defaulted[commercial_real_estate]"
    return float(d["unsecured_or_unknown"]["risk_weight"]), "defaulted[unsecured_or_unknown]"


def assign_row(r: pd.Series, table: dict, valid_buckets: set[str]):
    """Return (risk_weight, source) for one classified exposure."""
    cls = _norm(_get(r, "exposure_class", default="other"))

    if cls == "defaulted":
        return _lookup_defaulted(table, r)
    if cls in table.get("by_class_ltv", {}):
        return _lookup_ltv(table, cls, r)
    if cls in table.get("by_class_rating", {}):
        return _lookup_rating(table, cls, r, valid_buckets)

    # Unknown class the YAML can't price -> conservative `other` fallback,
    # recorded so it surfaces in the audit rather than failing silently.
    rw = float(table["by_class_rating"]["other"]["fallback"])
    return rw, f"by_class_rating[other][fallback]<-unknown_class:{cls or 'missing'}"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def assign(
    df: pd.DataFrame,
    config: str | Path | dict | None = None,
    *,
    ead_column: str = "exposure_amount",
) -> pd.DataFrame:
    """Assign SA risk weights and compute RWA for a classified portfolio.

    Expects an ``exposure_class`` column (run ``classify`` first). If a boolean
    ``excluded`` column from the validator is present, excluded rows receive a
    NaN weight/RWA and are not priced.

    Adds: risk_weight, risk_weight_source, ead_after_crm, rwa.
    """
    if "exposure_class" not in df.columns:
        raise KeyError("assign() needs an 'exposure_class' column; run classify() first.")

    table = config if isinstance(config, dict) else load_risk_weights(config)
    valid_buckets = {str(b).upper() for b in table.get("rating_buckets", [])}

    df = df.copy()
    excluded = (
        df["excluded"].fillna(False).astype(bool)
        if "excluded" in df.columns
        else pd.Series(False, index=df.index)
    )

    weights: list[float] = []
    sources: list[str] = []
    for _, row in df.iterrows():
        rw, src = assign_row(row, table, valid_buckets)
        weights.append(rw)
        sources.append(src)

    df["risk_weight"] = weights
    df["risk_weight_source"] = sources

    # No unfunded credit-risk mitigation is modelled (out of scope per the YAML
    # header); funded RE collateral is already reflected in the LTV tables.
    # EAD-after-CRM therefore equals the booked exposure amount.
    df["ead_after_crm"] = pd.to_numeric(df[ead_column], errors="coerce")
    df["rwa"] = df["ead_after_crm"] * df["risk_weight"]

    # Excluded rows carry data defects that make a weight meaningless.
    df.loc[excluded, ["risk_weight", "ead_after_crm", "rwa"]] = float("nan")
    df.loc[excluded, "risk_weight_source"] = "excluded_by_validator"

    return df


def summarise_rwa(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate EAD, RWA and the EAD-weighted average risk weight per class."""
    priced = df[df["rwa"].notna()]
    g = priced.groupby("exposure_class", dropna=False)
    out = g.agg(
        n=("rwa", "size"),
        ead=("ead_after_crm", "sum"),
        rwa=("rwa", "sum"),
    )
    out["avg_risk_weight"] = out["rwa"] / out["ead"]
    out = out.sort_values("rwa", ascending=False)
    total = pd.DataFrame(
        {
            "n": [out["n"].sum()],
            "ead": [out["ead"].sum()],
            "rwa": [out["rwa"].sum()],
            "avg_risk_weight": [out["rwa"].sum() / out["ead"].sum()],
        },
        index=pd.Index(["TOTAL"], name="exposure_class"),
    )
    return pd.concat([out, total])
