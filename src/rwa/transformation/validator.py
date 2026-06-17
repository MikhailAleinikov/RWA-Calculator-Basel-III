from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PORTFOLIO_CONFIG_PATH = PROJECT_ROOT / "config" / "portfolio.yaml"


def load_thresholds() -> dict[str, Any]:
    with open(PORTFOLIO_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["thresholds"]


THRESHOLDS = load_thresholds()
DEFAULT_DPD_DAYS = THRESHOLDS["default_dpd_days"]

# Structural defects -> row dropped from the book.
EXCLUSION_CHECKS = ["bad_ead", "re_value_invalid", "bad_maturity"]
# Non-fatal observations -> row retained, issue reported.
FLAG_CHECKS = ["pd_oob", "lgd_oob", "missing_rating", "high_ltv", "dpd_default_mismatch"]
ALL_CHECKS = EXCLUSION_CHECKS + FLAG_CHECKS


def validate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    f = pd.DataFrame(index=df.index)

    ead = pd.to_numeric(df["exposure_amount"], errors="coerce")
    pv = pd.to_numeric(df["property_value"], errors="coerce")
    loan = pd.to_numeric(df["loan_amount"], errors="coerce")
    mat = pd.to_numeric(df["maturity_years"], errors="coerce")
    pd_ = pd.to_numeric(df["PD"], errors="coerce")
    lgd = pd.to_numeric(df["LGD"], errors="coerce")
    dpd = pd.to_numeric(df["days_past_due"], errors="coerce")
    default_flag = df["default_flag"].astype(bool)

    is_re = df["property_type"].notna()
    ltv = loan / pv

    # --- EXCLUSION: structural defects ------------------------------------
    f["bad_ead"] = (ead <= 0) | ead.isna()
    f["re_value_invalid"] = is_re & (pv.isna() | (pv <= 0))
    f["bad_maturity"] = (mat <= 0) | mat.isna()
    f["pd_oob"] = pd_.notna() & ~pd_.between(0, 1)
    f["lgd_oob"] = lgd.notna() & ~lgd.between(0, 1)
    f["missing_rating"] = df["external_rating"].isna()
    f["high_ltv"] = is_re & (ltv > 1.0)
    f["dpd_default_mismatch"] = (dpd > DEFAULT_DPD_DAYS) & ~default_flag
    df = df.join(f)
    df.loc[df["dpd_default_mismatch"], "default_flag"] = True

    df["excluded"] = df[EXCLUSION_CHECKS].any(axis=1)
    return df


def quality_summary(df: pd.DataFrame, checks: list[str] | None = None) -> pd.DataFrame:
    """Count and EUR-weight each check; defaults to every check present."""
    checks = checks if checks is not None else [c for c in ALL_CHECKS if c in df.columns]
    ead = pd.to_numeric(df["exposure_amount"], errors="coerce").clip(lower=0)

    rows = [
        {
            "check": c,
            "kind": "exclusion" if c in EXCLUSION_CHECKS else "flag",
            "n": int(df[c].sum()),
            "eur": float(ead[df[c].astype(bool)].sum()),
        }
        for c in checks
    ]
    return (
        pd.DataFrame(rows)
        .sort_values(["kind", "eur"], ascending=[True, False])
        .reset_index(drop=True)
    )