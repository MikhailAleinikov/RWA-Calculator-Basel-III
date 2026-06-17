"""
Reporting -- the *reporting* stage of the RWA pipeline.

Turns the aggregate stage's ``CapitalResult`` (plus the priced, validated frame)
into a self-contained Markdown reporting: capital summary, RWA by exposure class
and by rating, portfolio composition, and a data-quality section driven by the
validator's own checks. Markdown so it renders directly on GitHub.

Presentation only -- no regulatory logic lives here. Every number is read from
upstream stages; this module just formats and narrates.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from ..aggregation import CapitalResult
from ..transformation.validator import quality_summary, EXCLUSION_CHECKS


# --- cell formatters --------------------------------------------------------
def _eur(x: float) -> str:
    return f"{x:,.0f}" if pd.notna(x) else "—"


def _pct(x: float) -> str:
    return f"{x:.2%}" if pd.notna(x) else "—"


def _int(x: float) -> str:
    return f"{int(x):,}" if pd.notna(x) else "—"


def _md_table(df: pd.DataFrame, formatters: dict) -> str:
    """Render a DataFrame as a GitHub Markdown table (no external deps)."""
    cols = list(df.columns)
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for _, r in df.iterrows():
        cells = [formatters.get(c, str)(r[c]) for c in cols]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, sep, *rows])


def _breakdown_table(bd: pd.DataFrame, key_label: str) -> str:
    """Format a by-class / by-rating breakdown, with a TOTAL row appended."""
    t = bd.reset_index().rename(
        columns={
            bd.index.name or "index": key_label,
            "n": "N",
            "ead": "EAD (EUR)",
            "rwa": "RWA (EUR)",
            "density": "Density",
            "capital": "Capital (EUR)",
            "rwa_share": "RWA share",
        }
    )
    t[key_label] = t[key_label].fillna("(unrated/NaN)").astype(str)

    total = {
        key_label: "**TOTAL**",
        "N": bd["n"].sum(),
        "EAD (EUR)": bd["ead"].sum(),
        "RWA (EUR)": bd["rwa"].sum(),
        "Density": bd["rwa"].sum() / bd["ead"].sum() if bd["ead"].sum() else float("nan"),
        "Capital (EUR)": bd["capital"].sum(),
        "RWA share": 1.0,
    }
    t = pd.concat([t, pd.DataFrame([total])], ignore_index=True)

    return _md_table(
        t,
        {
            "N": _int,
            "EAD (EUR)": _eur,
            "RWA (EUR)": _eur,
            "Density": _pct,
            "Capital (EUR)": _eur,
            "RWA share": _pct,
        },
    )


def build_report(
    df: pd.DataFrame,
    result: CapitalResult,
    *,
    title: str = "Risk-Weighted Assets — CRR3 Standardised Approach",
    as_of: date | None = None,
) -> str:
    """Build the Markdown reporting string from the priced frame and the result."""
    as_of = as_of or date.today()

    # --- capital summary -----------------------------------------------------
    summary = pd.DataFrame(
        [
            ("Exposures", f"{result.n_exposures:,} "
                          f"({result.n_priced:,} priced, {result.n_excluded:,} excluded)"),
            ("EAD, priced (EUR)", _eur(result.total_ead)),
            ("EAD, excluded (EUR)", _eur(result.excluded_ead)),
            ("Credit RWA (EUR)", _eur(result.credit_rwa)),
        ],
        columns=["Metric", "Value"],
    )
    if result.market_rwa or result.operational_rwa:
        summary = pd.concat([summary, pd.DataFrame([
            ("Market RWA (EUR)", _eur(result.market_rwa)),
            ("Operational RWA (EUR)", _eur(result.operational_rwa)),
            ("Total RWA (EUR)", _eur(result.total_rwa)),
        ], columns=["Metric", "Value"])], ignore_index=True)
    summary = pd.concat([summary, pd.DataFrame([
        ("RWA density", _pct(result.density)),
        (f"Capital requirement ({result.min_capital_ratio:.1%})", _eur(result.capital_requirement)),
    ], columns=["Metric", "Value"])], ignore_index=True)
    if result.own_funds is not None:
        status = "surplus" if (result.capital_surplus or 0) >= 0 else "**shortfall**"
        summary = pd.concat([summary, pd.DataFrame([
            ("Own funds (EUR)", _eur(result.own_funds)),
            ("Capital ratio", _pct(result.capital_ratio)),
            (f"Headroom vs requirement ({status})", _eur(abs(result.capital_surplus or 0))),
        ], columns=["Metric", "Value"])], ignore_index=True)

    # --- portfolio composition ----------------------------------------------
    priced = df[df["rwa"].notna()]
    comp = (
        priced.groupby("counterparty_type")
        .agg(N=("rwa", "size"), ead=("ead_after_crm", "sum"))
        .sort_values("ead", ascending=False)
        .reset_index()
        .rename(columns={"counterparty_type": "Counterparty type", "ead": "EAD (EUR)"})
    )

    # --- data quality --------------------------------------------------------
    q = quality_summary(df).rename(
        columns={"check": "Check", "kind": "Kind", "n": "N", "eur": "EAD (EUR)"}
    )

    parts = [
        f"# {title}",
        "",
        f"*Generated {as_of.isoformat()} · synthetic portfolio · "
        f"{result.n_exposures:,} exposures*",
        "",
        "Standardised-approach RWA and own-funds requirement under CRR3 / Basel III "
        "(Regulation (EU) 575/2013 as amended by Regulation (EU) 2024/1623), credit risk only. "
        "Figures are computed from a synthetic portfolio and are illustrative.",
        "",
        "## Capital summary",
        "",
        _md_table(summary, {}),
        "",
        "## RWA by exposure class",
        "",
        _breakdown_table(result.by_class, "Exposure class"),
        "",
        "## RWA by external rating",
        "",
        _breakdown_table(result.by_rating, "Rating"),
        "",
        "## Portfolio composition",
        "",
        _md_table(comp, {"N": _int, "EAD (EUR)": _eur}),
        "",
        "## Data quality",
        "",
        f"Rows failing an *exclusion* check ({', '.join(EXCLUSION_CHECKS)}) are dropped "
        "from the priced book; *flag* checks are retained and reported.",
        "",
        _md_table(q, {"N": _int, "EAD (EUR)": _eur}),
        "",
        "## Methodology",
        "",
        "- **Standardised approach (CRR3).** Risk weights are looked up from "
        "`config/risk_weights.yaml`; exposure classes are resolved by the ordered "
        "predicate rules in the classifier.",
        "- **Real estate.** Whole-loan LTV-bucket treatment with the IPRE / non-IPRE "
        "split introduced by the Basel III finalisation, not the legacy flat Basel II "
        "weights.",
        "- **Defaulted exposures** are routed by collateral (RE-secured vs "
        "unsecured/unknown).",
        "- **Credit risk mitigation.** Only funded real-estate collateral is modelled "
        "(via the LTV tables). Unfunded CRM, guarantees and financial-collateral "
        "haircuts are out of scope.",
        "- **Output floor.** Not applicable to a standardised-only book: the 72.5% "
        "floor only binds banks using internal models. It is a planned phase-two "
        "addition alongside an IRB leg.",
        "",
    ]
    return "\n".join(parts)


def write_report(markdown: str, path: str | Path) -> Path:
    """Write the reporting to disk, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8")
    return path
