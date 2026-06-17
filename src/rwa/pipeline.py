"""
End-to-end RWA pipeline orchestrator.

Wires the stage packages together:

    extract     extraction.generate_portfolio   (or load a CSV)
        |
    validate    transformation.validate         (+ clean / exclude)
        |
    classify    transformation.classify         (exposure-class precedence)
        |
    assign      transformation.assign           (SA risk weight -> per-loan RWA)
        |
    aggregate   aggregation.aggregate           (totals, capital, breakdowns)
        |
    reporting      reporting.build_report          (Markdown deliverable)

Run as a module:

    python -m rwa.pipeline                       # generate + full run + write
    python -m rwa.pipeline --from-csv data/raw/portfolio.csv
    python -m rwa.pipeline --own-funds 1.6e9 --out-dir reports

Or import:

    from rwa.pipeline import run_pipeline
    out = run_pipeline(own_funds=1.6e9)
    print(out.result.headline())
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .extraction import generate_portfolio
from .transformation import validate, classify, assign
from .aggregation import aggregate, CapitalResult
from .reporting import build_report, write_report


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = PROJECT_ROOT / "reports"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed"


@dataclass(frozen=True)
class PipelineOutput:
    """Everything the run produced: the priced frame, the result, the reporting."""

    portfolio: pd.DataFrame          # per-exposure, fully audited
    result: CapitalResult            # aggregate-stage capital figures
    report_markdown: str
    paths: dict[str, Path]           # written-file paths (empty if write=False)


def run_pipeline(
    config: str | Path | dict | None = None,
    *,
    from_csv: str | Path | None = None,
    n: int | None = None,
    seed: int | None = None,
    inject_dirt: bool | None = None,
    market_rwa: float = 0.0,
    operational_rwa: float = 0.0,
    own_funds: float | None = None,
    min_capital_ratio: float = 0.08,
    write: bool = True,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    data_dir: str | Path = DEFAULT_DATA_DIR,
) -> PipelineOutput:
    """Run extract -> validate -> classify -> assign -> aggregate -> reporting.

    Either generates a synthetic portfolio (default) or loads one from
    ``from_csv``. Writes the priced portfolio and the Markdown reporting unless
    ``write=False``.
    """
    # --- extract -------------------------------------------------------------
    if from_csv is not None:
        raw = pd.read_csv(from_csv)
    else:
        raw = generate_portfolio(config, n=n, seed=seed, inject_dirt=inject_dirt)

    # --- transform -----------------------------------------------------------
    priced = assign(classify(validate(raw)))

    # --- aggregate -----------------------------------------------------------
    result = aggregate(
        priced,
        market_rwa=market_rwa,
        operational_rwa=operational_rwa,
        own_funds=own_funds,
        min_capital_ratio=min_capital_ratio,
    )

    # --- reporting --------------------------------------------------------------
    report_md = build_report(priced, result)

    paths: dict[str, Path] = {}
    if write:
        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        csv_path = data_dir / "portfolio_priced.csv"
        priced.to_csv(csv_path, index=False)
        paths["portfolio"] = csv_path
        paths["reporting"] = write_report(report_md, Path(out_dir) / "rwa_report.md")

    return PipelineOutput(priced, result, report_md, paths)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Run the end-to-end RWA pipeline.")
    ap.add_argument("--config", default=None, help="path to portfolio.yaml")
    ap.add_argument("--from-csv", default=None, help="load this CSV instead of generating")
    ap.add_argument("--n", type=int, default=None, help="override number of exposures")
    ap.add_argument("--seed", type=int, default=None, help="override RNG seed")
    ap.add_argument("--clean", action="store_true", help="disable dirt injection")
    ap.add_argument("--market-rwa", type=float, default=0.0, help="market-risk RWA add-on")
    ap.add_argument("--operational-rwa", type=float, default=0.0, help="operational-risk RWA add-on")
    ap.add_argument("--own-funds", type=float, default=None, help="own funds for the capital ratio")
    ap.add_argument("--min-ratio", type=float, default=0.08, help="minimum capital ratio")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="reporting output directory")
    ap.add_argument("--no-write", action="store_true", help="compute only, write nothing")
    args = ap.parse_args()

    out = run_pipeline(
        args.config,
        from_csv=args.from_csv,
        n=args.n,
        seed=args.seed,
        inject_dirt=False if args.clean else None,
        market_rwa=args.market_rwa,
        operational_rwa=args.operational_rwa,
        own_funds=args.own_funds,
        min_capital_ratio=args.min_ratio,
        write=not args.no_write,
        out_dir=args.out_dir,
    )

    print(out.result.headline())
    for label, p in out.paths.items():
        print(f"wrote {label:10s}: {p}")


if __name__ == "__main__":
    main()
