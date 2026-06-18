"""
rwa -- a CRR3 / Basel III standardised-approach risk-weighted-assets calculator.

A seven-stage pipeline over a synthetic (or real) loan portfolio:

    extract    generate_portfolio        synthetic book from config/portfolio.yaml
    validate   validate                  clean, correct, exclude unpriceable rows
    classify   classify                  resolve CRR3 exposure class (ordered rules)
    assign     assign                    SA risk weight -> per-loan RWA
    aggregate  aggregate                 totals, capital requirement, breakdowns
    report     build_report              Markdown deliverable
    (orchestrated by) run_pipeline

Public names are imported lazily (PEP 562): nothing under the sub-packages is
pulled in until first accessed, which keeps ``import rwa`` cheap and avoids
re-importing a stage module that is simultaneously being run via
``python -m rwa.<stage>``.

Typical use:

    from rwa import run_pipeline
    out = run_pipeline(own_funds=1.6e9)
    print(out.result.headline())
"""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

__version__ = "0.1.0"

# Public attribute -> the sub-module that defines (and re-exports) it.
_LAZY: dict[str, str] = {
    # orchestrator
    "run_pipeline": "pipeline",
    "PipelineOutput": "pipeline",
    # extract
    "generate_portfolio": "extraction",
    # transform
    "validate": "transformation",
    "quality_summary": "transformation",
    "classify": "transformation",
    "assign": "transformation",
    "summarise_rwa": "transformation",
    "load_risk_weights": "transformation",
    # aggregate
    "aggregate": "aggregation",
    "CapitalResult": "aggregation",
    "MIN_CAPITAL_RATIO": "aggregation",
    # report
    "build_report": "reporting",
    "write_report": "reporting",
}

__all__ = ["__version__", *sorted(_LAZY)]


def __getattr__(name: str):
    """Resolve a public name to its sub-module on first access (PEP 562)."""
    sub = _LAZY.get(name)
    if sub is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f".{sub}", __name__)
    return getattr(module, name)


def __dir__() -> list[str]:
    return sorted(__all__)


# Eager, zero-cost imports for type checkers and IDE autocompletion only.
if TYPE_CHECKING:
    from .aggregation import CapitalResult, MIN_CAPITAL_RATIO, aggregate
    from .extraction import generate_portfolio
    from .pipeline import PipelineOutput, run_pipeline
    from .reporting import build_report, write_report
    from .transformation import (
        assign,
        classify,
        load_risk_weights,
        quality_summary,
        summarise_rwa,
        validate,
    )