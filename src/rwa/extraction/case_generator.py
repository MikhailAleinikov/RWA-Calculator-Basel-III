"""
Synthetic portfolio generator -- the *extract* stage of the RWA pipeline.

Every distribution parameter is read from ``config/portfolio.yaml``, so this
module carries no magic numbers: change the YAML, change the book. The output is
one row per exposure, with the column schema consumed by
``transformation/validator.py`` and ``transformation/classificator.py``.

Design rule (mirrors portfolio.yaml): draw the DRIVER first, then derive what
depends on it.
    counterparty_type -> EAD scale + rating distribution
    property_value    -> loan_amount (via LTV) -> overwrites EAD for RE rows
    rating            -> PD anchor + default likelihood
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

# config/portfolio.yaml relative to this file:
#   src/rwa/extraction/case_generator.py  ->  parents[3] == repo root
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "portfolio.yaml"

# Not specified in the YAML. portfolio.yaml gives dpd_if_performing as a range
# plus the note "most 0, with a small arrears tail"; this is the size of that
# tail (the fraction of performing exposures that carry 1..max arrears days).
PERFORMING_ARREARS_SHARE = 0.08

# Final column schema. The original 13 columns are preserved unchanged;
# `property_type` is added because portfolio.yaml's real_estate block is defined
# to "create residential/commercial property flags" and classificator.py's RE
# rules read `property_type`. validator.py ignores it, so nothing breaks.
COLUMNS = [
    "exposure_id", "counterparty_type", "exposure_amount", "external_rating",
    "maturity_years", "property_value", "loan_amount", "property_type",
    "collateral_type", "collateral_value", "days_past_due", "default_flag",
    "PD", "LGD",
]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load portfolio.yaml (defaults to the repo's config/portfolio.yaml)."""
    with open(path or DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _probs(d: dict[str, float], order: list[str]) -> np.ndarray:
    """Turn a {key: weight} mapping into a probability vector in `order`."""
    p = np.array([float(d.get(k, 0.0)) for k in order], dtype=float)
    return p / p.sum()


def generate_portfolio(
    config: str | Path | dict | None = None,
    *,
    n: int | None = None,
    seed: int | None = None,
    inject_dirt: bool | None = None,
) -> pd.DataFrame:
    """Generate a synthetic exposure portfolio from portfolio.yaml.

    Parameters
    ----------
    config
        Path to portfolio.yaml or a pre-loaded dict. Defaults to the repo's
        ``config/portfolio.yaml``.
    n, seed, inject_dirt
        Optional overrides of ``generation.n_exposures`` / ``.seed`` /
        ``.inject_dirt`` -- handy for tests, e.g.
        ``generate_portfolio(n=500, inject_dirt=False)``.
    """
    cfg = config if isinstance(config, dict) else load_config(config)

    gen = cfg["generation"]
    N = int(n if n is not None else gen["n_exposures"])
    seed = int(seed if seed is not None else gen["seed"])
    inject_dirt = gen["inject_dirt"] if inject_dirt is None else inject_dirt
    rng = np.random.default_rng(seed)  # one generator -> fully reproducible

    buckets = cfg["rating_buckets"]

    # --- DRIVER: counterparty type -----------------------------------------
    mix = cfg["counterparty_mix"]
    type_names = list(mix.keys())
    types = rng.choice(type_names, size=N, p=_probs(mix, type_names))

    # --- EAD: lognormal per type, with mu = ln(median_eur) -----------------
    ead_cfg = cfg["ead"]["lognormal_by_type"]
    mu = np.array([np.log(ead_cfg[t]["median_eur"]) for t in types])
    sigma = np.array([ead_cfg[t]["sigma_log"] for t in types])
    ead = rng.lognormal(mu, sigma)                       # euros
    ead = np.maximum(ead, cfg["ead"]["min_eur"])         # floor

    # --- external rating: conditional on type ------------------------------
    rating = np.empty(N, dtype=object)
    rbt = cfg["rating_by_type"]
    for t in type_names:
        m = types == t
        if not m.any():
            continue
        dist = rbt[t]
        if set(dist) == {"unrated"}:                     # retail / individual
            rating[m] = "unrated"
        else:
            rating[m] = rng.choice(buckets, size=int(m.sum()), p=_probs(dist, buckets))

    # --- real-estate selection (drives RE class + EAD overwrite) -----------
    re_cfg = cfg["real_estate"]
    re_secured = np.zeros(N, dtype=bool)
    for t, frac in re_cfg["secured_fraction_by_type"].items():
        re_secured |= (types == t) & (rng.random(N) < frac)
    n_re = int(re_secured.sum())

    # residential vs commercial flag
    property_type = np.full(N, None, dtype=object)
    is_resid = re_secured & (rng.random(N) < re_cfg["residential_share"])
    property_type[is_resid] = "residential"
    property_type[re_secured & ~is_resid] = "commercial"

    property_value = np.full(N, np.nan)
    loan_amount = np.full(N, np.nan)
    if n_re:
        pv_cfg = re_cfg["property_value"]["lognormal"]
        pv = rng.lognormal(np.log(pv_cfg["median_eur"]), pv_cfg["sigma_log"], n_re)
        lt = re_cfg["ltv"]
        ltv = rng.beta(lt["a"], lt["b"], n_re) * (lt["max"] - lt["min"]) + lt["min"]
        property_value[re_secured] = pv
        loan_amount[re_secured] = pv * ltv
        ead[re_secured] = loan_amount[re_secured]        # EAD = loan for RE rows

    # --- maturity: RE rows run long, everything else gamma -----------------
    mat = cfg["maturity"]
    d = mat["default"]
    maturity = rng.gamma(d["shape"], d["scale"], N).clip(d["min"], d["max"])
    if n_re:
        r = mat["real_estate"]
        maturity[re_secured] = rng.uniform(r["min"], r["max"], n_re)

    # --- collateral (non-RE rows only; RE collateral IS the property) ------
    col = cfg["collateral"]
    collateral_type = np.full(N, None, dtype=object)
    collateral_value = np.full(N, np.nan)
    drawn = rng.choice(col["types"], size=N, p=_probs(
        dict(zip(col["types"], col["type_probs"])), col["types"]))
    has_col = (~re_secured) & (drawn != "none")
    collateral_type[has_col] = drawn[has_col]
    vf = col["value_as_fraction_of_ead"]
    collateral_value[has_col] = ead[has_col] * rng.uniform(
        vf["min"], vf["max"], int(has_col.sum()))

    # --- default & days-past-due -------------------------------------------
    dfl = cfg["default"]
    mult = dfl["rating_multiplier"]
    p_def = np.clip(np.array([dfl["base_rate"] * mult[rb] for rb in rating]), 0.0, 1.0)
    default_flag = rng.random(N) < p_def

    dpd = np.zeros(N, dtype=int)
    dd = dfl["dpd_if_defaulted"]
    dpd[default_flag] = rng.integers(dd["min"], dd["max"] + 1, int(default_flag.sum()))
    dp = dfl["dpd_if_performing"]
    tail = (~default_flag) & (rng.random(N) < PERFORMING_ARREARS_SHARE)
    dpd[tail] = rng.integers(max(int(dp["min"]), 1), int(dp["max"]) + 1, int(tail.sum()))

    # --- IRB inputs: PD / LGD ----------------------------------------------
    irb = cfg["irb"]
    pdr = irb["pd_by_rating"]
    PD = np.array([pdr[rb] for rb in rating]) * rng.lognormal(
        0.0, irb["pd_jitter_sigma_log"], N)
    pb = irb["pd_bounds"]
    PD = np.clip(PD, pb["min"], pb["max"])
    PD[default_flag] = 1.0                               # defaulted names: PD = 1
    lg = irb["lgd"]
    LGD = rng.beta(lg["a"], lg["b"], N) * (lg["max"] - lg["min"]) + lg["min"]

    df = pd.DataFrame({
        "exposure_id": [f"EXP{i:06d}" for i in range(N)],
        "counterparty_type": types,
        "exposure_amount": ead.round(2),
        "external_rating": rating,
        "maturity_years": maturity.round(2),
        "property_value": property_value.round(2),
        "loan_amount": loan_amount.round(2),
        "property_type": property_type,
        "collateral_type": collateral_type,
        "collateral_value": collateral_value.round(2),
        "days_past_due": dpd,
        "default_flag": default_flag,
        "PD": PD.round(6),
        "LGD": LGD.round(4),
    })[COLUMNS]

    if inject_dirt:
        _inject_dirt(df, cfg["dirt"], rng)
    return df


def _inject_dirt(df: pd.DataFrame, dirt: dict, rng: np.random.Generator) -> None:
    """Salt the clean frame with the exact defects the validation stage checks.

    Each ``dirt:`` count maps 1:1 onto a validate check. Defects are placed on
    disjoint rows so the counts stay exact and independently testable.
    Mutates ``df`` in place.
    """
    N = len(df)
    used: set[int] = set()

    def take(candidates: np.ndarray, k: int) -> np.ndarray:
        cand = [int(i) for i in candidates if int(i) not in used]
        k = min(int(k), len(cand))
        if k == 0:
            return np.array([], dtype=int)
        sel = rng.choice(cand, size=k, replace=False)
        used.update(int(i) for i in sel)
        return np.asarray(sel, dtype=int)

    allrows = np.arange(N)

    # 1. non-positive EAD  (exposure_amount <= 0)
    idx = take(allrows, dirt["n_nonpositive_ead"])
    if len(idx):
        df.loc[idx, "exposure_amount"] = (
            rng.choice([0.0, -1.0], len(idx)) * rng.uniform(1, 5000, len(idx)))

    # 2. missing property_value on a real-estate row  (LTV breaks)
    re_rows = np.where(df["property_value"].notna().to_numpy())[0]
    idx = take(re_rows, dirt["n_missing_property_value"])
    if len(idx):
        df.loc[idx, "property_value"] = np.nan

    # 3. days_past_due > 90 but default_flag = False  (invariant break)
    idx = take(allrows, dirt["n_dpd_default_inconsistency"])
    if len(idx):
        df.loc[idx, "days_past_due"] = rng.integers(91, 200, len(idx))
        df.loc[idx, "default_flag"] = False

    # 4. PD outside [0, 1]
    idx = take(allrows, dirt["n_pd_out_of_range"])
    if len(idx):
        df.loc[idx, "PD"] = rng.choice([-0.05, 1.2, 1.5], len(idx))

    # 5. LGD outside [0, 1]
    idx = take(allrows, dirt["n_lgd_out_of_range"])
    if len(idx):
        df.loc[idx, "LGD"] = rng.choice([-0.05, 1.3, 1.8], len(idx))

    # 6. rated type with external_rating blanked
    rated = np.where(
        (df["external_rating"].to_numpy() != "unrated")
        & df["external_rating"].notna().to_numpy())[0]
    idx = take(rated, dirt["n_missing_rating"])
    if len(idx):
        df.loc[idx, "external_rating"] = np.nan


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Generate a synthetic RWA portfolio.")
    ap.add_argument("--config", default=None, help="path to portfolio.yaml")
    ap.add_argument("--out", default=None, help="output CSV path")
    ap.add_argument("--clean", action="store_true", help="disable dirt injection")
    args = ap.parse_args()

    df = generate_portfolio(args.config, inject_dirt=False if args.clean else None)
    out = (Path(args.out) if args.out
           else DEFAULT_CONFIG_PATH.parents[1] / "data" / "raw" / "portfolio.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {len(df):,} exposures -> {out}")


if __name__ == "__main__":
    main()