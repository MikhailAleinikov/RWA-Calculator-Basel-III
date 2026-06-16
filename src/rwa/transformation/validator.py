from dataclasses import dataclass
from typing import Callable

import pandas as pd


@dataclass(frozen=True)
class Rule:
    name: str
    predicate: Callable[[pd.Series], bool]
    exposure_class: str


def is_defaulted(r: pd.Series) -> bool:
    return bool(r.default_flag)


def is_equity(r: pd.Series) -> bool:
    """
    case_generator.py does not generate equity exposures.
    Keep the rule defined because it exists in the rule hierarchy,
    but it should never match the current generated dataset.
    """
    return False


def is_residential_re(r: pd.Series) -> bool:
    return r.counterparty_type == "individual"


def is_commercial_re(r: pd.Series) -> bool:
    """
    case_generator.py currently does not generate commercial real estate.

    Non-individual collateral types are:
        cash, securities, guarantee

    So this rule is defined but never matches the current generated dataset.
    """
    return False


def is_sovereign(r: pd.Series) -> bool:
    return r.counterparty_type == "sovereign"


def is_institution(r: pd.Series) -> bool:
    return r.counterparty_type == "institution"


def is_retail(r: pd.Series) -> bool:
    return r.counterparty_type == "retail"


def is_corporate(r: pd.Series) -> bool:
    return r.counterparty_type == "corporate"


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