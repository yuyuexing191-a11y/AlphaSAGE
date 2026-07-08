from __future__ import annotations

import re


PRICE_TOKENS = ("$open", "$close", "$high", "$low", "$vwap")


def classify_feature_group(expr_str: str) -> str:
    has_price = any(token in expr_str for token in PRICE_TOKENS)
    has_volume = "$volume" in expr_str

    if has_price and has_volume:
        return "price_volume"
    if has_volume:
        return "volume"
    if has_price:
        return "price"
    return "other"


def classify_operator_group(expr_str: str) -> str:
    if "TsCorr" in expr_str or "TsCov" in expr_str:
        return "correlation"
    if "Rank" in expr_str:
        return "rank"
    if any(op in expr_str for op in ("TsStd", "TsVar", "TsSkew", "TsKurt", "TsMad")):
        return "volatility"
    if any(op in expr_str for op in ("TsMean", "TsSum", "TsWMA", "TsEMA", "TsDelta", "TsPctChange")):
        return "trend"
    if "Greater" in expr_str or "Less" in expr_str:
        return "comparison"
    if any(op in expr_str for op in ("Add", "Sub", "Mul", "Div", "Pow", "Inv", "Log", "SLog1p")):
        return "arithmetic"
    return "other"


def classify_window_group(expr_str: str) -> str:
    windows = [int(match) for match in re.findall(r",(10|20|30|40|50)\)", expr_str)]
    if not windows:
        return "none"

    max_window = max(windows)
    if max_window <= 20:
        return "short"
    if max_window <= 40:
        return "medium"
    return "long"


def qd_bucket_key(expr_str: str) -> tuple[str, str, str]:
    return (
        classify_feature_group(expr_str),
        classify_operator_group(expr_str),
        classify_window_group(expr_str),
    )
