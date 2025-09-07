from __future__ import annotations

"""
# ─────────────────────────────────────────────────────────────────────────────
# app/core/indicators.py
# Common TA utilities (moved from app/utils/indicators.py).
# ─────────────────────────────────────────────────────────────────────────────
"""
import pandas as pd


def _require(df: pd.DataFrame, cols: set[str]) -> None:
    missing = cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")


def calculate_macd(
    df: pd.DataFrame,
    short_period: int = 12,
    long_period: int = 26,
    signal_period: int = 9,
) -> pd.DataFrame:
    """
    Classic MACD with signal and histogram.
    """
    out = df.copy()
    _require(out, {"close"})
    ema_s = out["close"].ewm(span=short_period, adjust=False).mean()
    ema_l = out["close"].ewm(span=long_period, adjust=False).mean()
    macd = ema_s - ema_l
    sig = macd.ewm(span=signal_period, adjust=False).mean()
    hist = macd - sig
    return pd.DataFrame({"macd": macd, "macd_signal": sig, "macd_histogram": hist}, index=out.index)


def calculate_bollinger_bands(
    df: pd.DataFrame,
    window: int = 20,
    num_std_dev: int = 2,
) -> pd.DataFrame:
    """
    Simple Bollinger Bands using a rolling mean and stdev.
    """
    out = df.copy()
    _require(out, {"close"})
    s = out["close"].astype(float).ffill().bfill()
    mid = s.rolling(window=window, min_periods=1).mean()
    std = s.rolling(window=window, min_periods=1).std().fillna(0.0)
    upper = mid + num_std_dev * std
    lower = mid - num_std_dev * std
    return pd.DataFrame({"bb_upper": upper, "bb_middle": mid, "bb_lower": lower}, index=out.index)
