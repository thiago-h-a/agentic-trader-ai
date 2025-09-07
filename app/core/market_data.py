from __future__ import annotations

"""
# ─────────────────────────────────────────────────────────────────────────────
# app/core/market_data.py
# Lightweight yfinance wrapper (moved from app/utils/stock_data.py).
# ─────────────────────────────────────────────────────────────────────────────
"""
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)


class StockService:
    """Minimal price loader with a small in-memory cache (30 minutes)."""

    def __init__(self) -> None:
        self._cache: dict[str, tuple[List[Dict[str, Any]], datetime]] = {}

    def get_market_data(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if not symbol:
            return []

        # Cache key depends on either a period window or [start, end] explicit range
        if start_date and end_date:
            key = f"{symbol}|{start_date}|{end_date}|{interval}"
        else:
            key = f"{symbol}|{period}|{interval}"

        # Simple TTL cache (~30 min)
        if key in self._cache:
            data, t0 = self._cache[key]
            if datetime.now() - t0 < timedelta(minutes=30):
                return data

        try:
            ticker = yf.Ticker(symbol)
            if start_date and end_date:
                # yfinance's end is exclusive; we nudge by +1 day
                end_inc = (pd.to_datetime(end_date) + pd.DateOffset(days=1)).strftime("%Y-%m-%d")
                hist = ticker.history(start=start_date, end=end_inc, interval=interval)
            else:
                hist = ticker.history(period=period, interval=interval)
            if hist.empty:
                return []

            out: List[Dict[str, Any]] = []
            for dt, row in hist.iterrows():
                out.append(
                    {
                        "date": dt.strftime("%Y-%m-%d"),
                        "open": float(row["Open"]),
                        "high": float(row["High"]),
                        "low": float(row["Low"]),
                        "close": float(row["Close"]),
                        "volume": int(row.get("Volume", 0) or 0),
                    }
                )
            self._cache[key] = (out, datetime.now())
            return out
        except Exception as e:
            log.warning(f"yfinance error for {symbol}: {e}")
            return []
