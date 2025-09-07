"""
LLM Smart Strategy (compact)
- Computes minimal indicators (MACD, Bollinger Bands, simple MAs)
- Detects a few events
- Asks an LLM for a BUY/SELL/HOLD JSON
- Applies fixed-lot trades to keep backtests deterministic
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from agent.client import get_llm_client
from app.core.indicators import calculate_bollinger_bands, calculate_macd
from app.core.logger import BacktestLogger
from .base import SignalType, StrategyConfig, TradingSignal, TradingStrategy


@dataclass
class _Profile:
    vol_annual: float = 0.0
    trend_score: float = 0.5


def _safe_f(v: Any, default: float = 0.0) -> float:
    """
    Coerce possibly bad values to a finite float for prompts/logging.
    """
    try:
        x = float(v)
        if np.isnan(x) or np.isinf(x):
            return default
        return x
    except Exception:
        return default


class LLMSmartStrategy(TradingStrategy):
    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)
        p = self.parameters

        # LLM client
        self.llm = get_llm_client(temperature=float(p.get("temperature", 0.1)))

        # Simple indicator params
        self.ma_short = int(p.get("ma_short", 10))
        self.ma_long = int(p.get("ma_long", 20))
        self.conf_thr = float(p.get("confidence_threshold", 0.6))
        self.max_daily_trades = int(p.get("max_daily_trades", 3))

        # Position state (kept tiny on purpose)
        self.cash = float(p.get("initial_capital", 100_000.0))
        self.shares = 0
        self.current_position: Optional[str] = None

        # Optional logger
        self.logger = (
            BacktestLogger(
                db_path=p.get("log_path", "backend/data/backtest_logs.db"),
                session_id=p.get("session_id"),
            )
            if p.get("enable_logging", True)
            else None
        )

        # Housekeeping
        self._day_count: Dict[pd.Timestamp, int] = {}
        self._symbol = p.get("symbol", "UNKNOWN")

    # ---------- public ----------
    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        """
        Produce signals from a price dataframe.
        Dataframe columns expected: open, high, low, close, volume (lowercase).
        """
        df = self._prep(data)
        if len(df) < 30:
            return []

        df = self._indicators(df)
        profile = self._profile(df)
        signals: List[TradingSignal] = []

        for i in range(30, len(df)):
            ts = pd.to_datetime(df.index[i])
            day = ts.normalize()
            self._day_count[day] = self._day_count.get(day, 0)
            if self._day_count[day] >= self.max_daily_trades:
                continue

            cur = df.iloc[i]
            prev = df.iloc[i - 1]
            events = self._events(cur, prev)
            if not events:
                continue

            prompt = self._build_prompt(cur, events, profile)
            raw = self._invoke(prompt)
            dec = self._parse(raw)
            self._log(day, df.iloc[: i + 1], events, dec)

            if not dec or dec.get("action") not in {"BUY", "SELL"}:
                continue
            if float(dec.get("confidence", 0)) < self.conf_thr:
                continue

            price = _safe_f(cur["close"])
            sig = self._apply(dec["action"], ts, price, float(dec.get("confidence", 0.0)), dec)
            if sig:
                signals.append(sig)
                self._day_count[day] += 1

        return signals

    # ---------- internals ----------
    def _prep(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "date" in out.columns:
            out.index = pd.to_datetime(out["date"])
        out = out.rename(columns={c: c.lower() for c in out.columns})
        return out.sort_index()

    def _indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        m = calculate_macd(out)
        out["macd"], out["macd_signal"], out["macd_histogram"] = (
            m["macd"],
            m["macd_signal"],
            m["macd_histogram"],
        )
        b = calculate_bollinger_bands(out)
        out["bb_upper"], out["bb_middle"], out["bb_lower"] = b["bb_upper"], b["bb_middle"], b["bb_lower"]
        out[f"ma_{self.ma_short}"] = out["close"].rolling(self.ma_short).mean()
        out[f"ma_{self.ma_long}"] = out["close"].rolling(self.ma_long).mean()
        return out

    def _profile(self, df: pd.DataFrame) -> _Profile:
        if len(df) < 40:
            return _Profile()
        r = df["close"].pct_change().dropna()
        vol_annual = float(r.std() * np.sqrt(252)) if len(r) else 0.0
        # trend_score: normalized slope of last 20 closes
        y = df["close"].tail(20).to_numpy()
        x = np.arange(len(y))
        slope = (np.polyfit(x, y, 1)[0] / (y.mean() or 1.0)) if len(y) >= 2 else 0.0
        return _Profile(vol_annual=vol_annual, trend_score=float(np.clip(abs(slope), 0, 1)))

    def _events(self, cur: pd.Series, prev: pd.Series) -> List[Dict[str, Any]]:
        """
        Detect compact set of high-signal events.
        """
        ev: List[Dict[str, Any]] = []
        if cur["macd"] > cur["macd_signal"] and prev["macd"] <= prev["macd_signal"]:
            ev.append({"event_type": "MACD_GOLDEN_CROSS", "severity": "high"})
        if cur["macd"] < cur["macd_signal"] and prev["macd"] >= prev["macd_signal"]:
            ev.append({"event_type": "MACD_DEATH_CROSS", "severity": "high"})
        if cur["close"] >= cur["bb_upper"] and prev["close"] < prev["bb_upper"]:
            ev.append({"event_type": "BB_UPPER_TOUCH", "severity": "high"})
        if cur["close"] <= cur["bb_lower"] and prev["close"] > prev["bb_lower"]:
            ev.append({"event_type": "BB_LOWER_TOUCH", "severity": "high"})
        s, l = cur.get(f"ma_{self.ma_short}"), cur.get(f"ma_{self.ma_long}")
        ps, pl = prev.get(f"ma_{self.ma_short}"), prev.get(f"ma_{self.ma_long}")
        if s is not None and l is not None and ps is not None and pl is not None:
            if s > l and ps <= pl:
                ev.append({"event_type": "MA_GOLDEN_CROSS", "severity": "medium"})
            if s < l and ps >= pl:
                ev.append({"event_type": "MA_DEATH_CROSS", "severity": "medium"})
        return ev

    def _build_prompt(self, cur: pd.Series, events: List[Dict[str, Any]], prof: _Profile) -> str:
        lines = [
            "You are a strict trading decision bot.",
            "Return ONLY a JSON with keys: action, confidence, reasoning, risk_level, expected_outcome.",
            f"price={_safe_f(cur.get('close')):.2f}",
            f"macd={_safe_f(cur.get('macd')):.4f} macd_sig={_safe_f(cur.get('macd_signal')):.4f}",
            f"bb_upper={_safe_f(cur.get('bb_upper')):.2f} bb_lower={_safe_f(cur.get('bb_lower')):.2f}",
            f"profile_vol_annual={prof.vol_annual:.3f} trend_score={prof.trend_score:.2f}",
            "events: " + ", ".join(e["event_type"] for e in events),
            "Rules: prefer BUY after golden cross or lower-band reversal; SELL after death cross or upper-band exhaustion.",
            "Risk levels: low/medium/high. Confidence 0..1.",
        ]
        return "\n".join(lines)

    def _invoke(self, prompt: str) -> str:
        try:
            r = self.llm.invoke(prompt)
            return getattr(r, "content", "") or str(r)
        except Exception:
            return ""

    def _parse(self, text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        try:
            s, e = text.find("{"), text.rfind("}")
            if s >= 0 and e > s:
                return json.loads(text[s : e + 1])
        except Exception:
            return None
        return None

    def _apply(
        self, action: str, ts: pd.Timestamp, price: float, conf: float, dec: Dict[str, Any]
    ) -> Optional[TradingSignal]:
        if action == "BUY" and self.shares == 0:
            qty = 100  # fixed lot for deterministic backtests
            cost = qty * price
            if cost <= self.cash:
                self.cash -= cost
                self.shares += qty
                self.current_position = "long"
                return TradingSignal(
                    SignalType.BUY,
                    ts,
                    price,
                    qty,
                    conf,
                    f"LLM: {dec.get('reasoning','')}",
                    {"decision": dec},
                )
        if action == "SELL" and self.shares > 0:
            qty = self.shares
            self.cash += qty * price
            self.shares = 0
            self.current_position = None
            return TradingSignal(
                SignalType.SELL,
                ts,
                price,
                qty,
                conf,
                f"LLM: {dec.get('reasoning','')}",
                {"decision": dec},
            )
        return None

    def _log(
        self, day: pd.Timestamp, hist: pd.DataFrame, events: List[Dict[str, Any]], dec: Optional[Dict[str, Any]]
    ) -> None:
        """
        Minimal daily row persisted to SQLite (if logger enabled).
        """
        if not self.logger:
            return
        row = hist.iloc[-1]
        md = {
            "price": _safe_f(row.get("close")),
            "volume": int(_safe_f(row.get("volume", 0))),
        }
        llm_dec = (
            {
                "decision_made": True,
                "decision_type": dec.get("action", "HOLD"),
                "confidence": float(dec.get("confidence", 0.0)),
                "reasoning": dec.get("reasoning", ""),
            }
            if dec
            else {"decision_made": False}
        )
        self.logger.log_daily_analysis(
            symbol=self._symbol,
            date=day.strftime("%Y-%m-%d"),
            market_data=md,
            trend_analysis={"long_term": "n/a", "trend_strength": 0.0, "confidence": 0.0},
            comprehensive_technical_analysis=None,
            triggered_events=events,
            llm_decision=llm_dec,
            trading_signal=None,
            strategy_state={"position": "long" if self.current_position else "flat", "cash": float(self.cash)},
        )
