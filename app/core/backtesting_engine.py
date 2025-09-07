from __future__ import annotations

"""
# ─────────────────────────────────────────────────────────────────────────────
# app/core/backtesting_engine.py
# Deterministic long-only backtester. This is the old `app/backtesting.py`
# rewritten with clearer naming and comments. Public method + return schema
# are preserved to avoid breaking the UI.
# ─────────────────────────────────────────────────────────────────────────────
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import math
import numpy as np
import pandas as pd

from agent.strategies import SignalType, TradingSignal, TradingStrategy


@dataclass
class BacktestConfig:
    initial_capital: float = 100_000.0
    commission_rate: float = 0.001


class BacktestEngine:
    """
    Very small, repeatable backtest engine:
    - BUY: purchase a fixed lot if flat
    - SELL: liquidate the position
    We keep the logic intentionally simple so unit tests and the UI remain predictable.
    """

    def __init__(self, config: Optional[BacktestConfig] = None) -> None:
        self.config = config or BacktestConfig()

    def run_backtest(
        self,
        stock_data: pd.DataFrame,
        strategy: TradingStrategy,
        initial_cash: float = 100_000.0,
        transaction_cost: float = 0.001,
        symbol: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute the backtest and return a JSON-serializable dict.
        Note: key names and shapes are maintained for front-end compatibility.
        """
        if stock_data is None or stock_data.empty:
            raise ValueError("stock_data is empty")

        # Normalize and sort index
        data = stock_data.copy()
        if "date" in data.columns:
            data = data.set_index(pd.to_datetime(data["date"]))
        data = data.sort_index()

        # Soft-normalize column capitalization
        for c in ["open", "high", "low", "close", "volume"]:
            if c not in data.columns and c.capitalize() in data.columns:
                data[c] = data[c.capitalize()]
        data = data[[c for c in ["open", "high", "low", "close", "volume"] if c in data.columns]]

        # Generate all signals up front (strategy side-effect free)
        signals: List[TradingSignal] = strategy.generate_signals(data)

        # Group signals by day for execution
        sig_by_day: Dict[pd.Timestamp, List[TradingSignal]] = {}
        for s in signals:
            key = pd.to_datetime(getattr(s, "timestamp", list(data.index)[0])).normalize()
            sig_by_day.setdefault(key, []).append(s)

        # Portfolio state
        cash = float(initial_cash)
        shares = 0

        # Audit trails
        trades: List[Dict[str, Any]] = []
        hist: List[Dict[str, Any]] = []
        trade_pairs: List[Tuple[float, float]] = []  # (buy_price, sell_price)

        for ts, row in data.iterrows():
            day = pd.to_datetime(ts).normalize()
            price = float(row["close"])

            for s in sig_by_day.get(day, []):
                if s.signal_type == SignalType.BUY and shares == 0:
                    qty = 100  # fixed lot for determinism
                    eff_price = price * (1 + transaction_cost)
                    cost = qty * eff_price
                    if cost <= cash:
                        cash -= cost
                        shares += qty
                        trades.append({
                            "trade_id": f"B{len(trades):04d}",
                            "timestamp": pd.to_datetime(getattr(s, "timestamp", day)).isoformat(),
                            "symbol": symbol or "UNKNOWN",
                            "order_type": "buy",
                            "shares": qty,
                            "price": price,
                            "commission": qty * price * transaction_cost,
                            "total_cost": cost,
                            "status": "EXECUTED",
                            "signal_confidence": float(getattr(s, "confidence", 1.0)),
                            "reason": getattr(s, "reason", ""),
                        })
                elif s.signal_type == SignalType.SELL and shares > 0:
                    eff_price = price * (1 - transaction_cost)
                    proceeds = shares * eff_price
                    trade_pairs.append((trades[-1]["price"] if trades else price, price))
                    cash += proceeds
                    trades.append({
                        "trade_id": f"S{len(trades):04d}",
                        "timestamp": pd.to_datetime(getattr(s, "timestamp", day)).isoformat(),
                        "symbol": symbol or "UNKNOWN",
                        "order_type": "sell",
                        "shares": shares,
                        "price": price,
                        "commission": shares * price * transaction_cost,
                        "total_cost": proceeds,
                        "status": "EXECUTED",
                        "signal_confidence": float(getattr(s, "confidence", 1.0)),
                        "reason": getattr(s, "reason", ""),
                    })
                    shares = 0

            # Daily equity snapshot
            stock_value = shares * price
            total_value = cash + stock_value
            hist.append({
                "date": ts,
                "cash": cash,
                "position": int(shares),
                "stock_price": price,
                "stock_value": stock_value,
                "total_value": total_value,
                "cumulative_return": (total_value - initial_cash) / initial_cash,
            })

        # Liquidate on the final bar if still long
        if shares > 0:
            price = float(data.iloc[-1]["close"])
            eff_price = price * (1 - transaction_cost)
            proceeds = shares * eff_price
            trade_pairs.append((trades[-1]["price"] if trades else price, price))
            cash += proceeds
            trades.append({
                "trade_id": f"S{len(trades):04d}",
                "timestamp": pd.to_datetime(data.index[-1]).isoformat(),
                "symbol": symbol or "UNKNOWN",
                "order_type": "sell",
                "shares": shares,
                "price": price,
                "commission": shares * price * transaction_cost,
                "total_cost": proceeds,
                "status": "EXECUTED",
                "signal_confidence": 1.0,
                "reason": "finalize",
            })
            shares = 0
            hist[-1].update({
                "cash": cash,
                "position": 0,
                "stock_value": 0.0,
                "total_value": cash,
                "cumulative_return": (cash - initial_cash) / initial_cash,
            })

        # Performance metrics
        final_value = hist[-1]["total_value"]
        total_return = (final_value - initial_cash) / initial_cash
        values = [h["total_value"] for h in hist]
        cummax = np.maximum.accumulate(values)
        drawdowns = (np.array(values) - cummax) / cummax
        max_dd = float(drawdowns.min()) if len(drawdowns) else 0.0

        years = max(1, len(hist)) / 365.25
        annual_return = (final_value / initial_cash) ** (1 / years) - 1 if years > 0 else 0.0

        eq = pd.Series(values)
        daily = eq.pct_change().dropna()
        vol = float(daily.std() * math.sqrt(252)) if not daily.empty else 0.0

        wins = sum(1 for b, s in trade_pairs if s > b)
        win_rate = (wins / len(trade_pairs)) if trade_pairs else 0.0
        total_realized_pnl = sum((s - b) for b, s in trade_pairs)
        cum_trade_ret = sum(((s - b) / b) for b, s in trade_pairs) if trade_pairs else 0.0

        perf = {
            "final_value": float(final_value),
            "total_return": float(total_return),
            "annual_return": float(annual_return),
            "volatility": float(vol),
            "max_drawdown": float(max_dd),
            "num_trades": int(len(trades)),
            "win_rate": float(win_rate),
        }
        bench_ret = (data["close"].iloc[-1] / data["close"].iloc[0]) - 1
        bench = {
            "buy_hold_return": float(bench_ret),
            "strategy_return": float(total_return),
            "alpha": float(total_return - bench_ret),
            "outperformed": bool(total_return > bench_ret),
        }

        signals_json = [{
            "timestamp": pd.to_datetime(getattr(s, "timestamp", data.index[0])).isoformat(),
            "signal_type": s.signal_type.name.upper(),
            "confidence": float(getattr(s, "confidence", 1.0)),
            "price": float(getattr(s, "price", 0.0)),
            "reason": getattr(s, "reason", ""),
            "metadata": getattr(s, "metadata", {}),
        } for s in signals]

        stock_json = [{
            "timestamp": pd.to_datetime(idx).isoformat(),
            "open": float(r.get("open", 0.0)),
            "high": float(r.get("high", 0.0)),
            "low": float(r.get("low", 0.0)),
            "close": float(r.get("close", 0.0)),
            "volume": int(r.get("volume", 0) or 0),
        } for idx, r in data.iterrows()]

        stats = {
            "total_realized_pnl": float(total_realized_pnl),
            "total_trades": int(len(trades)),
            "winning_trades": int(wins),
            "strategy_win_rate": float(win_rate),
            "cumulative_trade_return_rate": float(cum_trade_ret),
        }

        return {
            "basic_info": {
                "symbol": str(symbol or "UNKNOWN"),
                "strategy_name": getattr(strategy, "name", type(strategy).__name__),
                "start_date": pd.to_datetime(data.index[0]).isoformat(),
                "end_date": pd.to_datetime(data.index[-1]).isoformat(),
                "total_days": int(len(data)),
                "initial_capital": float(initial_cash),
                "max_shares_per_trade": 100,
            },
            "performance_metrics": perf,
            "benchmark_comparison": bench,
            "trades": trades,
            "trading_signals": signals_json,
            "portfolio_history": hist,
            "trading_events": [],
            "stock_data": stock_json,
            "strategy_statistics": stats,
        }
