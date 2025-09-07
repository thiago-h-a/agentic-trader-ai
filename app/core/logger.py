from __future__ import annotations

"""
# ─────────────────────────────────────────────────────────────────────────────
# app/core/logger.py
# Small SQLite logger (moved from app/utils/backtest_logger.py).
# Only the columns actually used by the API are retained.
# ─────────────────────────────────────────────────────────────────────────────
"""
import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class _Row:
    session_id: str
    symbol: str
    date: str
    timestamp: str
    price: Optional[float]
    volume: Optional[int]
    trend_analysis: Optional[Dict[str, Any]]
    comprehensive_technical_analysis: Optional[Dict[str, Any]]
    triggered_events: Optional[List[Dict[str, Any]]]
    llm_decision: Optional[Dict[str, Any]]
    trading_signal: Optional[Dict[str, Any]]
    strategy_state: Optional[Dict[str, Any]]


class BacktestLogger:
    """Tiny, dependency-free SQLite logger."""

    def __init__(self, db_path: str = "backend/data/backtest_logs.db", session_id: str | None = None) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id or str(uuid.uuid4())
        self._init()

    def _init(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS daily_analysis_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    date TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    price REAL,
                    volume INTEGER,
                    trend_analysis TEXT,
                    comprehensive_technical_analysis TEXT,
                    triggered_events TEXT,
                    llm_decision TEXT,
                    trading_signal TEXT,
                    strategy_state TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_session ON daily_analysis_logs(session_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol_date ON daily_analysis_logs(symbol, date)")

    def log_daily_analysis(
        self,
        symbol: str,
        date: str,
        market_data: Dict[str, Any],
        trend_analysis: Dict[str, Any] | None = None,
        comprehensive_technical_analysis: Dict[str, Any] | None = None,
        triggered_events: List[Dict[str, Any]] | None = None,
        llm_decision: Dict[str, Any] | None = None,
        trading_signal: Dict[str, Any] | None = None,
        strategy_state: Dict[str, Any] | None = None,
    ) -> int:
        row = _Row(
            session_id=self.session_id,
            symbol=symbol,
            date=date,
            timestamp=datetime.now().isoformat(),
            price=_to_float(market_data.get("price")),
            volume=_to_int(market_data.get("volume")),
            trend_analysis=trend_analysis,
            comprehensive_technical_analysis=comprehensive_technical_analysis,
            triggered_events=triggered_events,
            llm_decision=llm_decision,
            trading_signal=trading_signal,
            strategy_state=strategy_state,
        )
        with sqlite3.connect(self.db_path) as conn:
            # Keep a single row per (symbol, date) for simplicity
            conn.execute("DELETE FROM daily_analysis_logs WHERE symbol = ? AND date = ?", (symbol, date))
            conn.execute(
                """
                INSERT INTO daily_analysis_logs (
                    session_id, symbol, date, timestamp, price, volume,
                    trend_analysis, comprehensive_technical_analysis,
                    triggered_events, llm_decision, trading_signal, strategy_state
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row.session_id,
                    row.symbol,
                    row.date,
                    row.timestamp,
                    row.price,
                    row.volume,
                    _dumps(row.trend_analysis),
                    _dumps(row.comprehensive_technical_analysis),
                    _dumps(row.triggered_events),
                    _dumps(row.llm_decision),
                    _dumps(row.trading_signal),
                    _dumps(row.strategy_state),
                ),
            )
            return int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])

    def query_logs(
        self,
        symbol: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int | None = 100,
    ) -> List[Dict[str, Any]]:
        q = ["SELECT * FROM daily_analysis_logs WHERE session_id = ?"]
        params: list[Any] = [self.session_id]
        if symbol:
            q.append("AND symbol = ?")
            params.append(symbol)
        if date_from:
            q.append("AND date >= ?")
            params.append(date_from)
        if date_to:
            q.append("AND date <= ?")
            params.append(date_to)
        q.append("ORDER BY date DESC, timestamp DESC")
        if limit:
            q.append("LIMIT ?")
            params.append(int(limit))
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(" ".join(q), params).fetchall()
        out: List[Dict[str, Any]] = []
        for r in rows:
            rec = dict(r)
            for col in (
                "trend_analysis",
                "comprehensive_technical_analysis",
                "triggered_events",
                "llm_decision",
                "trading_signal",
                "strategy_state",
            ):
                if rec.get(col):
                    try:
                        rec[col] = json.loads(rec[col])
                    except Exception:
                        rec[col] = None
            out.append(rec)
        return out

    def get_session_summary(self) -> Dict[str, Any]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """
                SELECT COUNT(*) AS total_days, COUNT(DISTINCT symbol) AS symbols_count,
                       MIN(date) AS start_date, MAX(date) AS end_date
                FROM daily_analysis_logs WHERE session_id = ?
                """,
                (self.session_id,),
            ).fetchone()
        return dict(row) if row else {}

    def export_to_json(self, filepath: str) -> None:
        data = {"session_summary": self.get_session_summary(), "logs": self.query_logs(limit=None)}
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def _dumps(x: Any) -> str | None:
    return json.dumps(x) if x is not None else None


def _to_float(x: Any) -> float | None:
    try:
        return None if x is None else float(x)
    except Exception:
        return None


def _to_int(x: Any) -> int | None:
    try:
        return None if x is None else int(x)
    except Exception:
        return None
