from __future__ import annotations

"""
# ─────────────────────────────────────────────────────────────────────────────
# app/api/routes.py (REST + SSE)
# This file is a reorganized version of the old `app/endpoints.py`.
# Public endpoints and response shapes remain the same to keep the front-end intact.
# ─────────────────────────────────────────────────────────────────────────────
"""
from datetime import datetime
from pathlib import Path
import json
import os
import queue
import sqlite3
import threading
import time
from typing import Any, Dict, Generator, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from app.core.backtesting_engine import BacktestConfig, BacktestEngine
from app.core.logger import BacktestLogger
from app.core.market_data import StockService
from agent.client import get_llm_client
from agent.strategies.base import StrategyConfig
from agent.strategies.llm_strategy import LLMSmartStrategy


# Public routers (names kept so import paths in app/main.py stay identical)
llm_stream_router = APIRouter()
backtest_router = APIRouter()
daily_router = APIRouter()


def _safe_json(obj: Any) -> str:
    """
    JSON dumps that handles numpy/pandas/datetime cleanly.
    """
    def _default(o: Any):
        if isinstance(o, (pd.Timestamp, datetime)):
            return o.isoformat()
        if hasattr(o, "to_dict"):
            return o.to_dict()
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        return str(o)

    return json.dumps(obj, default=_default, ensure_ascii=False)


def _db_path(p: Optional[str]) -> str:
    """
    Ensure the parent folder of the SQLite DB exists before use.
    """
    path = p or "backend/data/backtest_logs.db"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return path


# ----------------------- LLM stream (SSE) -----------------------
@llm_stream_router.get("/llm-backtest-stream")
async def llm_stream(
    symbol: str = Query(...),
    period: str = Query("1y"),
    max_position_size: float = Query(0.3),
    stop_loss: float = Query(0.05),
    take_profit: float = Query(0.1),
):
    """
    Streams backtest progress via Server-Sent Events and then emits a final result payload.
    The query parameters are accepted for UI compatibility; the current strategy uses a fixed lot to stay deterministic.
    """
    initial_capital = 100_000.0

    def gen() -> Generator[str, None, None]:
        q: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue()

        def progress(day: int, total: int, kind: str, msg: str, extra: Dict[str, Any] | None = None):
            # Keep shape/keys for the front-end listener
            payload = {
                "type": "trading_progress",
                "day": day,
                "total_days": total,
                "progress": round(day / max(1, total) * 100, 1),
                "event_type": kind,
                "message": msg,
            }
            if extra:
                payload.update(extra)
            q.put(payload)

        def worker():
            try:
                q.put({"type": "progress", "step": "load", "message": f"Loading {symbol}..."})
                data = StockService().get_market_data(symbol, period)
                if not data or len(data) < 30:
                    q.put({"type": "error", "message": "Insufficient market data"})
                    return

                # Normalize dataframe and index by date
                df = pd.DataFrame(data).rename(columns=str.lower)
                df["date"] = pd.to_datetime(df["date"])
                df.set_index("date", inplace=True)
                q.put({"type": "progress", "step": "loaded", "message": f"{len(df)} days loaded"})

                # Strategy + engine (interfaces preserved)
                strat = LLMSmartStrategy(
                    StrategyConfig(
                        name="LLM Smart Strategy",
                        description="Event-driven LLM",
                        parameters={
                            "initial_capital": initial_capital,
                            "confidence_threshold": 0.6,
                            "progress_callback": progress,
                            "enable_logging": True,
                            "log_path": _db_path("backend/data/backtest_logs.db"),
                            "session_id": f"session_{symbol}_{int(time.time())}",
                            "symbol": symbol,
                        },
                    )
                )

                engine = BacktestEngine(BacktestConfig(initial_capital=initial_capital))
                q.put({"type": "progress", "step": "run", "message": "Running backtest..."})

                result = engine.run_backtest(df, strat, initial_cash=initial_capital, symbol=symbol)
                q.put({"type": "result", "data": result})
                q.put({"type": "complete"})
            except Exception as e:
                q.put({"type": "error", "message": f"{e}"})
            finally:
                q.put(None)

        # SSE stream: send a start message, then pump the queue
        yield f"data: {_safe_json({'type': 'start', 'message': 'Starting'})}\n\n"
        t = threading.Thread(target=worker, daemon=True)
        t.start()
        while True:
            msg = q.get()
            if msg is None:
                break
            yield f"data: {_safe_json(msg)}\n\n"
        t.join()

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@llm_stream_router.get("/llm-backtest-stream/status")
async def llm_stream_status():
    """
    Lightweight readiness probe for the SSE route.
    """
    return {"status": "ready", "timestamp": datetime.now().isoformat()}


# ----------------------- Backtest analysis (DB) -----------------------
@backtest_router.get("/available-dates")
async def available_dates(
    symbol: Optional[str] = Query(None),
    db_path: str = Query("backend/data/backtest_logs.db"),
):
    path = _db_path(db_path)
    with sqlite3.connect(path) as conn:
        cur = conn.cursor()
        q = "SELECT DISTINCT date FROM daily_analysis_logs"
        params: List[Any] = []
        if symbol:
            q += " WHERE symbol = ?"
            params.append(symbol)
        q += " ORDER BY date DESC"
        cur.execute(q, params)
        dates = [r[0] for r in cur.fetchall()]
    dr = {"start": dates[-1], "end": dates[0]} if dates else {}
    return {"dates": dates, "date_range": dr, "total_days": len(dates)}


@backtest_router.get("/available-dates/{run_id}")
async def available_dates_by_run(
    run_id: str,
    symbol: Optional[str] = Query(None),
    db_path: str = Query("backend/data/backtest_logs.db"),
):
    """
    Backwards compatible: we pick the latest session_id, ignoring the path parameter.
    """
    path = _db_path(db_path)
    with sqlite3.connect(path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT session_id FROM daily_analysis_logs ORDER BY session_id DESC")
        rows = [r[0] for r in cur.fetchall()]
    if not rows:
        raise HTTPException(status_code=404, detail="No sessions found")
    session_id = rows[0]

    with sqlite3.connect(path) as conn:
        cur = conn.cursor()
        q = "SELECT DISTINCT date FROM daily_analysis_logs WHERE session_id = ?"
        params: List[Any] = [session_id]
        if symbol:
            q += " AND symbol = ?"
            params.append(symbol)
        q += " ORDER BY date DESC"
        cur.execute(q, params)
        dates = [r[0] for r in cur.fetchall()]
    dr = {"start": dates[-1], "end": dates[0]} if dates else {}
    return {"dates": dates, "date_range": dr, "total_days": len(dates)}


@backtest_router.get("/analysis/day/{run_id}")
async def day_analysis(
    run_id: str,
    date: str = Query(...),
    symbol: Optional[str] = Query(None),
    db_path: str = Query("backend/data/backtest_logs.db"),
    include_retrospective: bool = Query(True),
):
    """
    Fetch a single day's log and (optionally) produce a short LLM retrospective note.
    """
    # Basic YYYY-MM-DD validation
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Use YYYY-MM-DD")

    path = _db_path(db_path)

    # Latest session resolution
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT session_id FROM daily_analysis_logs ORDER BY session_id DESC")
        rows = cur.fetchall()
        if not rows:
            raise HTTPException(status_code=404, detail="No sessions")
        session_id = rows[0][0]

    logger = BacktestLogger(path, session_id=session_id)
    logs = logger.query_logs(symbol=symbol, date_from=date, date_to=date, limit=1)
    if not logs:
        raise HTTPException(status_code=404, detail=f"No data for {date}")
    log = logs[0]

    historical = {
        "date": log["date"],
        "symbol": log["symbol"],
        "price": log.get("price"),
        "market_data": log.get("market_data") or {"close": log.get("price")},
        "trend_analysis": log.get("trend_analysis"),
        "comprehensive_technical_analysis": log.get("comprehensive_technical_analysis"),
        "technical_events": log.get("triggered_events") or [],
        "llm_decision": log.get("llm_decision"),
        "strategy_state": log.get("strategy_state"),
    }

    retrospective = None
    if include_retrospective and historical.get("llm_decision"):
        retrospective = await _retrospective(historical)

    return {"historical_data": historical, "retrospective_analysis": retrospective}


async def _retrospective(day: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ask the LLM to reflect very briefly on the logged decision.
    """
    try:
        dec = day.get("llm_decision") or {}
        events = day.get("technical_events") or []
        text = f"""Evaluate the following decision very briefly (<300 words).

Date: {day.get('date')} | Symbol: {day.get('symbol')} | Price: {day.get('price')}
Decision: {dec.get('decision_type')} | Confidence: {dec.get('confidence')}
Events: {", ".join(e.get("event_type","") for e in events[:5])}
"""
        llm = get_llm_client(temperature=0.7, max_tokens=600)
        resp = llm.invoke(text)
        commentary = (getattr(resp, "content", "") or str(resp)).strip()
        return {"llm_commentary": commentary}
    except Exception as e:
        return {"llm_commentary": f"LLM failed: {e}"}


@backtest_router.get("/session-dates/{session_id}")
async def session_dates(session_id: str, db_path: str = Query("backend/data/backtest_logs.db")):
    """
    List all dates for a given session.
    """
    path = _db_path(db_path)
    with sqlite3.connect(path) as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT DISTINCT date FROM daily_analysis_logs WHERE session_id = ? ORDER BY date DESC",
            (session_id,),
        )
        dates = [r[0] for r in cur.fetchall()]
    dr = {"start": dates[-1], "end": dates[0]} if dates else {}
    return {"dates": dates, "date_range": dr, "total_days": len(dates)}


# ----------------------- Daily feedback -----------------------
@daily_router.post("/daily-feedback")
async def daily_feedback(payload: Dict[str, Any], db_path: str = Query("backend/data/backtest_logs.db")):
    """
    Accepts a free-form feedback string and returns a few LLM-generated suggestions.
    """
    feedback = payload.get("feedback", "")
    date = payload.get("date")
    symbol = payload.get("symbol")
    if not date:
        raise HTTPException(status_code=400, detail="date is required")

    path = _db_path(db_path)
    with sqlite3.connect(path) as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT DISTINCT session_id, symbol FROM daily_analysis_logs WHERE date = ? ORDER BY session_id DESC",
            (date,),
        )
        rows = cur.fetchall()
    if not rows:
        raise HTTPException(status_code=404, detail=f"No trading data for {date}")

    # Pick the most recent session, optionally matching a provided symbol
    session_id, target_symbol = rows[0]
    if symbol:
        for sid, sym in rows:
            if symbol.upper() in (sym or ""):
                session_id, target_symbol = sid, sym
                break

    logger = BacktestLogger(path, session_id=session_id)
    logs = logger.query_logs(symbol=target_symbol, date_from=date, date_to=date, limit=1)
    if not logs:
        raise HTTPException(status_code=404, detail=f"No data for {target_symbol} on {date}")
    daily = logs[0]

    strategy_text = "# Hints\nBuy on strength; sell on weakness; avoid over-trading."
    ctx = f"""Feedback: {feedback}
Date: {date}  Symbol: {target_symbol}  Price: {daily.get('price')}
Events: {", ".join(e.get("event_type","") for e in (daily.get('triggered_events') or [])[:5])}
Strategy: {strategy_text}
Please give 3 concrete, numbered suggestions."""
    llm = get_llm_client(temperature=0.6, max_tokens=600)
    try:
        r = llm.invoke(ctx)
        txt = (getattr(r, "content", "") or str(r)).strip()
    except Exception as e:
        txt = f"LLM unavailable: {e}"

    # Naive parse for "1.", "2.", ...
    suggestions = []
    for line in txt.splitlines():
        s = line.strip()
        if s[:2].isdigit() or s[:2] in {"1.", "2.", "3.", "4.", "5."}:
            suggestions.append(s[2:].strip())
    if not suggestions:
        suggestions = [txt[:200]]

    return {"analysis": txt, "suggestions": suggestions[:5]}
