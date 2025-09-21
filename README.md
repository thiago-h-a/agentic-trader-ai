# LLM Stock Trader API

A compact FastAPI backend that streams an LLM-driven backtest and exposes a few helper endpoints.

## Quickstart

```bash
# create env
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -U pip
pip install fastapi uvicorn "pandas<3.0" numpy yfinance langchain-openai langchain-google-genai

# run
uvicorn app.main:app --reload --port 8000
```

## API

- `GET /` - health message
- `GET /health` - basic health
- `GET /docs` - Swagger UI
- `GET /api/v1/agent-stream/llm-backtest-stream` - SSE stream of a backtest
- `GET /api/v1/agent-stream/llm-backtest-stream/status` - status
- `GET /api/v1/backtest/available-dates`
- `GET /api/v1/backtest/available-dates/{run_id}`
- `GET /api/v1/backtest/analysis/day/{run_id}?date=YYYY-MM-DD`
- `GET /api/v1/backtest/session-dates/{session_id}`
- `POST /api/v1/daily/daily-feedback`

> The backend still expects SQLite logs at `backend/data/backtest_logs.db` when those routes are used.
