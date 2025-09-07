from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Routers live under app.api.routes (renamed from app.endpoints)
from app.api.routes import backtest_router, llm_stream_router, daily_router

# NOTE: The public API (paths and response shapes) is preserved verbatim so the UI does not break.
app = FastAPI(
    title="LLM Stock Backtesting Dashboard API",
    version="1.0.0",
    openapi_url="/api/v1/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS kept identical to original to avoid any front-end issues
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Same prefixes as before
app.include_router(llm_stream_router, prefix="/api/v1/agent-stream", tags=["agent-stream"])
app.include_router(backtest_router, prefix="/api/v1/backtest", tags=["backtest-analysis"])
app.include_router(daily_router, prefix="/api/v1/daily", tags=["daily-feedback"])

@app.get("/")
async def root():
    return {"message": "LLM Stock Backtesting Dashboard API", "version": "1.0.0"}

@app.get("/health")
async def health():
    return {"status": "healthy"}
