"""
Store Intelligence API — FastAPI entrypoint.

Endpoints:
  POST /events/ingest           — Batch event ingestion
  GET  /stores/{id}/metrics     — Real-time store metrics
  GET  /stores/{id}/funnel      — Conversion funnel
  GET  /stores/{id}/heatmap     — Zone heatmap
  GET  /stores/{id}/anomalies   — Active anomalies
  GET  /health                  — Service health

All requests are logged with trace_id, store_id, endpoint, latency_ms.
"""

from __future__ import annotations

import time
import sys
import uuid
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlmodel import Session

from app import anomalies as anomaly_mod
from app import funnel as funnel_mod
from app import health as health_mod
from app import heatmap as heatmap_mod
from app import metrics as metrics_mod
from app.database import create_tables, get_engine, get_session, load_pos_transactions
from app.ingestion import ingest_events
from app.models import IngestRequest, IngestResponse
from app.logging_config import configure_logging

configure_logging()
logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Lifespan: startup / shutdown hooks
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("startup.begin")
    create_tables()
    load_pos_transactions()
    logger.info("startup.complete")
    yield
    logger.info("shutdown.complete")


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Store Intelligence API",
    description=(
        "Retail analytics API that ingests CCTV detection events "
        "and exposes real-time store metrics for Apex Retail."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request logging middleware
# ---------------------------------------------------------------------------


@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    trace_id = str(uuid.uuid4())
    request.state.trace_id = trace_id
    start = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.error(
            "request.error",
            trace_id=trace_id,
            endpoint=str(request.url.path),
            method=request.method,
            latency_ms=elapsed_ms,
            error=str(exc),
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "trace_id": trace_id,
            },
        )

    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    store_id = request.path_params.get("store_id", "-")
    event_count = getattr(request.state, "event_count", None)
    
    log_kwargs = {
        "trace_id": trace_id,
        "store_id": store_id,
        "endpoint": str(request.url.path),
        "method": request.method,
        "latency_ms": elapsed_ms,
        "status_code": response.status_code,
    }
    if event_count is not None:
        log_kwargs["event_count"] = event_count
        
    logger.info("request.complete", **log_kwargs)
    response.headers["X-Trace-Id"] = trace_id
    return response


# ---------------------------------------------------------------------------
# Helper: safe DB session wrapper
# ---------------------------------------------------------------------------


def _db_session() -> Session:
    """Get a DB session; raises 503 if DB is unavailable."""
    try:
        engine = get_engine()
        return Session(engine)
    except Exception as exc:
        logger.error("db.unavailable", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": "Database unavailable", "detail": str(exc)},
        )


# ---------------------------------------------------------------------------
# POST /events/ingest
# ---------------------------------------------------------------------------


@app.post(
    "/events/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_200_OK,
    summary="Ingest a batch of detection events",
    description=(
        "Accepts batches of up to 500 events. "
        "Validates, deduplicates by event_id, and stores. "
        "Idempotent — safe to call multiple times with same payload."
    ),
)
async def ingest(req: IngestRequest, request: Request) -> IngestResponse:
    request.state.event_count = len(req.events)
    trace_id = getattr(request.state, "trace_id", str(uuid.uuid4()))
    with _db_session() as session:
        return ingest_events(req, session, trace_id)


# ---------------------------------------------------------------------------
# GET /stores/{store_id}/metrics
# ---------------------------------------------------------------------------


@app.get(
    "/stores/{store_id}/metrics",
    summary="Real-time store metrics",
    description=(
        "Returns today's unique visitors, conversion rate, "
        "average dwell per zone, current queue depth, and abandonment rate. "
        "Staff events are excluded. Never cached from yesterday."
    ),
)
async def get_metrics(store_id: str):
    with _db_session() as session:
        return metrics_mod.compute_metrics(store_id, session)


# ---------------------------------------------------------------------------
# GET /stores/{store_id}/funnel
# ---------------------------------------------------------------------------


@app.get(
    "/stores/{store_id}/funnel",
    summary="Conversion funnel",
    description=(
        "Session-based funnel: Entry → Zone Visit → Billing Queue → Purchase. "
        "Re-entries do not double-count visitors."
    ),
)
async def get_funnel(store_id: str):
    with _db_session() as session:
        return funnel_mod.compute_funnel(store_id, session)


# ---------------------------------------------------------------------------
# GET /stores/{store_id}/heatmap
# ---------------------------------------------------------------------------


@app.get(
    "/stores/{store_id}/heatmap",
    summary="Zone heatmap",
    description=(
        "Zone visit frequency and average dwell, normalised 0-100. "
        "Includes data_confidence flag when fewer than 20 sessions."
    ),
)
async def get_heatmap(store_id: str):
    with _db_session() as session:
        return heatmap_mod.compute_heatmap(store_id, session)


# ---------------------------------------------------------------------------
# GET /stores/{store_id}/anomalies
# ---------------------------------------------------------------------------


@app.get(
    "/stores/{store_id}/anomalies",
    summary="Active anomalies",
    description=(
        "Detects: BILLING_QUEUE_SPIKE, CONVERSION_DROP vs 7-day avg, "
        "DEAD_ZONE (no visits in 30 min). Severity: INFO / WARN / CRITICAL."
    ),
)
async def get_anomalies(store_id: str):
    with _db_session() as session:
        return anomaly_mod.compute_anomalies(store_id, session)


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


@app.get(
    "/health",
    summary="Service health",
    description=(
        "Reports DB status, last event per camera, and STALE_FEED warnings "
        "for any camera silent for > 10 minutes."
    ),
)
async def get_health():
    with _db_session() as session:
        return health_mod.compute_health(session)


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
async def root():
    return {
        "service": "Store Intelligence API",
        "version": "1.0.0",
        "docs": "/docs",
    }


# ---------------------------------------------------------------------------
# POST /process_video
# ---------------------------------------------------------------------------

class ProcessVideoRequest(BaseModel):
    video_path: str
    camera_id: str
    store_id: str = "STORE_BLR_001"
    clip_start: str = "2026-04-10T11:00:00Z"

@app.post("/process_video", summary="Process a new video dynamically")
async def process_video(req: ProcessVideoRequest):
    import subprocess
    import os
    
    # Strip any accidental quotes from user input
    video_path = req.video_path.strip('\"\'')
    
    # Launch the detect.py pipeline in the background
    cmd = [
        sys.executable,
        "pipeline/detect.py",
        "--video", video_path,
        "--camera-id", req.camera_id,
        "--store-id", req.store_id,
        "--clip-start", req.clip_start,
        "--layout", "data/store_layout.json",
        "--api-url", f"http://127.0.0.1:{os.getenv('UVICORN_PORT', '8000')}"
    ]
    
    try:
        subprocess.Popen(cmd, cwd=os.getcwd())
        return {"status": "started", "camera_id": req.camera_id, "message": f"Processing started for {req.camera_id}"}
    except Exception as e:
        logger.error("process_video.failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

