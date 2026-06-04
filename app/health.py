"""
Health endpoint for GET /health.
Reports service status, last event per camera, and STALE_FEED warnings.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import structlog
from sqlmodel import Session, text

from app.models import CameraHealth, HealthResponse

logger = structlog.get_logger(__name__)

STORE_LAYOUT_PATH = os.getenv("STORE_LAYOUT_PATH", "/data/store_layout.json")
STALE_FEED_MINUTES = 10


def compute_health(session: Session) -> HealthResponse:
    """
    Check system health:
    1. Database connectivity
    2. Per-camera last event timestamp
    3. Flag STALE_FEED if any camera has no events in last 10 minutes
    """
    now = datetime.now(timezone.utc)
    db_ok = True

    try:
        # Simple connectivity test
        session.exec(text("SELECT 1")).scalar()
    except Exception as exc:
        logger.error("health.db_error", error=str(exc))
        db_ok = False

    cameras = _check_cameras(now, session)

    stale_feeds = [c for c in cameras if c.is_stale]
    status = "ok"
    if not db_ok:
        status = "down"
    elif stale_feeds:
        status = "degraded"

    return HealthResponse(
        status=status,
        db_ok=db_ok,
        cameras=cameras,
        computed_at=now,
    )


def _check_cameras(now: datetime, session: Session) -> list[CameraHealth]:
    """Check each known camera's last event timestamp."""
    camera_ids = _load_camera_ids()

    # Get last event per camera
    stale_cutoff = (now - timedelta(minutes=STALE_FEED_MINUTES)).isoformat()

    results = session.exec(
        text(
            """
            SELECT camera_id, MAX(timestamp) as last_event
            FROM events
            GROUP BY camera_id
            """
        )
    ).all()
    last_event_map = {row[0]: row[1] for row in results}

    camera_health: list[CameraHealth] = []

    for cam_id in camera_ids:
        last_ts_str = last_event_map.get(cam_id)
        if last_ts_str:
            try:
                ts_clean = last_ts_str.replace("Z", "+00:00")
                last_event_dt = datetime.fromisoformat(ts_clean)
                lag_minutes = (now - last_event_dt).total_seconds() / 60
                is_stale = lag_minutes > STALE_FEED_MINUTES
            except ValueError:
                last_event_dt = None
                lag_minutes = None
                is_stale = True
        else:
            last_event_dt = None
            lag_minutes = None
            is_stale = False  # Camera has never reported = not stale yet

        camera_health.append(
            CameraHealth(
                camera_id=cam_id,
                last_event_at=last_event_dt,
                is_stale=is_stale,
                lag_minutes=round(lag_minutes, 1) if lag_minutes is not None else None,
            )
        )

    return camera_health


def _load_camera_ids() -> list[str]:
    """Load camera IDs from store layout."""
    layout_path = Path(STORE_LAYOUT_PATH)
    if not layout_path.exists():
        return []
    try:
        with open(layout_path) as f:
            layout = json.load(f)
        return [cam["camera_id"] for cam in layout.get("cameras", [])]
    except Exception:
        return []
