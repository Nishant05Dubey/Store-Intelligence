"""
Heatmap computation for GET /stores/{id}/heatmap.
Returns zone visit frequency + average dwell, normalised 0-100.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import structlog
from sqlmodel import Session, text

from app.models import HeatmapResponse, HeatmapZone

logger = structlog.get_logger(__name__)

STORE_LAYOUT_PATH = os.getenv("STORE_LAYOUT_PATH", "/data/store_layout.json")
MIN_SESSIONS_FOR_CONFIDENCE = 20


def compute_heatmap(store_id: str, session: Session) -> HeatmapResponse:
    import os
    now = datetime.now(timezone.utc)
    today_prefix = os.getenv("DEMO_DATE", now.strftime("%Y-%m-%d"))

    # Load zone names from layout
    zone_names = _load_zone_names()

    # Query zone visit frequency and avg dwell
    rows = session.exec(
        text(
            """
            SELECT zone_id,
                   COUNT(DISTINCT visitor_id) as visit_freq,
                   AVG(dwell_ms) as avg_dwell
            FROM events
            WHERE store_id = :store_id
              AND event_type IN ('ZONE_ENTER', 'ZONE_DWELL')
              AND is_staff = 0
              AND zone_id IS NOT NULL
              AND timestamp LIKE :today_prefix
            GROUP BY zone_id
            """
        ).bindparams(store_id=store_id, today_prefix=f"{today_prefix}%")
    ).all()

    # Check data confidence
    total_sessions = session.exec(
        text(
            """
            SELECT COUNT(DISTINCT visitor_id)
            FROM events
            WHERE store_id = :store_id
              AND event_type = 'ENTRY'
              AND is_staff = 0
              AND timestamp LIKE :today_prefix
            """
        ).bindparams(store_id=store_id, today_prefix=f"{today_prefix}%")
    ).scalar()
    total_sessions = int(total_sessions or 0)
    data_confidence = total_sessions >= MIN_SESSIONS_FOR_CONFIDENCE

    if not rows:
        return HeatmapResponse(
            store_id=store_id,
            zones=[],
            data_confidence=data_confidence,
            computed_at=now,
        )

    # Normalise visit frequency to 0-100
    max_freq = max(int(r[1] or 0) for r in rows) or 1

    zones = [
        HeatmapZone(
            zone_id=row[0],
            zone_name=zone_names.get(row[0], row[0]),
            visit_frequency=int(row[1] or 0),
            avg_dwell_ms=round(float(row[2] or 0), 1),
            heat_score=round((int(row[1] or 0) / max_freq) * 100, 1),
        )
        for row in rows
    ]

    # Sort by heat score descending
    zones.sort(key=lambda z: z.heat_score, reverse=True)

    return HeatmapResponse(
        store_id=store_id,
        zones=zones,
        data_confidence=data_confidence,
        computed_at=now,
    )


def _load_zone_names() -> dict[str, str]:
    """Load zone_id → zone_name mapping from store_layout.json."""
    layout_path = Path(STORE_LAYOUT_PATH)
    if not layout_path.exists():
        return {}
    try:
        with open(layout_path) as f:
            layout = json.load(f)
        return {
            zone["zone_id"]: zone["zone_name"]
            for zone in layout.get("zones", [])
        }
    except Exception:
        return {}
