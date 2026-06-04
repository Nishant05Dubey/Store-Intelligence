"""
Real-time metric computation for GET /stores/{id}/metrics.

All queries are computed against live data — no caching from previous days.
Staff events (is_staff=True) are excluded from all customer metrics.

Conversion logic:
  A visitor is "converted" if they had a BILLING_QUEUE_JOIN or ZONE_ENTER(CASH_COUNTER)
  in the 5-minute window before any POS transaction timestamp for the same store.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

import structlog
from sqlmodel import Session, select, text

from app.models import (
    EventRow,
    MetricsResponse,
    PosTransactionRow,
    ZoneDwellStats,
)

logger = structlog.get_logger(__name__)

# Conversion correlation window: visitor in billing zone within this window before POS txn
CONVERSION_WINDOW_MINUTES = 5
# Queue spike threshold: queue depth above this triggers WARN
QUEUE_SPIKE_THRESHOLD = 5
# Billing zone IDs
BILLING_ZONE_IDS = {"CASH_COUNTER"}


def compute_metrics(store_id: str, session: Session) -> MetricsResponse:
    """
    Compute real-time store metrics for today.
    Uses the date portion of stored event timestamps for "today" filter.
    Falls back gracefully when there is no data.
    """
    import os
    now = datetime.now(timezone.utc)
    today_prefix = os.getenv("DEMO_DATE", now.strftime("%Y-%m-%d"))

    # ----- Unique customer visitors today -----
    unique_visitors = _count_unique_visitors(store_id, today_prefix, session)

    # ----- Conversion rate -----
    conversion_rate = _compute_conversion_rate(
        store_id, today_prefix, session
    )

    # ----- Average dwell per zone -----
    dwell_stats = _compute_dwell_per_zone(store_id, today_prefix, session)

    # ----- Current queue depth -----
    queue_depth = _current_queue_depth(store_id, session)

    # ----- Billing abandonment rate -----
    abandonment_rate = _compute_abandonment_rate(store_id, today_prefix, session)

    return MetricsResponse(
        store_id=store_id,
        window="today",
        unique_visitors=unique_visitors,
        conversion_rate=round(conversion_rate, 4),
        avg_dwell_per_zone=dwell_stats,
        queue_depth_current=queue_depth,
        abandonment_rate=round(abandonment_rate, 4),
        computed_at=now,
    )


def _count_unique_visitors(
    store_id: str, today_prefix: str, session: Session
) -> int:
    """Count distinct visitor_ids with ENTRY events today, excluding staff."""
    result = session.exec(
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
    return int(result or 0)


def _compute_conversion_rate(
    store_id: str, today_prefix: str, session: Session
) -> float:
    """
    Conversion rate = converted sessions / total unique visitor sessions.

    A session is "converted" if the visitor was detected in the billing zone
    within CONVERSION_WINDOW_MINUTES before any POS transaction timestamp.
    """
    # Get all unique customer sessions today
    total_sessions_result = session.exec(
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
    total_sessions = int(total_sessions_result or 0)

    if total_sessions == 0:
        return 0.0

    # Get all POS transactions for today
    txn_timestamps = session.exec(
        text(
            """
            SELECT timestamp FROM pos_transactions
            WHERE store_id = :store_id
              AND timestamp LIKE :today_prefix
            """
        ).bindparams(store_id=store_id, today_prefix=f"{today_prefix}%")
    ).all()

    if not txn_timestamps:
        return 0.0

    # For each POS transaction, find visitors in billing zone within window
    converted_visitors: set[str] = set()
    window_delta = timedelta(minutes=CONVERSION_WINDOW_MINUTES)

    for (ts_str,) in txn_timestamps:
        try:
            ts_str_clean = ts_str.replace("Z", "+00:00")
            txn_time = datetime.fromisoformat(ts_str_clean)
        except ValueError:
            continue

        window_start = (txn_time - window_delta).isoformat()
        window_end = txn_time.isoformat()

        visitors = session.exec(
            text(
                """
                SELECT DISTINCT visitor_id
                FROM events
                WHERE store_id = :store_id
                  AND is_staff = 0
                  AND (
                        zone_id IN ('CASH_COUNTER')
                     OR event_type IN ('BILLING_QUEUE_JOIN', 'BILLING_QUEUE_ABANDON')
                  )
                  AND timestamp >= :window_start
                  AND timestamp <= :window_end
                """
            ).bindparams(
                store_id=store_id,
                window_start=window_start,
                window_end=window_end,
            )
        ).all()

        for (vid,) in visitors:
            converted_visitors.add(vid)

    converted = len(converted_visitors)
    # Cap at total_sessions (cannot convert more than visited)
    converted = min(converted, total_sessions)
    return converted / total_sessions


def _compute_dwell_per_zone(
    store_id: str, today_prefix: str, session: Session
) -> list[ZoneDwellStats]:
    """Compute average dwell time per zone from ZONE_DWELL events."""
    rows = session.exec(
        text(
            """
            SELECT zone_id,
                   AVG(dwell_ms) as avg_dwell,
                   COUNT(*) as visit_count
            FROM events
            WHERE store_id = :store_id
              AND event_type = 'ZONE_DWELL'
              AND is_staff = 0
              AND zone_id IS NOT NULL
              AND timestamp LIKE :today_prefix
            GROUP BY zone_id
            ORDER BY avg_dwell DESC
            """
        ).bindparams(store_id=store_id, today_prefix=f"{today_prefix}%")
    ).all()

    return [
        ZoneDwellStats(
            zone_id=row[0],
            avg_dwell_ms=round(float(row[1] or 0), 1),
            visit_count=int(row[2] or 0),
        )
        for row in rows
        if row[0] is not None
    ]


def _current_queue_depth(store_id: str, session: Session) -> int:
    """
    Current queue depth = most recent queue_depth value from
    BILLING_QUEUE_JOIN events in the last 30 minutes.
    """
    cutoff = (
        datetime.now(timezone.utc) - timedelta(minutes=30)
    ).isoformat()

    result = session.exec(
        text(
            """
            SELECT queue_depth
            FROM events
            WHERE store_id = :store_id
              AND event_type = 'BILLING_QUEUE_JOIN'
              AND queue_depth IS NOT NULL
              AND timestamp >= :cutoff
            ORDER BY timestamp DESC
            LIMIT 1
            """
        ).bindparams(store_id=store_id, cutoff=cutoff)
    ).scalar()

    return int(result or 0)


def _compute_abandonment_rate(
    store_id: str, today_prefix: str, session: Session
) -> float:
    """
    Abandonment rate = BILLING_QUEUE_ABANDON events /
                       (BILLING_QUEUE_JOIN events)
    Both filtered for customer-only (is_staff=False) events.
    """
    joins = session.exec(
        text(
            """
            SELECT COUNT(DISTINCT visitor_id)
            FROM events
            WHERE store_id = :store_id
              AND event_type = 'BILLING_QUEUE_JOIN'
              AND is_staff = 0
              AND timestamp LIKE :today_prefix
            """
        ).bindparams(store_id=store_id, today_prefix=f"{today_prefix}%")
    ).scalar()

    abandons = session.exec(
        text(
            """
            SELECT COUNT(DISTINCT visitor_id)
            FROM events
            WHERE store_id = :store_id
              AND event_type = 'BILLING_QUEUE_ABANDON'
              AND is_staff = 0
              AND timestamp LIKE :today_prefix
            """
        ).bindparams(store_id=store_id, today_prefix=f"{today_prefix}%")
    ).scalar()

    joins_count = int(joins or 0)
    abandons_count = int(abandons or 0)

    if joins_count == 0:
        return 0.0

    return min(abandons_count / joins_count, 1.0)
