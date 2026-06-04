"""
Conversion funnel and session logic.
GET /stores/{id}/funnel

The funnel tracks visitor journeys as sessions (not raw event counts).
Sessions are the unit — re-entries do NOT double-count a visitor.

Funnel stages:
  Entry → Zone Visit → Billing Queue → Purchase
"""

from __future__ import annotations

from datetime import datetime, timezone

import structlog
from sqlmodel import Session, text

from app.models import FunnelResponse, FunnelStage

logger = structlog.get_logger(__name__)


def compute_funnel(store_id: str, session: Session) -> FunnelResponse:
    """
    Compute session-based conversion funnel for today.

    Each stage counts unique visitor_ids (not event counts), ensuring
    re-entries do not double-count a visitor at any funnel stage.
    """
    import os
    now = datetime.now(timezone.utc)
    today_prefix = os.getenv("DEMO_DATE", now.strftime("%Y-%m-%d"))

    # Stage 1: Unique customer visitors (ENTRY events today, no staff)
    entry_count = _count_unique_at_event(
        store_id, today_prefix, "ENTRY", session
    )

    # Stage 2: Unique visitors who visited at least one zone
    zone_visit_count = session.exec(
        text(
            """
            SELECT COUNT(DISTINCT visitor_id)
            FROM events
            WHERE store_id = :store_id
              AND event_type IN ('ZONE_ENTER', 'ZONE_DWELL')
              AND is_staff = 0
              AND timestamp LIKE :today_prefix
            """
        ).bindparams(store_id=store_id, today_prefix=f"{today_prefix}%")
    ).scalar()
    zone_visit_count = int(zone_visit_count or 0)

    # Stage 3: Unique visitors who reached billing queue
    billing_count = session.exec(
        text(
            """
            SELECT COUNT(DISTINCT visitor_id)
            FROM events
            WHERE store_id = :store_id
              AND (
                event_type = 'BILLING_QUEUE_JOIN'
                OR (event_type = 'ZONE_ENTER' AND zone_id = 'CASH_COUNTER')
              )
              AND is_staff = 0
              AND timestamp LIKE :today_prefix
            """
        ).bindparams(store_id=store_id, today_prefix=f"{today_prefix}%")
    ).scalar()
    billing_count = int(billing_count or 0)

    # Stage 4: Unique visitors who converted (had a POS transaction)
    # Uses the same correlation logic as metrics.py
    purchase_count = _count_converted_sessions(
        store_id, today_prefix, session
    )

    # Build funnel stages with drop-off percentages
    stages = _build_stages(
        [
            ("entry", entry_count),
            ("zone_visit", zone_visit_count),
            ("billing_queue", billing_count),
            ("purchase", purchase_count),
        ]
    )

    return FunnelResponse(
        store_id=store_id,
        stages=stages,
        session_unit=True,
        reentry_excluded=True,
    )


def _count_unique_at_event(
    store_id: str, today_prefix: str, event_type: str, session: Session
) -> int:
    result = session.exec(
        text(
            """
            SELECT COUNT(DISTINCT visitor_id)
            FROM events
            WHERE store_id = :store_id
              AND event_type = :event_type
              AND is_staff = 0
              AND timestamp LIKE :today_prefix
            """
        ).bindparams(
            store_id=store_id,
            event_type=event_type,
            today_prefix=f"{today_prefix}%",
        )
    ).scalar()
    return int(result or 0)


def _count_converted_sessions(
    store_id: str, today_prefix: str, session: Session
) -> int:
    """
    Count unique visitors who converted: they were in the billing zone
    within 5 minutes before any POS transaction today.
    """
    from datetime import timedelta

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
        return 0

    converted: set[str] = set()
    window = timedelta(minutes=5)

    for (ts_str,) in txn_timestamps:
        try:
            ts_clean = ts_str.replace("Z", "+00:00")
            txn_time = datetime.fromisoformat(ts_clean)
        except ValueError:
            continue

        window_start = (txn_time - window).isoformat()
        window_end = txn_time.isoformat()

        rows = session.exec(
            text(
                """
                SELECT DISTINCT visitor_id
                FROM events
                WHERE store_id = :store_id
                  AND is_staff = 0
                  AND (
                    event_type IN ('BILLING_QUEUE_JOIN', 'BILLING_QUEUE_ABANDON')
                    OR (event_type = 'ZONE_ENTER' AND zone_id = 'CASH_COUNTER')
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

        for (vid,) in rows:
            converted.add(vid)

    return len(converted)


def _build_stages(
    stage_data: list[tuple[str, int]],
) -> list[FunnelStage]:
    """Convert raw stage counts to FunnelStage objects with drop-off percentages."""
    stages: list[FunnelStage] = []
    for i, (name, count) in enumerate(stage_data):
        if i == 0:
            drop_off_pct = 0.0
        else:
            prev_count = stage_data[i - 1][1]
            if prev_count == 0:
                drop_off_pct = 0.0
            else:
                drop_off_pct = round(
                    (1.0 - count / prev_count) * 100, 1
                )
        stages.append(
            FunnelStage(stage=name, count=count, drop_off_pct=drop_off_pct)
        )
    return stages
