"""
Anomaly detection for GET /stores/{id}/anomalies.

Detects three classes of operational anomalies:
1. BILLING_QUEUE_SPIKE  — queue depth > threshold (real-time)
2. CONVERSION_DROP      — today's conversion rate vs 7-day rolling avg
3. DEAD_ZONE            — no zone visits in the last 30 minutes
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import os

import structlog
from sqlmodel import Session, text

from app.models import AnomaliesResponse, AnomalyItem, AnomalySeverity

logger = structlog.get_logger(__name__)

# Thresholds
QUEUE_SPIKE_WARN = 4       # queue depth >= this → WARN
QUEUE_SPIKE_CRITICAL = 8   # queue depth >= this → CRITICAL
CONVERSION_DROP_WARN = 0.3  # today's rate < (7-day avg * (1 - this)) → WARN
CONVERSION_DROP_CRITICAL = 0.5  # today's rate < (7-day avg * (1 - this)) → CRITICAL
DEAD_ZONE_MINUTES = 30     # no visits in this window → dead zone anomaly
STALE_FEED_MINUTES = 10    # no events from camera in this window → STALE_FEED


def compute_anomalies(store_id: str, session: Session) -> AnomaliesResponse:
    now = datetime.now(timezone.utc)
    demo_date_str = os.getenv("DEMO_DATE")
    if demo_date_str:
        now = datetime.fromisoformat(f"{demo_date_str}T23:59:59+00:00")
    anomalies: list[AnomalyItem] = []

    # 1. Billing queue spike
    queue_anomaly = _check_queue_spike(store_id, now, session)
    if queue_anomaly:
        anomalies.append(queue_anomaly)

    # 2. Conversion drop vs 7-day average
    conv_anomaly = _check_conversion_drop(store_id, now, session)
    if conv_anomaly:
        anomalies.append(conv_anomaly)

    # 3. Dead zones (no visits in last 30 min)
    dead_zone_anomalies = _check_dead_zones(store_id, now, session)
    anomalies.extend(dead_zone_anomalies)

    return AnomaliesResponse(
        store_id=store_id,
        anomalies=anomalies,
        computed_at=now,
    )


def _check_queue_spike(
    store_id: str, now: datetime, session: Session
) -> AnomalyItem | None:
    """Check if current billing queue depth exceeds threshold."""
    cutoff = (now - timedelta(minutes=15)).isoformat()

    result = session.exec(
        text(
            """
            SELECT MAX(queue_depth)
            FROM events
            WHERE store_id = :store_id
              AND event_type = 'BILLING_QUEUE_JOIN'
              AND queue_depth IS NOT NULL
              AND timestamp >= :cutoff
            """
        ).bindparams(store_id=store_id, cutoff=cutoff)
    ).scalar()

    depth = int(result or 0)
    if depth < QUEUE_SPIKE_WARN:
        return None

    severity = (
        AnomalySeverity.CRITICAL
        if depth >= QUEUE_SPIKE_CRITICAL
        else AnomalySeverity.WARN
    )
    return AnomalyItem(
        anomaly_type="BILLING_QUEUE_SPIKE",
        severity=severity,
        detail=f"Billing queue depth {depth} detected in last 15 minutes",
        detected_at=now,
        suggested_action=(
            "Open an additional billing counter immediately"
            if severity == AnomalySeverity.CRITICAL
            else "Monitor billing queue — consider opening another counter"
        ),
    )


def _check_conversion_drop(
    store_id: str, now: datetime, session: Session
) -> AnomalyItem | None:
    """
    Compare today's conversion rate against 7-day rolling average.
    Only fires when there are sufficient sessions (>=5) to be meaningful.
    """
    today_prefix = os.getenv("DEMO_DATE", now.strftime("%Y-%m-%d"))

    # Today's unique visitors
    today_visitors = session.exec(
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
    today_visitors = int(today_visitors or 0)

    # Not enough data to fire anomaly
    if today_visitors < 5:
        return None

    # Today's POS transactions
    today_txns = session.exec(
        text(
            """
            SELECT COUNT(*) FROM pos_transactions
            WHERE store_id = :store_id
              AND timestamp LIKE :today_prefix
            """
        ).bindparams(store_id=store_id, today_prefix=f"{today_prefix}%")
    ).scalar()
    today_txns = int(today_txns or 0)

    today_rate = today_txns / today_visitors if today_visitors > 0 else 0.0

    # 7-day average: look back 7 days excluding today
    seven_days_ago = (now - timedelta(days=7)).strftime("%Y-%m-%d")

    past_visitors = session.exec(
        text(
            """
            SELECT COUNT(DISTINCT visitor_id)
            FROM events
            WHERE store_id = :store_id
              AND event_type = 'ENTRY'
              AND is_staff = 0
              AND timestamp >= :seven_days_ago
              AND timestamp < :today_prefix
            """
        ).bindparams(
            store_id=store_id,
            seven_days_ago=f"{seven_days_ago}T00:00:00",
            today_prefix=f"{today_prefix}T00:00:00",
        )
    ).scalar()
    past_visitors = int(past_visitors or 0)

    past_txns = session.exec(
        text(
            """
            SELECT COUNT(*) FROM pos_transactions
            WHERE store_id = :store_id
              AND timestamp >= :seven_days_ago
              AND timestamp < :today_prefix
            """
        ).bindparams(
            store_id=store_id,
            seven_days_ago=f"{seven_days_ago}T00:00:00",
            today_prefix=f"{today_prefix}T00:00:00",
        )
    ).scalar()
    past_txns = int(past_txns or 0)

    if past_visitors < 5:
        # Not enough historical data
        return None

    historical_rate = past_txns / past_visitors

    # Compute drop ratio
    if historical_rate == 0:
        return None

    drop_ratio = (historical_rate - today_rate) / historical_rate

    if drop_ratio < CONVERSION_DROP_WARN:
        return None

    severity = (
        AnomalySeverity.CRITICAL
        if drop_ratio >= CONVERSION_DROP_CRITICAL
        else AnomalySeverity.WARN
    )

    return AnomalyItem(
        anomaly_type="CONVERSION_DROP",
        severity=severity,
        detail=(
            f"Today's conversion rate {today_rate:.1%} is "
            f"{drop_ratio:.0%} below 7-day average {historical_rate:.1%}"
        ),
        detected_at=now,
        suggested_action=(
            "Immediate review of floor staff placement, promotions, and zone engagement"
            if severity == AnomalySeverity.CRITICAL
            else "Review promotions and floor staff engagement strategy"
        ),
    )


def _check_dead_zones(
    store_id: str, now: datetime, session: Session
) -> list[AnomalyItem]:
    """
    Detect zones with no visitor activity in the last DEAD_ZONE_MINUTES.
    Only flags zones that had activity earlier today (to avoid false alarms
    for zones not covered by all cameras).
    """
    cutoff_recent = (now - timedelta(minutes=DEAD_ZONE_MINUTES)).isoformat()
    today_prefix = now.strftime("%Y-%m-%d")

    # Zones active today
    active_today = session.exec(
        text(
            """
            SELECT DISTINCT zone_id
            FROM events
            WHERE store_id = :store_id
              AND zone_id IS NOT NULL
              AND is_staff = 0
              AND timestamp LIKE :today_prefix
            """
        ).bindparams(store_id=store_id, today_prefix=f"{today_prefix}%")
    ).all()
    active_zone_ids = {row[0] for row in active_today}

    # Zones active in recent window
    active_recent = session.exec(
        text(
            """
            SELECT DISTINCT zone_id
            FROM events
            WHERE store_id = :store_id
              AND zone_id IS NOT NULL
              AND is_staff = 0
              AND timestamp >= :cutoff
            """
        ).bindparams(store_id=store_id, cutoff=cutoff_recent)
    ).all()
    recent_zone_ids = {row[0] for row in active_recent}

    dead_zones = active_zone_ids - recent_zone_ids
    anomalies: list[AnomalyItem] = []

    for zone_id in sorted(dead_zones):
        anomalies.append(
            AnomalyItem(
                anomaly_type="DEAD_ZONE",
                severity=AnomalySeverity.INFO,
                detail=(
                    f"Zone '{zone_id}' has had no customer visits in "
                    f"the last {DEAD_ZONE_MINUTES} minutes"
                ),
                detected_at=now,
                suggested_action=(
                    f"Consider redirecting floor staff to zone '{zone_id}' "
                    "to engage customers or review display arrangement"
                ),
            )
        )

    return anomalies
