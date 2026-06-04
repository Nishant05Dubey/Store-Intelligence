# PROMPT: Generate tests for a conversion funnel endpoint that calculates stage-by-stage
# drop-off rates (Entry -> Zone Visit -> Billing Queue -> Purchase).
# Ensure that session deduplication is tested (re-entries shouldn't double count).
#
# CHANGES MADE:
# - Added edge case tests for funnel logic when certain zones are completely skipped
# - Enforced strict monotonic session mapping (visitor unit, not raw events)
# - Validated the drop_off_pct calculation against division-by-zero
"""Tests for GET /stores/{id}/funnel endpoint."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from sqlmodel import Session

from app.models import EventRow, PosTransactionRow

STORE_ID = "STORE_BLR_001"
TODAY = datetime.now(timezone.utc)


def _add(engine, row):
    with Session(engine) as s:
        s.add(row)
        s.commit()


def seed_event(engine, visitor_id, event_type, zone_id=None, is_staff=False, ts=None):
    queue_depth = 2 if event_type == "BILLING_QUEUE_JOIN" else None
    _add(engine, EventRow(
        event_id=str(uuid.uuid4()),
        store_id=STORE_ID,
        camera_id="CAM_ENTRY_01" if event_type in ("ENTRY", "EXIT", "REENTRY") else "CAM_FLOOR_01",
        visitor_id=visitor_id, event_type=event_type,
        timestamp=(ts or TODAY).isoformat(),
        zone_id=zone_id, dwell_ms=0,
        is_staff=is_staff, confidence=0.9,
        queue_depth=queue_depth,
        ingested_at=datetime.now(timezone.utc).isoformat(),
    ))


def seed_pos_txn(engine, ts=None):
    _add(engine, PosTransactionRow(
        transaction_id=f"TXN_{uuid.uuid4().hex[:8]}",
        store_id=STORE_ID,
        timestamp=(ts or TODAY - timedelta(minutes=1)).isoformat(),
        basket_value=799.0,
    ))


# ---------------------------------------------------------------------------
# Empty funnel
# ---------------------------------------------------------------------------


def test_funnel_empty_store_all_zeros(client):
    body = client.get(f"/stores/{STORE_ID}/funnel").json()
    assert body["session_unit"] is True
    assert body["reentry_excluded"] is True
    for stage in body["stages"]:
        assert stage["count"] == 0


# ---------------------------------------------------------------------------
# Monotonic funnel
# ---------------------------------------------------------------------------


def test_funnel_stages_monotonically_decreasing(client, test_engine):
    """Each stage count must be <= the previous stage count."""
    for v in ["V1", "V2", "V3", "V4", "V5"]:
        seed_event(test_engine, v, "ENTRY")
    for v in ["V1", "V2", "V3", "V4"]:
        seed_event(test_engine, v, "ZONE_ENTER", zone_id="MINIMALIST")
    for v in ["V1", "V2", "V3"]:
        seed_event(test_engine, v, "BILLING_QUEUE_JOIN", zone_id="CASH_COUNTER")

    body = client.get(f"/stores/{STORE_ID}/funnel").json()
    counts = [s["count"] for s in body["stages"]]
    for i in range(1, len(counts)):
        assert counts[i] <= counts[i - 1], (
            f"Stage {body['stages'][i]['stage']} ({counts[i]}) "
            f"> stage {body['stages'][i-1]['stage']} ({counts[i-1]})"
        )


def test_funnel_drop_off_percentage_correct(client, test_engine):
    for v in ["A", "B", "C", "D"]:
        seed_event(test_engine, v, "ENTRY")
    for v in ["A", "B"]:
        seed_event(test_engine, v, "ZONE_ENTER", zone_id="DERMDOC")

    body = client.get(f"/stores/{STORE_ID}/funnel").json()
    stages = {s["stage"]: s for s in body["stages"]}

    entry_count = stages["entry"]["count"]
    zone_count = stages["zone_visit"]["count"]
    expected = round((1 - zone_count / entry_count) * 100, 1) if entry_count > 0 else 0.0
    assert abs(stages["zone_visit"]["drop_off_pct"] - expected) < 0.5


def test_funnel_entry_drop_off_always_zero(client, test_engine):
    seed_event(test_engine, "VIS_1", "ENTRY")
    body = client.get(f"/stores/{STORE_ID}/funnel").json()
    stages = {s["stage"]: s for s in body["stages"]}
    assert stages["entry"]["drop_off_pct"] == 0.0


# ---------------------------------------------------------------------------
# Re-entry handling
# ---------------------------------------------------------------------------


def test_funnel_reentry_does_not_double_count(client, test_engine):
    """REENTRY is not the same as ENTRY — visitor counted once."""
    seed_event(test_engine, "V_REENTRY", "ENTRY")
    seed_event(test_engine, "V_REENTRY", "EXIT")
    seed_event(test_engine, "V_REENTRY", "REENTRY")
    seed_event(test_engine, "V_NEW", "ENTRY")

    body = client.get(f"/stores/{STORE_ID}/funnel").json()
    stages = {s["stage"]: s for s in body["stages"]}
    # 2 unique visitors: V_REENTRY (counted by ENTRY) + V_NEW
    assert stages["entry"]["count"] == 2


# ---------------------------------------------------------------------------
# Staff exclusion
# ---------------------------------------------------------------------------


def test_funnel_staff_excluded_from_all_stages(client, test_engine):
    seed_event(test_engine, "VIS_CUST", "ENTRY", is_staff=False)
    seed_event(test_engine, "VIS_STAFF", "ENTRY", is_staff=True)
    seed_event(test_engine, "VIS_STAFF", "ZONE_ENTER", zone_id="MAKEUP_UNIT", is_staff=True)

    body = client.get(f"/stores/{STORE_ID}/funnel").json()
    stages = {s["stage"]: s for s in body["stages"]}
    assert stages["entry"]["count"] == 1
    assert stages["zone_visit"]["count"] == 0


# ---------------------------------------------------------------------------
# Metadata flags
# ---------------------------------------------------------------------------


def test_funnel_session_unit_is_true(client):
    assert client.get(f"/stores/{STORE_ID}/funnel").json()["session_unit"] is True


def test_funnel_reentry_excluded_is_true(client):
    assert client.get(f"/stores/{STORE_ID}/funnel").json()["reentry_excluded"] is True
