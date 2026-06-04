# PROMPT: Generate tests for an API metrics endpoint that returns live store statistics.
# Tests should cover:
# 1. Unique visitors computation
# 2. Conversion rate (matching POS transactions within a 5-min window)
# 3. Staff exclusion logic
# 4. Zero purchase scenarios
#
# CHANGES MADE:
# - Created a comprehensive database fixture with mock events and POS data
# - Adjusted conversion rate test to use precise 5-min threshold
# - Ensured "is_staff=True" events are fully excluded from "unique visitors" count
"""Tests for GET /stores/{id}/metrics endpoint."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlmodel import Session

from app.models import EventRow, PosTransactionRow

STORE_ID = "STORE_BLR_001"
TODAY = datetime.now(timezone.utc)
TODAY_STR = TODAY.strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Helpers — seed directly into the test_engine
# ---------------------------------------------------------------------------


def _add(engine, row):
    with Session(engine) as s:
        s.add(row)
        s.commit()


def seed_entry(engine, visitor_id, is_staff=False, ts=None):
    _add(engine, EventRow(
        event_id=str(uuid.uuid4()),
        store_id=STORE_ID, camera_id="CAM_ENTRY_01",
        visitor_id=visitor_id, event_type="ENTRY",
        timestamp=(ts or TODAY).isoformat(),
        zone_id=None, dwell_ms=0,
        is_staff=is_staff, confidence=0.9,
        ingested_at=datetime.now(timezone.utc).isoformat(),
    ))


def seed_zone_dwell(engine, visitor_id, zone_id, dwell_ms, ts=None):
    _add(engine, EventRow(
        event_id=str(uuid.uuid4()),
        store_id=STORE_ID, camera_id="CAM_FLOOR_01",
        visitor_id=visitor_id, event_type="ZONE_DWELL",
        timestamp=(ts or TODAY).isoformat(),
        zone_id=zone_id, dwell_ms=dwell_ms,
        is_staff=False, confidence=0.85,
        ingested_at=datetime.now(timezone.utc).isoformat(),
    ))


def seed_billing_join(engine, visitor_id, queue_depth, ts=None):
    _add(engine, EventRow(
        event_id=str(uuid.uuid4()),
        store_id=STORE_ID, camera_id="CAM_BILLING_01",
        visitor_id=visitor_id, event_type="BILLING_QUEUE_JOIN",
        timestamp=(ts or TODAY).isoformat(),
        zone_id="CASH_COUNTER", dwell_ms=0,
        is_staff=False, confidence=0.88,
        queue_depth=queue_depth,
        ingested_at=datetime.now(timezone.utc).isoformat(),
    ))


def seed_billing_abandon(engine, visitor_id, ts=None):
    _add(engine, EventRow(
        event_id=str(uuid.uuid4()),
        store_id=STORE_ID, camera_id="CAM_BILLING_01",
        visitor_id=visitor_id, event_type="BILLING_QUEUE_ABANDON",
        timestamp=(ts or TODAY).isoformat(),
        zone_id="CASH_COUNTER", dwell_ms=15000,
        is_staff=False, confidence=0.8,
        ingested_at=datetime.now(timezone.utc).isoformat(),
    ))


def seed_pos_txn(engine, ts=None):
    _add(engine, PosTransactionRow(
        transaction_id=f"TXN_{uuid.uuid4().hex[:8]}",
        store_id=STORE_ID,
        timestamp=(ts or TODAY).isoformat(),
        basket_value=500.0,
    ))


# ---------------------------------------------------------------------------
# Test: Empty store
# ---------------------------------------------------------------------------


def test_metrics_empty_store_returns_zero_not_error(client):
    response = client.get(f"/stores/{STORE_ID}/metrics")
    assert response.status_code == 200
    body = response.json()
    assert body["store_id"] == STORE_ID
    assert body["unique_visitors"] == 0
    assert body["conversion_rate"] == 0.0
    assert body["queue_depth_current"] == 0
    assert body["abandonment_rate"] == 0.0
    assert isinstance(body["avg_dwell_per_zone"], list)


def test_metrics_zero_purchases_conversion_rate_is_zero(client, test_engine):
    seed_entry(test_engine, "VIS_A")
    seed_entry(test_engine, "VIS_B")

    body = client.get(f"/stores/{STORE_ID}/metrics").json()
    assert body["unique_visitors"] == 2
    assert body["conversion_rate"] == 0.0


# ---------------------------------------------------------------------------
# Test: Staff exclusion
# ---------------------------------------------------------------------------


def test_metrics_staff_excluded_from_unique_visitors(client, test_engine):
    seed_entry(test_engine, "VIS_C1")
    seed_entry(test_engine, "VIS_C2")
    seed_entry(test_engine, "VIS_C3")
    seed_entry(test_engine, "VIS_STAFF1", is_staff=True)
    seed_entry(test_engine, "VIS_STAFF2", is_staff=True)

    body = client.get(f"/stores/{STORE_ID}/metrics").json()
    assert body["unique_visitors"] == 3


def test_metrics_all_staff_clip_unique_visitors_is_zero(client, test_engine):
    seed_entry(test_engine, "VIS_STAFF_A", is_staff=True)
    seed_entry(test_engine, "VIS_STAFF_B", is_staff=True)

    body = client.get(f"/stores/{STORE_ID}/metrics").json()
    assert body["unique_visitors"] == 0


# ---------------------------------------------------------------------------
# Test: Conversion rate
# ---------------------------------------------------------------------------


def test_metrics_conversion_rate_computed_correctly(client, test_engine):
    """
    2 visitors in billing zone before a POS txn → 2 converted out of 5.
    Rate = 2/5 = 0.4
    """
    for v in ["V1", "V2", "V3", "V4", "V5"]:
        seed_entry(test_engine, v)

    billing_time = TODAY - timedelta(minutes=2)
    seed_billing_join(test_engine, "V1", queue_depth=1, ts=billing_time)
    seed_billing_join(test_engine, "V2", queue_depth=2, ts=billing_time)

    # POS txn 1 minute ago (within 5-min conversion window)
    txn_time = TODAY - timedelta(minutes=1)
    seed_pos_txn(test_engine, ts=txn_time)

    body = client.get(f"/stores/{STORE_ID}/metrics").json()
    assert body["unique_visitors"] == 5
    assert abs(body["conversion_rate"] - 0.4) < 0.01


def test_metrics_conversion_rate_not_exceed_1(client, test_engine):
    seed_entry(test_engine, "VIS_ONLY")
    seed_billing_join(test_engine, "VIS_ONLY", queue_depth=1)
    for _ in range(5):
        seed_pos_txn(test_engine, ts=TODAY - timedelta(minutes=1))

    body = client.get(f"/stores/{STORE_ID}/metrics").json()
    assert body["conversion_rate"] <= 1.0


# ---------------------------------------------------------------------------
# Test: Dwell per zone
# ---------------------------------------------------------------------------


def test_metrics_avg_dwell_per_zone_populated(client, test_engine):
    seed_entry(test_engine, "VIS_D1")
    seed_zone_dwell(test_engine, "VIS_D1", "MINIMALIST", dwell_ms=60000)
    seed_zone_dwell(test_engine, "VIS_D1", "MINIMALIST", dwell_ms=90000)

    body = client.get(f"/stores/{STORE_ID}/metrics").json()
    zone_dwells = {z["zone_id"]: z["avg_dwell_ms"] for z in body["avg_dwell_per_zone"]}
    assert "MINIMALIST" in zone_dwells
    # Average of 60000 and 90000 = 75000
    assert abs(zone_dwells["MINIMALIST"] - 75000) < 1


# ---------------------------------------------------------------------------
# Test: Abandonment rate
# ---------------------------------------------------------------------------


def test_metrics_abandonment_rate_computed(client, test_engine):
    """3 billing joins, 1 abandonment → rate = 1/3 ≈ 0.333"""
    seed_billing_join(test_engine, "VIS_Q0", queue_depth=1)
    seed_billing_join(test_engine, "VIS_Q1", queue_depth=2)
    seed_billing_join(test_engine, "VIS_Q2", queue_depth=3)
    seed_billing_abandon(test_engine, "VIS_Q0")

    body = client.get(f"/stores/{STORE_ID}/metrics").json()
    assert abs(body["abandonment_rate"] - 1 / 3) < 0.01


def test_metrics_abandonment_rate_zero_when_no_joins(client, test_engine):
    seed_entry(test_engine, "VIS_X")
    body = client.get(f"/stores/{STORE_ID}/metrics").json()
    assert body["abandonment_rate"] == 0.0


# ---------------------------------------------------------------------------
# Test: Real-time
# ---------------------------------------------------------------------------


def test_metrics_updates_after_new_events(client, test_engine):
    """Metrics reflect newly seeded data on next call."""
    count_before = client.get(f"/stores/{STORE_ID}/metrics").json()["unique_visitors"]
    seed_entry(test_engine, f"VIS_NEW_{uuid.uuid4().hex[:6]}")
    count_after = client.get(f"/stores/{STORE_ID}/metrics").json()["unique_visitors"]
    assert count_after == count_before + 1


# ---------------------------------------------------------------------------
# Test: Unknown store
# ---------------------------------------------------------------------------


def test_metrics_unknown_store_returns_zeros(client):
    body = client.get("/stores/STORE_UNKNOWN/metrics").json()
    assert body["unique_visitors"] == 0
    assert body["conversion_rate"] == 0.0
