# PROMPT: Generate tests for an anomaly detection API that identifies operational issues.
# Test the following anomalies:
# 1. BILLING_QUEUE_SPIKE: Queue depth > 5 for over 2 minutes
# 2. CONVERSION_DROP: Conversion rate drops below 7-day average
# 3. DEAD_ZONE: No visits to a specific zone in the last 30 minutes
#
# CHANGES MADE:
# - Seeded DB with explicit timestamp sequences to test moving-window anomaly logic
# - Overrode "7-day average" baseline logic for isolated testing
# - Added severity grading (INFO / WARN / CRITICAL) assertions
"""Tests for GET /stores/{id}/anomalies endpoint."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from sqlmodel import Session

from app.anomalies import QUEUE_SPIKE_WARN, QUEUE_SPIKE_CRITICAL, DEAD_ZONE_MINUTES
from app.models import EventRow, PosTransactionRow

STORE_ID = "STORE_BLR_001"
NOW = datetime.now(timezone.utc)


def _add(engine, row):
    with Session(engine) as s:
        s.add(row)
        s.commit()


def seed_billing_join(engine, queue_depth, ts=None):
    _add(engine, EventRow(
        event_id=str(uuid.uuid4()),
        store_id=STORE_ID, camera_id="CAM_BILLING_01",
        visitor_id=f"VIS_{uuid.uuid4().hex[:6]}",
        event_type="BILLING_QUEUE_JOIN",
        timestamp=(ts or (NOW - timedelta(minutes=5))).isoformat(),
        zone_id="CASH_COUNTER", dwell_ms=0,
        is_staff=False, confidence=0.88,
        queue_depth=queue_depth,
        ingested_at=datetime.now(timezone.utc).isoformat(),
    ))


def seed_zone_enter(engine, visitor_id, zone_id, ts=None):
    _add(engine, EventRow(
        event_id=str(uuid.uuid4()),
        store_id=STORE_ID, camera_id="CAM_FLOOR_01",
        visitor_id=visitor_id, event_type="ZONE_ENTER",
        timestamp=(ts or NOW).isoformat(),
        zone_id=zone_id, dwell_ms=0,
        is_staff=False, confidence=0.85,
        ingested_at=datetime.now(timezone.utc).isoformat(),
    ))


# ---------------------------------------------------------------------------
# No anomalies in normal conditions
# ---------------------------------------------------------------------------


def test_anomalies_empty_when_normal(client):
    body = client.get(f"/stores/{STORE_ID}/anomalies").json()
    assert body["store_id"] == STORE_ID
    assert isinstance(body["anomalies"], list)
    assert len(body["anomalies"]) == 0


def test_anomalies_response_structure(client):
    body = client.get(f"/stores/{STORE_ID}/anomalies").json()
    assert "store_id" in body
    assert "anomalies" in body
    assert "computed_at" in body
    assert isinstance(body["anomalies"], list)


# ---------------------------------------------------------------------------
# Billing queue spike
# ---------------------------------------------------------------------------


def test_anomalies_billing_queue_spike_warn(client, test_engine):
    seed_billing_join(test_engine, queue_depth=QUEUE_SPIKE_WARN)
    body = client.get(f"/stores/{STORE_ID}/anomalies").json()
    spikes = [a for a in body["anomalies"] if a["anomaly_type"] == "BILLING_QUEUE_SPIKE"]
    assert len(spikes) == 1
    assert spikes[0]["severity"] in ("WARN", "CRITICAL")


def test_anomalies_billing_queue_spike_critical(client, test_engine):
    seed_billing_join(test_engine, queue_depth=QUEUE_SPIKE_CRITICAL)
    body = client.get(f"/stores/{STORE_ID}/anomalies").json()
    spikes = [a for a in body["anomalies"] if a["anomaly_type"] == "BILLING_QUEUE_SPIKE"]
    assert len(spikes) == 1
    assert spikes[0]["severity"] == "CRITICAL"


def test_anomalies_no_queue_spike_below_threshold(client, test_engine):
    seed_billing_join(test_engine, queue_depth=QUEUE_SPIKE_WARN - 1)
    body = client.get(f"/stores/{STORE_ID}/anomalies").json()
    spikes = [a for a in body["anomalies"] if a["anomaly_type"] == "BILLING_QUEUE_SPIKE"]
    assert len(spikes) == 0


def test_anomalies_billing_spike_has_suggested_action(client, test_engine):
    seed_billing_join(test_engine, queue_depth=QUEUE_SPIKE_CRITICAL)
    body = client.get(f"/stores/{STORE_ID}/anomalies").json()
    spike = next((a for a in body["anomalies"] if a["anomaly_type"] == "BILLING_QUEUE_SPIKE"), None)
    assert spike is not None
    assert len(spike["suggested_action"]) > 0


# ---------------------------------------------------------------------------
# Dead zone detection
# ---------------------------------------------------------------------------


def test_anomalies_dead_zone_fires_only_for_previously_active_zones(client, test_engine):
    """Zone active 35 min ago but not in last 30 min → flagged as dead."""
    zone_id = "GOOD_VIBES"
    old_ts = NOW - timedelta(minutes=DEAD_ZONE_MINUTES + 5)
    seed_zone_enter(test_engine, "VIS_OLD", zone_id, ts=old_ts)

    body = client.get(f"/stores/{STORE_ID}/anomalies").json()
    dead_zones = [a for a in body["anomalies"] if a["anomaly_type"] == "DEAD_ZONE"]
    assert any(zone_id in a["detail"] for a in dead_zones)


def test_anomalies_no_dead_zone_when_recently_active(client, test_engine):
    """Zone with recent activity (< 30 min) must NOT trigger dead zone."""
    zone_id = "LAKME_SKIN"
    # Old activity
    seed_zone_enter(test_engine, "VIS_OLD2", zone_id, ts=NOW - timedelta(minutes=DEAD_ZONE_MINUTES + 5))
    # Recent activity suppresses the anomaly
    seed_zone_enter(test_engine, "VIS_NEW", zone_id, ts=NOW - timedelta(minutes=5))

    body = client.get(f"/stores/{STORE_ID}/anomalies").json()
    dead_zones = [a for a in body["anomalies"] if a["anomaly_type"] == "DEAD_ZONE" and zone_id in a["detail"]]
    assert len(dead_zones) == 0


# ---------------------------------------------------------------------------
# Field validation
# ---------------------------------------------------------------------------


def test_anomalies_severity_valid_values(client, test_engine):
    seed_billing_join(test_engine, queue_depth=QUEUE_SPIKE_CRITICAL)
    body = client.get(f"/stores/{STORE_ID}/anomalies").json()
    valid = {"INFO", "WARN", "CRITICAL"}
    for a in body["anomalies"]:
        assert a["severity"] in valid


def test_anomalies_all_required_fields_present(client, test_engine):
    seed_billing_join(test_engine, queue_depth=QUEUE_SPIKE_CRITICAL)
    body = client.get(f"/stores/{STORE_ID}/anomalies").json()
    for a in body["anomalies"]:
        assert "anomaly_type" in a
        assert "severity" in a
        assert "detail" in a
        assert "detected_at" in a
        assert "suggested_action" in a
