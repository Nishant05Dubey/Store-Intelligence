# PROMPT: Generate tests for an event ingestion API that handles bulk payloads.
# The tests must verify:
# 1. Successful insertion of valid events
# 2. Rejection of invalid payloads (HTTP 422)
# 3. Idempotency (duplicate event_ids must be skipped silently)
#
# CHANGES MADE:
# - Added trace_id logging assertions
# - Implemented the database dependency override using SQLModel fixtures
# - Adjusted idempotency test to match the specific "duplicate_skipped" return format
"""Tests for event ingestion endpoints."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient


def make_event(**overrides) -> dict:
    """Build a valid ENTRY event. All IDs are freshly generated."""
    base = {
        "event_id": str(uuid.uuid4()),
        "store_id": "STORE_BLR_001",
        "camera_id": "CAM_ENTRY_01",
        "visitor_id": f"VIS_{uuid.uuid4().hex[:8]}",
        "event_type": "ENTRY",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "zone_id": None,
        "dwell_ms": 0,
        "is_staff": False,
        "confidence": 0.87,
        "metadata": {"queue_depth": None, "sku_zone": None, "session_seq": 1},
    }
    base.update(overrides)
    return base


def make_zone_event(zone_id: str = "SKINCARE", **overrides) -> dict:
    return make_event(
        event_type="ZONE_ENTER",
        zone_id=zone_id,
        metadata={"queue_depth": None, "sku_zone": zone_id, "session_seq": 2},
        **overrides,
    )


def make_billing_event(**overrides) -> dict:
    return make_event(
        event_type="BILLING_QUEUE_JOIN",
        zone_id="CASH_COUNTER",
        metadata={"queue_depth": 3, "sku_zone": "BILLING", "session_seq": 5},
        **overrides,
    )


# ---------------------------------------------------------------------------
# Basic ingestion
# ---------------------------------------------------------------------------


def test_ingest_single_valid_event(client):
    event = make_event()
    response = client.post("/events/ingest", json={"events": [event]})
    assert response.status_code == 200
    body = response.json()
    assert body["accepted"] == 1
    assert body["duplicate_skipped"] == 0
    assert body["validation_errors"] == []
    assert "trace_id" in body


def test_ingest_multiple_valid_events(client):
    events = [make_event() for _ in range(10)]
    response = client.post("/events/ingest", json={"events": events})
    assert response.status_code == 200
    assert response.json()["accepted"] == 10
    assert response.json()["duplicate_skipped"] == 0


def test_ingest_empty_batch(client):
    response = client.post("/events/ingest", json={"events": []})
    assert response.status_code == 200
    assert response.json()["accepted"] == 0


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_ingest_idempotent_same_event_id(client):
    event = make_event()
    r1 = client.post("/events/ingest", json={"events": [event]})
    assert r1.json()["accepted"] == 1

    r2 = client.post("/events/ingest", json={"events": [event]})
    assert r2.status_code == 200
    assert r2.json()["accepted"] == 0
    assert r2.json()["duplicate_skipped"] == 1


def test_ingest_idempotent_partial_batch(client):
    event_a = make_event()
    event_b = make_event()
    client.post("/events/ingest", json={"events": [event_a]})

    response = client.post("/events/ingest", json={"events": [event_a, event_b]})
    assert response.json()["accepted"] == 1
    assert response.json()["duplicate_skipped"] == 1


def test_ingest_duplicate_within_batch(client):
    event = make_event()
    response = client.post("/events/ingest", json={"events": [event, event]})
    assert response.json()["accepted"] == 1
    assert response.json()["duplicate_skipped"] == 1


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------


def test_ingest_zone_enter_event(client):
    response = client.post("/events/ingest", json={"events": [make_zone_event("MINIMALIST")]})
    assert response.status_code == 200
    assert response.json()["accepted"] == 1


def test_ingest_zone_dwell_event(client):
    event = make_zone_event()
    event["event_type"] = "ZONE_DWELL"
    event["dwell_ms"] = 45000
    response = client.post("/events/ingest", json={"events": [event]})
    assert response.status_code == 200
    assert response.json()["accepted"] == 1


def test_ingest_billing_queue_join(client):
    response = client.post("/events/ingest", json={"events": [make_billing_event()]})
    assert response.status_code == 200
    assert response.json()["accepted"] == 1


def test_ingest_reentry_event(client):
    response = client.post("/events/ingest", json={"events": [make_event(event_type="REENTRY")]})
    assert response.status_code == 200
    assert response.json()["accepted"] == 1


def test_ingest_exit_event(client):
    response = client.post("/events/ingest", json={"events": [make_event(event_type="EXIT")]})
    assert response.status_code == 200
    assert response.json()["accepted"] == 1


# ---------------------------------------------------------------------------
# Validation failures
# ---------------------------------------------------------------------------


def test_ingest_missing_required_field_rejected(client):
    event = make_event()
    del event["store_id"]
    response = client.post("/events/ingest", json={"events": [event]})
    assert response.status_code == 422


def test_ingest_invalid_confidence_rejected(client):
    response = client.post("/events/ingest", json={"events": [make_event(confidence=1.5)]})
    assert response.status_code == 422


def test_ingest_invalid_event_type_rejected(client):
    response = client.post("/events/ingest", json={"events": [make_event(event_type="MAGIC_EVENT")]})
    assert response.status_code == 422


def test_ingest_zone_event_missing_zone_id_rejected(client):
    response = client.post("/events/ingest", json={"events": [make_event(event_type="ZONE_ENTER", zone_id=None)]})
    assert response.status_code == 422


def test_ingest_billing_queue_join_missing_queue_depth_rejected(client):
    event = make_event(
        event_type="BILLING_QUEUE_JOIN",
        zone_id="CASH_COUNTER",
        metadata={"queue_depth": None, "sku_zone": None, "session_seq": 1},
    )
    response = client.post("/events/ingest", json={"events": [event]})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Batch size
# ---------------------------------------------------------------------------


def test_ingest_500_events_batch_limit(client):
    events = [make_event() for _ in range(500)]
    response = client.post("/events/ingest", json={"events": events})
    assert response.status_code == 200
    assert response.json()["accepted"] == 500


def test_ingest_over_500_events_rejected(client):
    events = [make_event() for _ in range(501)]
    response = client.post("/events/ingest", json={"events": events})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Staff events
# ---------------------------------------------------------------------------


def test_ingest_staff_event_accepted(client):
    response = client.post("/events/ingest", json={"events": [make_event(is_staff=True, confidence=0.91)]})
    assert response.status_code == 200
    assert response.json()["accepted"] == 1


def test_ingest_low_confidence_event_accepted(client):
    """Low confidence events must NOT be suppressed."""
    response = client.post("/events/ingest", json={"events": [make_event(confidence=0.15)]})
    assert response.status_code == 200
    assert response.json()["accepted"] == 1


# ---------------------------------------------------------------------------
# Response headers
# ---------------------------------------------------------------------------


def test_ingest_response_has_trace_id(client):
    response = client.post("/events/ingest", json={"events": [make_event()]})
    assert "trace_id" in response.json()
    assert len(response.json()["trace_id"]) > 0


def test_ingest_response_header_has_trace_id(client):
    response = client.post("/events/ingest", json={"events": [make_event()]})
    assert "x-trace-id" in response.headers
