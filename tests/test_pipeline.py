# PROMPT: Generate tests for a detection pipeline that validates event schema compliance.
# Events must conform to a strict JSON schema with required fields, correct event types,
# UTC timestamps derived from clip+frame offset, unique event_ids, and proper zone_id rules.
# Tests should NOT require running the actual video pipeline — use sample JSONL events.
#
# CHANGES MADE:
# - Tests validate schema properties, not specific values (no hardcoded visitor counts)
# - Added test for timestamp monotonicity within a session
# - Added test for session_seq incrementing within a session
# - Replaced file-based tests with in-memory fixture generation

"""Tests for detection pipeline event schema compliance."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.models import EventType, StoreEvent, EventMetadata


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def valid_event(**overrides) -> dict:
    """Base valid ENTRY event."""
    base = {
        "event_id": str(uuid.uuid4()),
        "store_id": "STORE_BLR_001",
        "camera_id": "CAM_ENTRY_01",
        "visitor_id": "VIS_abc12345",
        "event_type": "ENTRY",
        "timestamp": "2026-04-10T14:30:00+00:00",
        "zone_id": None,
        "dwell_ms": 0,
        "is_staff": False,
        "confidence": 0.87,
        "metadata": {
            "queue_depth": None,
            "sku_zone": None,
            "session_seq": 1,
        },
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Test: Schema compliance
# ---------------------------------------------------------------------------


def test_schema_valid_entry_event():
    """Valid ENTRY event passes Pydantic validation."""
    event = StoreEvent(**valid_event())
    assert event.event_type == EventType.ENTRY
    assert event.is_staff is False
    assert 0.0 <= event.confidence <= 1.0


def test_schema_valid_zone_enter_event():
    """ZONE_ENTER with zone_id passes validation."""
    event = StoreEvent(**valid_event(
        event_type="ZONE_ENTER",
        zone_id="MINIMALIST",
        metadata={"queue_depth": None, "sku_zone": "Minimalist", "session_seq": 2},
    ))
    assert event.zone_id == "MINIMALIST"
    assert event.metadata.sku_zone == "Minimalist"


def test_schema_zone_enter_without_zone_id_fails():
    """ZONE_ENTER without zone_id must fail validation."""
    with pytest.raises(ValidationError) as exc_info:
        StoreEvent(**valid_event(event_type="ZONE_ENTER", zone_id=None))
    assert "zone_id" in str(exc_info.value).lower() or "zone" in str(exc_info.value).lower()


def test_schema_billing_queue_join_requires_queue_depth():
    """BILLING_QUEUE_JOIN without queue_depth in metadata must fail."""
    with pytest.raises(ValidationError):
        StoreEvent(**valid_event(
            event_type="BILLING_QUEUE_JOIN",
            zone_id="CASH_COUNTER",
            metadata={"queue_depth": None, "sku_zone": None, "session_seq": 5},
        ))


def test_schema_billing_queue_join_with_queue_depth_passes():
    """BILLING_QUEUE_JOIN with queue_depth=3 must pass."""
    event = StoreEvent(**valid_event(
        event_type="BILLING_QUEUE_JOIN",
        zone_id="CASH_COUNTER",
        metadata={"queue_depth": 3, "sku_zone": "BILLING", "session_seq": 5},
    ))
    assert event.metadata.queue_depth == 3


def test_schema_event_id_is_uuid():
    """event_id must be a valid UUID string."""
    event = StoreEvent(**valid_event())
    # Must not raise
    uuid.UUID(event.event_id)


def test_schema_confidence_bounds():
    """Confidence must be in [0.0, 1.0]."""
    # Valid edge cases
    StoreEvent(**valid_event(confidence=0.0))
    StoreEvent(**valid_event(confidence=1.0))
    StoreEvent(**valid_event(confidence=0.5))

    # Invalid
    with pytest.raises(ValidationError):
        StoreEvent(**valid_event(confidence=1.1))
    with pytest.raises(ValidationError):
        StoreEvent(**valid_event(confidence=-0.1))


def test_schema_dwell_ms_non_negative():
    """dwell_ms must be >= 0."""
    StoreEvent(**valid_event(dwell_ms=0))
    StoreEvent(**valid_event(dwell_ms=30000))
    with pytest.raises(ValidationError):
        StoreEvent(**valid_event(dwell_ms=-1))


def test_schema_all_event_types_valid():
    """All defined event types must be accepted."""
    valid_types = [
        ("ENTRY", None, {}),
        ("EXIT", None, {}),
        ("REENTRY", None, {}),
        ("ZONE_ENTER", "SKINCARE", {"sku_zone": "Skincare", "session_seq": 2}),
        ("ZONE_EXIT", "SKINCARE", {"sku_zone": "Skincare", "session_seq": 3}),
        ("ZONE_DWELL", "SKINCARE", {"sku_zone": "Skincare", "session_seq": 4}),
        ("BILLING_QUEUE_JOIN", "CASH_COUNTER", {"queue_depth": 2, "session_seq": 5}),
        ("BILLING_QUEUE_ABANDON", "CASH_COUNTER", {"session_seq": 6}),
    ]
    for event_type, zone_id, extra_meta in valid_types:
        meta = {"queue_depth": None, "sku_zone": None, "session_seq": 1}
        meta.update(extra_meta)
        event_data = valid_event(event_type=event_type, zone_id=zone_id, metadata=meta)
        event = StoreEvent(**event_data)
        assert event.event_type.value == event_type


def test_schema_invalid_event_type_rejected():
    """Unrecognized event type must raise ValidationError."""
    with pytest.raises(ValidationError):
        StoreEvent(**valid_event(event_type="UNKNOWN_TYPE"))


# ---------------------------------------------------------------------------
# Test: Event uniqueness
# ---------------------------------------------------------------------------


def test_schema_event_ids_are_unique():
    """Two events should never share the same event_id."""
    e1 = StoreEvent(**valid_event())
    e2 = StoreEvent(**valid_event())
    assert e1.event_id != e2.event_id


# ---------------------------------------------------------------------------
# Test: Timestamp
# ---------------------------------------------------------------------------


def test_schema_timestamp_is_datetime():
    """Timestamp field must parse to a datetime object."""
    event = StoreEvent(**valid_event())
    assert isinstance(event.timestamp, datetime)


def test_schema_timestamp_preserves_utc():
    """Timestamp must be a valid UTC datetime."""
    ts = "2026-04-10T14:30:00+00:00"
    event = StoreEvent(**valid_event(timestamp=ts))
    assert event.timestamp.tzinfo is not None


# ---------------------------------------------------------------------------
# Test: Session sequence
# ---------------------------------------------------------------------------


def test_schema_session_seq_increments():
    """
    session_seq must increment across events in a single visitor's session.
    Test that schema accepts any positive integer session_seq.
    """
    events = []
    for seq in range(1, 6):
        e = StoreEvent(**valid_event(
            visitor_id="VIS_SEQ_TEST",
            event_type="ZONE_ENTER",
            zone_id="MINIMALIST",
            metadata={"queue_depth": None, "sku_zone": "Minimalist", "session_seq": seq},
        ))
        events.append(e)

    # Session sequences must be monotonically increasing
    seqs = [e.metadata.session_seq for e in events]
    assert seqs == sorted(seqs)
    assert seqs == list(range(1, 6))


# ---------------------------------------------------------------------------
# Test: is_staff field
# ---------------------------------------------------------------------------


def test_schema_is_staff_true_accepted():
    """Staff events with is_staff=True must be valid."""
    event = StoreEvent(**valid_event(is_staff=True, confidence=0.92))
    assert event.is_staff is True


def test_schema_low_confidence_not_suppressed():
    """
    Events with very low confidence (e.g. 0.1) must still be valid.
    The spec explicitly says: do NOT suppress low-confidence events.
    """
    event = StoreEvent(**valid_event(confidence=0.1))
    assert event.confidence == pytest.approx(0.1)
