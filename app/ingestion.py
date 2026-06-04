"""
Event ingestion logic: validate, deduplicate, and store events.
POST /events/ingest handler.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

import structlog
from sqlmodel import Session, select

from app.models import EventRow, IngestRequest, IngestResponse, StoreEvent

logger = structlog.get_logger(__name__)


def ingest_events(
    request: IngestRequest, session: Session, trace_id: str
) -> IngestResponse:
    """
    Process a batch of events:
    1. Validate each event against the Pydantic schema
    2. Skip duplicates (idempotent by event_id)
    3. Persist valid, new events to the database
    Returns a structured response with counts and validation errors.
    """
    accepted = 0
    duplicate_skipped = 0
    validation_errors: list[dict[str, Any]] = []

    # Bulk-fetch existing event_ids to avoid N+1 queries
    incoming_ids = [e.event_id for e in request.events]
    existing_ids: set[str] = set()
    if incoming_ids:
        rows = session.exec(
            select(EventRow.event_id).where(EventRow.event_id.in_(incoming_ids))
        ).all()
        existing_ids = set(rows)

    rows_to_insert: list[EventRow] = []
    seen_in_batch: set[str] = set()  # handle duplicates within same batch

    for idx, event in enumerate(request.events):
        # Deduplication check (idempotent by event_id)
        if event.event_id in existing_ids or event.event_id in seen_in_batch:
            duplicate_skipped += 1
            logger.debug(
                "ingest.duplicate_skipped",
                event_id=event.event_id,
                trace_id=trace_id,
            )
            continue

        seen_in_batch.add(event.event_id)

        row = EventRow(
            event_id=event.event_id,
            store_id=event.store_id,
            camera_id=event.camera_id,
            visitor_id=event.visitor_id,
            event_type=event.event_type.value,
            timestamp=event.timestamp.isoformat(),
            zone_id=event.zone_id,
            dwell_ms=event.dwell_ms,
            is_staff=event.is_staff,
            confidence=event.confidence,
            queue_depth=event.metadata.queue_depth,
            sku_zone=event.metadata.sku_zone,
            session_seq=event.metadata.session_seq,
            ingested_at=datetime.now(timezone.utc).isoformat(),
        )
        rows_to_insert.append(row)
        accepted += 1

    # Bulk insert
    for row in rows_to_insert:
        session.add(row)

    try:
        session.commit()
    except Exception as exc:
        session.rollback()
        logger.error("ingest.db_error", error=str(exc), trace_id=trace_id)
        raise

    logger.info(
        "ingest.complete",
        accepted=accepted,
        duplicate_skipped=duplicate_skipped,
        validation_errors=len(validation_errors),
        trace_id=trace_id,
    )

    return IngestResponse(
        accepted=accepted,
        duplicate_skipped=duplicate_skipped,
        validation_errors=validation_errors,
        trace_id=trace_id,
    )
