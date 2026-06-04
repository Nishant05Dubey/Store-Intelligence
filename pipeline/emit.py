"""
emit.py — Construct and emit schema-compliant store events.

Converts tracker state changes into structured JSON events and
sends them to the Store Intelligence API via HTTP POST /events/ingest.

Event types emitted:
  ENTRY, EXIT, ZONE_ENTER, ZONE_EXIT, ZONE_DWELL,
  BILLING_QUEUE_JOIN, BILLING_QUEUE_ABANDON, REENTRY
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import Optional

import requests
import structlog

logger = structlog.get_logger(__name__)

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
BATCH_SIZE = 1  # emit events in batches of this size


class EventEmitter:
    """
    Accumulates events and flushes them to the API in batches.
    Thread-safe for single-process use.
    """

    def __init__(
        self,
        store_id: str,
        camera_id: str,
        clip_start_utc: datetime,
        fps: float,
        api_url: str = API_BASE_URL,
    ):
        self.store_id = store_id
        self.camera_id = camera_id
        self.clip_start_utc = clip_start_utc
        self.fps = fps
        self.api_url = api_url
        self._buffer: list[dict] = []
        self._session_seq: dict[str, int] = {}  # visitor_id → sequence counter

    def frame_to_timestamp(self, frame_number: int) -> datetime:
        """Convert frame number to UTC timestamp based on clip start time."""
        offset_seconds = frame_number / self.fps
        return datetime.fromtimestamp(
            self.clip_start_utc.timestamp() + offset_seconds,
            tz=timezone.utc,
        )

    def _next_seq(self, visitor_id: str) -> int:
        seq = self._session_seq.get(visitor_id, 0) + 1
        self._session_seq[visitor_id] = seq
        return seq

    def emit_entry(
        self,
        visitor_id: str,
        frame_number: int,
        confidence: float,
        is_staff: bool,
        is_reentry: bool = False,
    ) -> None:
        event_type = "REENTRY" if is_reentry else "ENTRY"
        if is_reentry:
            # Reset session sequence on reentry
            self._session_seq[visitor_id] = 0

        self._buffer.append(
            self._build_event(
                visitor_id=visitor_id,
                event_type=event_type,
                frame_number=frame_number,
                confidence=confidence,
                is_staff=is_staff,
                zone_id=None,
                dwell_ms=0,
            )
        )
        self._maybe_flush()

    def emit_exit(
        self,
        visitor_id: str,
        frame_number: int,
        confidence: float,
        is_staff: bool,
    ) -> None:
        self._buffer.append(
            self._build_event(
                visitor_id=visitor_id,
                event_type="EXIT",
                frame_number=frame_number,
                confidence=confidence,
                is_staff=is_staff,
                zone_id=None,
                dwell_ms=0,
            )
        )
        self._maybe_flush()

    def emit_zone_enter(
        self,
        visitor_id: str,
        frame_number: int,
        zone_id: str,
        sku_zone: Optional[str],
        confidence: float,
        is_staff: bool,
    ) -> None:
        self._buffer.append(
            self._build_event(
                visitor_id=visitor_id,
                event_type="ZONE_ENTER",
                frame_number=frame_number,
                confidence=confidence,
                is_staff=is_staff,
                zone_id=zone_id,
                dwell_ms=0,
                sku_zone=sku_zone,
            )
        )
        self._maybe_flush()

    def emit_zone_exit(
        self,
        visitor_id: str,
        frame_number: int,
        zone_id: str,
        dwell_ms: int,
        confidence: float,
        is_staff: bool,
    ) -> None:
        self._buffer.append(
            self._build_event(
                visitor_id=visitor_id,
                event_type="ZONE_EXIT",
                frame_number=frame_number,
                confidence=confidence,
                is_staff=is_staff,
                zone_id=zone_id,
                dwell_ms=dwell_ms,
            )
        )
        self._maybe_flush()

    def emit_zone_dwell(
        self,
        visitor_id: str,
        frame_number: int,
        zone_id: str,
        dwell_ms: int,
        confidence: float,
        is_staff: bool,
    ) -> None:
        """Emitted every 30 seconds of continuous zone presence."""
        self._buffer.append(
            self._build_event(
                visitor_id=visitor_id,
                event_type="ZONE_DWELL",
                frame_number=frame_number,
                confidence=confidence,
                is_staff=is_staff,
                zone_id=zone_id,
                dwell_ms=dwell_ms,
            )
        )
        self._maybe_flush()

    def emit_billing_queue_join(
        self,
        visitor_id: str,
        frame_number: int,
        queue_depth: int,
        confidence: float,
        is_staff: bool,
    ) -> None:
        self._buffer.append(
            self._build_event(
                visitor_id=visitor_id,
                event_type="BILLING_QUEUE_JOIN",
                frame_number=frame_number,
                confidence=confidence,
                is_staff=is_staff,
                zone_id="CASH_COUNTER",
                dwell_ms=0,
                queue_depth=queue_depth,
            )
        )
        self._maybe_flush()

    def emit_billing_queue_abandon(
        self,
        visitor_id: str,
        frame_number: int,
        dwell_ms: int,
        confidence: float,
        is_staff: bool,
    ) -> None:
        self._buffer.append(
            self._build_event(
                visitor_id=visitor_id,
                event_type="BILLING_QUEUE_ABANDON",
                frame_number=frame_number,
                confidence=confidence,
                is_staff=is_staff,
                zone_id="CASH_COUNTER",
                dwell_ms=dwell_ms,
            )
        )
        self._maybe_flush()

    def flush(self) -> int:
        """Send all buffered events to the API. Returns number sent."""
        if not self._buffer:
            return 0

        batch = self._buffer[:]
        self._buffer.clear()

        try:
            resp = requests.post(
                f"{self.api_url}/events/ingest",
                json={"events": batch},
                timeout=30,
            )
            resp.raise_for_status()
            result = resp.json()
            logger.info(
                "emit.flushed",
                sent=len(batch),
                accepted=result.get("accepted"),
                duplicates=result.get("duplicate_skipped"),
            )
            return len(batch)
        except requests.RequestException as exc:
            logger.error("emit.flush_failed", error=str(exc), queued=len(batch))
            # Put events back in buffer for retry
            self._buffer = batch + self._buffer
            return 0

    def _maybe_flush(self) -> None:
        if len(self._buffer) >= BATCH_SIZE:
            self.flush()

    def _build_event(
        self,
        visitor_id: str,
        event_type: str,
        frame_number: int,
        confidence: float,
        is_staff: bool,
        zone_id: Optional[str],
        dwell_ms: int,
        sku_zone: Optional[str] = None,
        queue_depth: Optional[int] = None,
    ) -> dict:
        """Build a schema-compliant event dictionary."""
        ts = self.frame_to_timestamp(frame_number)
        seq = self._next_seq(visitor_id)
        return {
            "event_id": str(uuid.uuid4()),
            "store_id": self.store_id,
            "camera_id": self.camera_id,
            "visitor_id": visitor_id,
            "event_type": event_type,
            "timestamp": ts.isoformat(),
            "zone_id": zone_id,
            "dwell_ms": dwell_ms,
            "is_staff": is_staff,
            "confidence": round(float(confidence), 4),
            "metadata": {
                "queue_depth": queue_depth,
                "sku_zone": sku_zone,
                "session_seq": seq,
            },
        }
