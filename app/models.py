"""
Pydantic / SQLModel schema definitions for Store Intelligence events.
All event types must conform to the required output schema.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator
from sqlmodel import Column, Field as SQLField, JSON, SQLModel


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class EventType(str, Enum):
    ENTRY = "ENTRY"
    EXIT = "EXIT"
    ZONE_ENTER = "ZONE_ENTER"
    ZONE_EXIT = "ZONE_EXIT"
    ZONE_DWELL = "ZONE_DWELL"
    BILLING_QUEUE_JOIN = "BILLING_QUEUE_JOIN"
    BILLING_QUEUE_ABANDON = "BILLING_QUEUE_ABANDON"
    REENTRY = "REENTRY"


class AnomalySeverity(str, Enum):
    INFO = "INFO"
    WARN = "WARN"
    CRITICAL = "CRITICAL"


class AnomalyType(str, Enum):
    BILLING_QUEUE_SPIKE = "BILLING_QUEUE_SPIKE"
    CONVERSION_DROP = "CONVERSION_DROP"
    DEAD_ZONE = "DEAD_ZONE"
    STALE_FEED = "STALE_FEED"


# ---------------------------------------------------------------------------
# Event metadata nested model
# ---------------------------------------------------------------------------


class EventMetadata(BaseModel):
    queue_depth: Optional[int] = Field(
        default=None,
        description="Queue depth at billing counter; populated for BILLING_QUEUE_JOIN",
    )
    sku_zone: Optional[str] = Field(
        default=None, description="Zone label from store_layout"
    )
    session_seq: Optional[int] = Field(
        default=None, description="Ordinal position of this event in visitor's session"
    )

    model_config = {"extra": "allow"}


# ---------------------------------------------------------------------------
# Inbound event (POST /events/ingest)
# ---------------------------------------------------------------------------


class StoreEvent(BaseModel):
    """Single event emitted by the detection pipeline."""

    event_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="UUID v4 — must be globally unique",
    )
    store_id: str = Field(..., description="Store identifier from store_layout.json")
    camera_id: str = Field(..., description="Camera that produced this event")
    visitor_id: str = Field(
        ..., description="Re-ID token — unique per visit session"
    )
    event_type: EventType
    timestamp: datetime = Field(..., description="ISO-8601 UTC timestamp")
    zone_id: Optional[str] = Field(
        default=None, description="Zone identifier; null for ENTRY/EXIT events"
    )
    dwell_ms: int = Field(
        default=0, ge=0, description="Duration in ms; 0 for instantaneous events"
    )
    is_staff: bool = Field(
        ..., description="True if detected person is classified as store staff"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Detection confidence — never suppressed"
    )
    metadata: EventMetadata = Field(default_factory=EventMetadata)

    @field_validator("zone_id")
    @classmethod
    def zone_required_for_zone_events(cls, v: Optional[str], info: Any) -> Optional[str]:
        return v

    @model_validator(mode="after")
    def validate_zone_consistency(self) -> "StoreEvent":
        """Zone events must have a zone_id; ENTRY/EXIT/REENTRY must not."""
        zone_required = {
            EventType.ZONE_ENTER,
            EventType.ZONE_EXIT,
            EventType.ZONE_DWELL,
            EventType.BILLING_QUEUE_JOIN,
            EventType.BILLING_QUEUE_ABANDON,
        }
        if self.event_type in zone_required and not self.zone_id:
            raise ValueError(
                f"event_type={self.event_type} requires a zone_id"
            )
        if self.event_type == EventType.BILLING_QUEUE_JOIN:
            if self.metadata.queue_depth is None:
                raise ValueError(
                    "BILLING_QUEUE_JOIN requires metadata.queue_depth to be set"
                )
        return self


# ---------------------------------------------------------------------------
# Ingest request / response
# ---------------------------------------------------------------------------


class IngestRequest(BaseModel):
    events: list[StoreEvent] = Field(
        ..., max_length=500, description="Batch of up to 500 events"
    )


class IngestResponse(BaseModel):
    accepted: int
    duplicate_skipped: int
    validation_errors: list[dict[str, Any]]
    trace_id: str


# ---------------------------------------------------------------------------
# Database row model (SQLModel)
# ---------------------------------------------------------------------------


class EventRow(SQLModel, table=True):
    __tablename__ = "events"

    event_id: str = SQLField(primary_key=True)
    store_id: str = SQLField(index=True)
    camera_id: str
    visitor_id: str = SQLField(index=True)
    event_type: str = SQLField(index=True)
    timestamp: str = SQLField(index=True)  # stored as ISO string for SQLite compat
    zone_id: Optional[str] = SQLField(default=None, index=True)
    dwell_ms: int = SQLField(default=0)
    is_staff: bool = SQLField(default=False)
    confidence: float
    queue_depth: Optional[int] = SQLField(default=None)
    sku_zone: Optional[str] = SQLField(default=None)
    session_seq: Optional[int] = SQLField(default=None)
    ingested_at: str = SQLField(
        default_factory=lambda: datetime.utcnow().isoformat()
    )


class PosTransactionRow(SQLModel, table=True):
    __tablename__ = "pos_transactions"

    transaction_id: str = SQLField(primary_key=True)
    store_id: str = SQLField(index=True)
    timestamp: str = SQLField(index=True)
    basket_value: float = SQLField(default=0.0)
    customer_number: Optional[str] = SQLField(default=None)


# ---------------------------------------------------------------------------
# API response models
# ---------------------------------------------------------------------------


class ZoneDwellStats(BaseModel):
    zone_id: str
    avg_dwell_ms: float
    visit_count: int


class MetricsResponse(BaseModel):
    store_id: str
    window: str = "today"
    unique_visitors: int
    conversion_rate: float
    avg_dwell_per_zone: list[ZoneDwellStats]
    queue_depth_current: int
    abandonment_rate: float
    computed_at: datetime


class FunnelStage(BaseModel):
    stage: str
    count: int
    drop_off_pct: float


class FunnelResponse(BaseModel):
    store_id: str
    stages: list[FunnelStage]
    session_unit: bool = True
    reentry_excluded: bool = True


class HeatmapZone(BaseModel):
    zone_id: str
    zone_name: str
    visit_frequency: int
    avg_dwell_ms: float
    heat_score: float  # normalised 0-100


class HeatmapResponse(BaseModel):
    store_id: str
    zones: list[HeatmapZone]
    data_confidence: bool  # False if fewer than 20 sessions in window
    computed_at: datetime


class AnomalyItem(BaseModel):
    anomaly_type: str
    severity: AnomalySeverity
    detail: str
    detected_at: datetime
    suggested_action: str


class AnomaliesResponse(BaseModel):
    store_id: str
    anomalies: list[AnomalyItem]
    computed_at: datetime


class CameraHealth(BaseModel):
    camera_id: str
    last_event_at: Optional[datetime]
    is_stale: bool
    lag_minutes: Optional[float]


class HealthResponse(BaseModel):
    status: str  # "ok" | "degraded" | "down"
    db_ok: bool
    cameras: list[CameraHealth]
    computed_at: datetime
