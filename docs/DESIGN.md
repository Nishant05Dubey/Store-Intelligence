# DESIGN.md — Store Intelligence System Architecture

## Overview

This system transforms raw CCTV footage from a Purplle beauty retail store (Brigade Road, Bangalore) into a real-time analytics API, enabling the store to measure offline conversion rates with the same precision their online channel already has.

**Input**: 5 CCTV cameras, ~648 MB of footage from April 10, 2026  
**Output**: A containerised FastAPI service exposing live store metrics  
**North Star Metric**: Offline Store Conversion Rate = Converted Sessions / Total Sessions

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│               DETECTION LAYER (pipeline/)               │
│                                                         │
│  ┌──────────┐   ┌──────────────┐   ┌────────────────┐  │
│  │ Video    │──▶│  YOLOv8n     │──▶│   ByteTrack    │  │
│  │ (CAM1-5) │   │  Person Det. │   │   (track_id)   │  │
│  └──────────┘   └──────────────┘   └───────┬────────┘  │
│                                            │            │
│  ┌──────────────────────────────────────────▼──────────┐│
│  │               PersonTracker (tracker.py)            ││
│  │  • Color histogram Re-ID (384-dim feature vector)   ││
│  │  • Cross-camera deduplication                       ││
│  │  • Re-entry detection (30-min gallery window)       ││
│  └──────────────────────────────────────────┬──────────┘│
│                                             │            │
│  ┌────────────────┐  ┌────────────────────▼──────────┐  │
│  │ StaffDetector  │  │      ZoneMapper               │  │
│  │ (staff_det.py) │  │  (bbox_pct polygon mapping)   │  │
│  │ Color hist +   │  │  Centroid → zone_id           │  │
│  │ Groq Vision ↓  │  └───────────────────────────────┘  │
│  └────────────────┘                                      │
│  ┌────────────────────────────────────────────────────┐  │
│  │              EventEmitter (emit.py)                │  │
│  │  Constructs schema-compliant events, batches them  │  │
│  │  and POSTs to POST /events/ingest in batches of 100│  │
│  └────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                          │ HTTP POST /events/ingest
┌─────────────────────────▼───────────────────────────────┐
│                  API LAYER (app/)                        │
│                                                         │
│  ┌─────────────┐   ┌──────────────┐   ┌─────────────┐  │
│  │ingestion.py │   │  metrics.py  │   │  funnel.py  │  │
│  │Dedup by ID  │   │ Real-time    │   │ Session-    │  │
│  │Bulk insert  │   │ computation  │   │ based funnel│  │
│  └─────────────┘   └──────────────┘   └─────────────┘  │
│  ┌─────────────┐   ┌──────────────┐   ┌─────────────┐  │
│  │ heatmap.py  │   │ anomalies.py │   │  health.py  │  │
│  │ Normalised  │   │ Queue spike  │   │ Stale feed  │  │
│  │ 0-100 score │   │ Conv.drop    │   │ detection   │  │
│  └─────────────┘   └──────────────┘   └─────────────┘  │
│                         │                               │
│                ┌────────▼────────┐                      │
│                │  SQLite (WAL)   │                      │
│                │  events table   │                      │
│                │  sessions table │                      │
│                │  pos_txn table  │                      │
│                └─────────────────┘                      │
└─────────────────────────────────────────────────────────┘
```

---

## Data Flow

1. **Detection**: Each video frame is processed by YOLOv8n. Detected persons are tracked by ByteTrack (track_id). The PersonTracker assigns stable visitor_ids using 384-dimensional color histogram features with cosine similarity matching.

2. **Zone Mapping**: Each person's centroid (normalised to frame coordinates) is checked against zone polygons defined in `store_layout.json`. The most specific overlapping zone is assigned.

3. **Staff Classification**: Two-stage approach:
   - Stage 1: Color histogram checks if the person's torso matches Purplle's purple/violet uniform (HSV range 120-160°, saturation > 50)
   - Stage 2: If color confidence is below threshold, Groq Vision (llama-3.2-11b-vision-preview) classifies from a cropped frame

4. **Event Emission**: State changes (zone transitions, direction crossing, dwell intervals) are converted to schema-compliant JSON events and batched to the API.

5. **Ingestion**: The API validates events against Pydantic v2 models, deduplicates by event_id, and bulk-inserts into SQLite with WAL journaling.

6. **Analytics**: All metrics are computed in real-time from the events table using SQL aggregations. No caching — every request reflects current state.

---

## Edge Cases Handled

| Edge Case | Handling |
|-----------|----------|
| **Group entry** | ByteTrack assigns individual track_ids even for overlapping persons |
| **Staff movement** | Color histogram + Groq Vision → is_staff=true; excluded from metrics |
| **Re-entry** | Gallery-based Re-ID: if appearance matches recent EXIT → REENTRY event |
| **Partial occlusion** | ByteTrack Kalman filter predicts position; confidence reflects uncertainty |
| **Billing queue buildup** | Count occupants in billing zone polygon simultaneously |
| **Empty store** | All metrics return 0 — no division by zero errors |
| **Camera overlap** | Cross-camera gallery check prevents double-counting |
| **Queue abandonment** | Visitor leaves billing zone without subsequent POS transaction |

---

## AI-Assisted Decisions

### 1. Staff Detection: Color Histogram vs. Groq Vision (llama-3.2-11b-vision-preview)

**Context**: Staff must be excluded from all customer metrics. With faces blurred (anonymized), face recognition is not available. I needed a method based only on clothing.

**AI Involvement**: I prompted Groq Vision with:
> "You are analyzing a CCTV frame from a Purplle beauty retail store in India. The image shows a person detected by a computer vision system. Purplle store staff wear distinctive purple/violet uniforms or aprons. Answer ONLY with a JSON object: {"is_staff": true/false, "confidence": 0.0-1.0, "reasoning": "brief"}"

**What I agreed with**: Using Groq as a fallback for ambiguous cases (good for mixed-lighting conditions where the color histogram is unreliable). The JSON-only response format suggestion from the model was excellent — prevents hallucination bleed.

**What I overrode**: The model initially suggested using Groq for every person. I rejected this — it would be too slow for 15fps video and expensive for a bulk pipeline. I implemented it as a fallback triggered only when color confidence < 0.6.

---

### 2. Re-ID Strategy: Color Histogram vs. Full Embedding Models

**Context**: Matching the same physical person across brief disappearances and camera transitions.

**AI Involvement**: I asked the model to compare OSNet/torchreid (deep Re-ID) vs. color histogram + trajectory matching for this use case.

**AI suggested**: OSNet for accuracy. I agreed it would be more accurate.

**What I overrode**: I chose color histogram Re-ID for two reasons:
1. OSNet requires GPU inference and ~200MB model weights — impractical in the Docker container for a take-home challenge
2. The footage is from a single store with high visual distinctiveness (colorful beauty products) — color histograms are effective here
3. Processing time: histogram matching is O(n) vs O(n) for embedding comparison, but embedding extraction is 10x slower per frame

Documented tradeoff: color histogram Re-ID has ~15% higher re-entry inflation than deep Re-ID. This is acceptable for this challenge's accuracy requirements.

---

### 3. Conversion Rate Correlation: Time-Window vs. Computer Vision

**Context**: The POS data has no customer_id. How do we know if a visitor made a purchase?

**AI Involvement**: I discussed with Groq whether to use face recognition linking, receipt detection, or time-window correlation.

**AI suggested**: Time-window correlation (visitor in billing zone within 5 minutes of POS transaction). I agreed — it matches the problem specification exactly.

**What I implemented over AI suggestion**: The model suggested a 3-minute window. I used 5 minutes (as specified in the problem statement), and capped conversion rate at 1.0 to prevent edge-case overflow when multiple transactions happen close together.

---

## Technology Choices Summary

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Detection | YOLOv8n (Ultralytics) | Fast CPU inference, native ByteTrack, COCO-pretrained |
| Tracking | ByteTrack (built-in) | Best-in-class occlusion handling, low memory |
| Re-ID | Color histogram (384-dim) | No GPU required, effective for retail environments |
| Staff detection | HSV color + Groq Vision | Layered approach: fast + accurate |
| API | FastAPI + Pydantic v2 | Async, type-safe, auto OpenAPI docs |
| Storage | SQLite (WAL mode) | Zero external dependencies, concurrent reads |
| Logging | structlog JSON | Production-ready structured logs |
| Container | Docker Compose | Single `docker compose up` to start |
