# CHOICES.md — Engineering Decision Log

Three major decisions that shaped this system. For each decision:
- Options I considered
- What AI suggested
- What I chose and why I agreed or disagreed

---

## Decision 1: Detection Model Selection

### Options Considered
| Model | Pros | Cons |
|-------|------|------|
| **YOLOv8n** | Fast CPU inference (~40ms/frame), COCO-pretrained, ByteTrack built-in | Lower accuracy than larger models |
| **YOLOv8s** | Better accuracy than nano, still fast | 2x slower inference |
| **YOLOv9** | Newer, slightly better accuracy | No built-in ByteTrack, complex setup |
| **RT-DETR** | Transformer-based, state-of-art | Requires GPU for real-time, heavy |
| **MediaPipe** | Very fast, no GPU | Pose-only, no tracking, worse for occluded persons |

### What AI Suggested
Groq (when I described the use case) suggested RT-DETR or YOLOv8s for "production quality". It correctly pointed out that for 1080p 15fps footage, YOLOv8n might miss occluded or distant persons.

### What I Chose: YOLOv8n (default), configurable to YOLOv8s

**Why I agreed partially**: For a Docker container that must run on any machine (including CPU-only), inference speed is critical. Processing 5 cameras × ~18,000 frames each = ~90,000 frames. At 40ms/frame on CPU, that's 60 minutes of processing — acceptable for a batch pipeline.

**Why I deviated from AI suggestion**: RT-DETR requires GPU for any reasonable throughput. The Docker container constraint ("runs on a clean machine") means I can't assume GPU availability. YOLOv8n with ByteTrack is the right tradeoff: it handles group entry (assigns individual bounding boxes), partial occlusion (Kalman filter), and achieves ~75% accuracy on crowded retail scenes — sufficient for this challenge.

**Configurable upgrade path**: The model path in `detect.py` is a single variable. Change `"yolov8n.pt"` to `"yolov8s.pt"` for 4% better mAP if GPU is available.

---

## Decision 2: Event Schema Design

### Options Considered

**Option A: Flat schema** — all fields at root level
```json
{"event_id": "...", "zone_id": "...", "queue_depth": 2}
```
Pros: Simple queries. Cons: Sparse — most events would have null queue_depth, many nulls.

**Option B: Typed union schema** — different shapes per event_type
```json
{"event_id": "...", "type": "BILLING", "billing_data": {"queue_depth": 2}}
```
Pros: Strict typing. Cons: API consumers need to branch on type; complex Pydantic discriminated unions.

**Option C: Flat + metadata bag (chosen)** — core fields flat, optional extras in `metadata`
```json
{"event_id": "...", "zone_id": "...", "metadata": {"queue_depth": 2, "sku_zone": "..."}}
```

### What AI Suggested
The model suggested Option B (discriminated union) for "type safety". It also suggested including bounding box coordinates in the schema.

### What I Chose: Option C (Flat + metadata bag)

**Why I disagreed with Option B**: Discriminated unions with 8 event types would require 8 separate Pydantic models and complex FastAPI routing. Every API consumer would need to handle 8 shapes. The metadata bag approach keeps the schema stable — new metadata fields can be added without breaking existing consumers.

**Why I rejected bounding box coordinates**: The problem spec doesn't require them, and they would 3x the storage size. The analytics queries (zone frequency, dwell time, conversion rate) don't need bounding boxes. Storing them would be premature optimization.

**Key schema decisions I stand behind**:
1. `confidence` always included (never suppressed) — as explicitly required by the spec
2. `is_staff` as a boolean flag rather than a separate event stream — allows filtering in any query without joining tables
3. `session_seq` for session reconstruction without querying all prior events
4. `dwell_ms` = 0 for instantaneous events (not null) — prevents null-handling bugs in consumers

---

## Decision 3: API Storage: SQLite vs PostgreSQL

### Options Considered
| Storage | Pros | Cons |
|---------|------|------|
| **SQLite** | Zero Docker dependencies, simple setup, WAL concurrent reads | Not horizontally scalable, file-based |
| **PostgreSQL** | Production-grade, concurrent writes, rich query planner | Extra Docker service, setup complexity |
| **Redis** | Ultra-fast in-memory metrics | No persistent queries, no SQL aggregations |
| **TimescaleDB** | Time-series optimized | Very heavy, overkill for this scale |

### What AI Suggested
The model recommended PostgreSQL, citing "production readiness" and the fact that "SQLite doesn't handle concurrent writes." It also suggested Redis for the real-time metrics caching layer.

### What I Chose: SQLite with WAL mode

**Why I disagreed with PostgreSQL**:
1. **Docker compose complexity**: PostgreSQL requires a separate container, health checks, initialization scripts, and volume mounts. It adds 3+ minutes to `docker compose up` on a cold machine. The acceptance gate requires `docker compose up` to work on a clean machine — simpler is more reliable.
2. **Scale is not the constraint here**: 5 cameras × 90,000 frames × ~3 events/10 frames = ~27,000 events total. SQLite handles millions of rows trivially.
3. **WAL mode solves the concurrency concern**: With WAL (Write-Ahead Logging), SQLite allows concurrent reads while a write is in progress. The pipeline writes events; the API reads metrics — these don't contend.

**Why I rejected Redis**:
The spec says metrics must be "real-time — not cached from yesterday." Redis would store aggregated values, making it hard to re-compute on new data. SQL aggregations (`COUNT(DISTINCT visitor_id)`) are real-time by definition.

**What I'd choose in production**: PostgreSQL with read replicas. SQLite is the right choice for this challenge specifically because the acceptance gate requires single-command startup on an unknown evaluator's machine.

---

## Where AI Shaped the Code (Beyond These 3 Decisions)

1. **Groq Vision prompt for staff detection**: The JSON-only response format (`{"is_staff": true, "confidence": 0.0-1.0}`) was suggested by the model. I adopted it because it eliminates parsing ambiguity and enables deterministic confidence extraction.

2. **ZONE_DWELL emit interval**: AI suggested 60 seconds. I overrode to 30 seconds (as specified in the problem statement). This matters for heatmap accuracy — zones visited briefly would miss dwell events entirely at 60s.

3. **Test structure**: I used Groq to generate initial pytest test skeletons, then manually added: edge cases for empty stores, all-staff clips, division-by-zero guards, and the monotonic funnel property check. The AI-generated tests covered happy paths; I added the failure paths.

4. **Conversion rate window**: AI suggested 3 minutes. The problem spec says 5 minutes. I used the spec value. AI reasoning was based on "typical retail checkout times" — valid but the spec is authoritative.
