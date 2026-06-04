"""
detect.py — Main CCTV detection and tracking pipeline.

Processes video clips using YOLOv8 + ByteTrack and emits structured events.

Pipeline per frame:
  1. YOLOv8 detects persons (class 0) with bounding boxes + confidence
  2. ByteTrack assigns stable track_ids across frames
  3. PersonTracker does Re-ID (cross-camera + re-entry detection)
  4. StaffDetector classifies staff vs customer per person
  5. ZoneMapper determines which zone the person is in
  6. EventEmitter emits appropriate events based on state changes

Usage:
  python detect.py --video /path/to/CAM_1.mp4 --camera-id CAM_ENTRY_01 \\
                   --store-id STORE_BLR_001 --clip-start 2026-04-10T12:00:00Z
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import structlog

# Setup path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.emit import EventEmitter
from pipeline.staff_detector import classify_staff
from pipeline.tracker import PersonTracker
from pipeline.zone_mapper import ZoneMapper

logger = structlog.get_logger(__name__)

# Configuration
SKIP_FRAMES = 3          # Process every Nth frame (15fps → ~5fps effective)
DWELL_EMIT_INTERVAL_MS = 30_000  # Emit ZONE_DWELL every 30 seconds of continued presence
ENTRY_LINE_Y_PCT = 0.5   # Entry/exit detection line at 50% of frame height
DETECTION_CONF_THRESHOLD = 0.35  # Minimum YOLO confidence to consider
PERSON_CLASS_ID = 0      # COCO class ID for person

# Billing zone parameters
BILLING_QUEUE_DEPTH_WINDOW = 5  # Number of persons in billing zone = queue depth
MIN_BILLING_DWELL_MS_FOR_ABANDON = 10_000  # Must wait at least 10s to count as abandonment


def run_detection(
    video_path: str,
    camera_id: str,
    store_id: str,
    clip_start_utc: datetime,
    layout_path: str,
    api_url: str,
    use_groq: bool = True,
    max_frames: Optional[int] = None,
    output_jsonl: Optional[str] = None,
) -> int:
    """
    Main detection loop for a single video clip.

    Args:
        video_path: Path to the .mp4 file
        camera_id: Camera identifier from store_layout.json
        store_id: Store identifier
        clip_start_utc: UTC datetime when this clip started
        layout_path: Path to store_layout.json
        api_url: Base URL of the Intelligence API
        use_groq: Whether to use Groq Vision for staff detection
        max_frames: Limit processing to N frames (for testing)
        output_jsonl: If provided, also write events to this file

    Returns:
        Total events emitted
    """
    # Load YOLOv8 model (auto-detects GPU vs CPU)
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("detect.yolo_not_installed")
        sys.exit(1)

    logger.info(
        "detect.start",
        video=video_path,
        camera_id=camera_id,
        store_id=store_id,
        clip_start=clip_start_utc.isoformat(),
    )

    # Initialize components
    model = YOLO("yolov8n.pt")  # Nano model for speed; use yolov8s.pt for accuracy
    zone_mapper = ZoneMapper(layout_path)
    tracker = PersonTracker(store_id)
    emitter = EventEmitter(
        store_id=store_id,
        camera_id=camera_id,
        clip_start_utc=clip_start_utc,
        fps=1.0,  # Will be set after cap opens
        api_url=api_url,
    )

    # Groq client (optional)
    groq_client = None
    if use_groq and os.getenv("GROQ_API_KEY"):
        try:
            from groq import Groq
            groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
            logger.info("detect.groq_enabled")
        except ImportError:
            logger.warning("detect.groq_not_installed")

    # JSONL output file
    jsonl_file = None
    if output_jsonl:
        jsonl_file = open(output_jsonl, "a", encoding="utf-8")

    # State tracking per visitor
    # visitor_id → {zone_id, zone_enter_frame, last_dwell_frame, in_billing, billing_enter_frame}
    visitor_state: dict[str, dict] = {}
    # Direction tracking for entry/exit: visitor_id → [y_positions]
    direction_buffer: dict[str, list[float]] = {}
    # Currently active visitors (visible this frame)
    known_entrants: set[str] = set()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("detect.video_open_failed", path=video_path)
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    emitter.fps = fps

    logger.info(
        "detect.video_info",
        fps=fps,
        total_frames=total_frames,
        duration_min=round(total_frames / fps / 60, 1),
    )

    frame_number = 0
    total_events = 0
    prev_frame_visitors: set[str] = set()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            frame_number += 1
            if max_frames and frame_number > max_frames:
                break

            # Process every Nth frame
            if frame_number % SKIP_FRAMES != 0:
                continue

            frame_h, frame_w = frame.shape[:2]

            # ---- YOLOv8 detection + ByteTrack ----
            results = model.track(
                frame,
                persist=True,
                classes=[PERSON_CLASS_ID],
                conf=DETECTION_CONF_THRESHOLD,
                verbose=False,
                tracker="bytetrack.yaml",
            )

            current_frame_visitors: set[str] = set()

            if results and results[0].boxes is not None:
                boxes = results[0].boxes
                for box in boxes:
                    if box.id is None:
                        continue  # No track ID yet (ByteTrack warmup)

                    track_id = int(box.id.item())
                    conf = float(box.conf.item())
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

                    # Centroid (normalised)
                    cx = ((x1 + x2) / 2) / frame_w
                    cy = ((y1 + y2) / 2) / frame_h

                    # ---- Re-ID ----
                    visitor_id, is_reentry = tracker.get_or_assign_visitor_id(
                        frame, (x1, y1, x2, y2), track_id, camera_id
                    )
                    current_frame_visitors.add(visitor_id)

                    # ---- Staff detection ----
                    is_staff, staff_conf = classify_staff(
                        frame,
                        (x1, y1, x2, y2),
                        groq_client=groq_client,
                        use_groq=use_groq,
                    )
                    # Blend detection confidence with staff confidence
                    effective_conf = (conf + staff_conf) / 2

                    # Draw bounding box
                    color = (255, 0, 0) if is_staff else (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"ID: {visitor_id[:8]} {'[STAFF]' if is_staff else ''}", (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # ---- Direction tracking (for entry/exit detection) ----
                    if visitor_id not in direction_buffer:
                        direction_buffer[visitor_id] = []
                    direction_buffer[visitor_id].append(cy)
                    # Keep last 10 positions
                    direction_buffer[visitor_id] = direction_buffer[visitor_id][-10:]

                    # ---- Entry / Exit detection ----
                    if visitor_id not in known_entrants:
                        # First time seeing this visitor
                        direction = _determine_direction(
                            direction_buffer.get(visitor_id, [])
                        )
                        if direction == "inbound" or len(direction_buffer.get(visitor_id, [])) < 3:
                            # New visitor entering
                            known_entrants.add(visitor_id)
                            visitor_state[visitor_id] = {
                                "zone_id": None,
                                "zone_enter_frame": None,
                                "last_dwell_frame": None,
                                "in_billing": False,
                                "billing_enter_frame": None,
                                "billing_joined_queue": False,
                            }
                            _emit_and_log(
                                emitter,
                                jsonl_file,
                                "entry",
                                visitor_id=visitor_id,
                                frame_number=frame_number,
                                confidence=effective_conf,
                                is_staff=is_staff,
                                is_reentry=is_reentry,
                            )
                            total_events += 1
                    elif is_reentry and visitor_id not in current_frame_visitors:
                        # Reentry case (handled by tracker)
                        emitter.emit_entry(
                            visitor_id, frame_number, effective_conf, is_staff, is_reentry=True
                        )
                        total_events += 1

                    # ---- Zone mapping ----
                    if visitor_id in visitor_state:
                        state = visitor_state[visitor_id]
                        current_zone = zone_mapper.get_zone(cx, cy, camera_id)
                        current_zone_id = current_zone.zone_id if current_zone else None
                        prev_zone_id = state.get("zone_id")

                        if current_zone_id != prev_zone_id:
                            # Zone transition
                            if prev_zone_id is not None:
                                # Zone exit
                                dwell_ms = _frames_to_ms(
                                    frame_number - (state.get("zone_enter_frame") or frame_number),
                                    fps,
                                )
                                emitter.emit_zone_exit(
                                    visitor_id, frame_number, prev_zone_id,
                                    dwell_ms, effective_conf, is_staff
                                )
                                total_events += 1

                                # Billing abandonment check
                                if zone_mapper.is_billing_zone(prev_zone_id) and state.get("in_billing"):
                                    if dwell_ms >= MIN_BILLING_DWELL_MS_FOR_ABANDON:
                                        emitter.emit_billing_queue_abandon(
                                            visitor_id, frame_number, dwell_ms, effective_conf, is_staff
                                        )
                                        total_events += 1
                                    state["in_billing"] = False
                                    state["billing_enter_frame"] = None
                                    state["billing_joined_queue"] = False

                            if current_zone_id is not None:
                                # Zone enter
                                sku_zone = current_zone.sku_zone if current_zone else None
                                emitter.emit_zone_enter(
                                    visitor_id, frame_number, current_zone_id,
                                    sku_zone, effective_conf, is_staff
                                )
                                total_events += 1

                                # Billing queue join
                                if zone_mapper.is_billing_zone(current_zone_id):
                                    billing_occupants = _count_billing_occupants(
                                        visitor_state, zone_mapper
                                    )
                                    if billing_occupants > 0:
                                        emitter.emit_billing_queue_join(
                                            visitor_id, frame_number,
                                            queue_depth=billing_occupants,
                                            confidence=effective_conf,
                                            is_staff=is_staff,
                                        )
                                        total_events += 1
                                    state["in_billing"] = True
                                    state["billing_enter_frame"] = frame_number
                                    state["billing_joined_queue"] = True

                            state["zone_id"] = current_zone_id
                            state["zone_enter_frame"] = frame_number
                            state["last_dwell_frame"] = frame_number

                        else:
                            # Same zone — check for ZONE_DWELL emission
                            last_dwell = state.get("last_dwell_frame") or frame_number
                            dwell_so_far_ms = _frames_to_ms(frame_number - last_dwell, fps)
                            if dwell_so_far_ms >= DWELL_EMIT_INTERVAL_MS and current_zone_id:
                                total_dwell_ms = _frames_to_ms(
                                    frame_number - (state.get("zone_enter_frame") or frame_number),
                                    fps,
                                )
                                emitter.emit_zone_dwell(
                                    visitor_id, frame_number, current_zone_id,
                                    total_dwell_ms, effective_conf, is_staff
                                )
                                total_events += 1
                                state["last_dwell_frame"] = frame_number

            # ---- Exit detection: visitors no longer visible ----
            vanished = prev_frame_visitors - current_frame_visitors
            for visitor_id in vanished:
                if visitor_id not in visitor_state:
                    continue
                state = visitor_state.get(visitor_id, {})

                # Allow a grace period before treating as exit
                # (handles brief occlusions)
                # For simplicity here: if vanished for >1s, emit EXIT
                # Full implementation uses tracker's last_seen timestamp
                direction = _determine_direction(
                    direction_buffer.get(visitor_id, [])
                )
                if direction == "outbound":
                    tracker.mark_exited(visitor_id)
                    if visitor_id in known_entrants:
                        known_entrants.discard(visitor_id)
                        emitter.emit_exit(
                            visitor_id, frame_number, confidence=0.7, is_staff=False
                        )
                        total_events += 1

            # Display the frame
            # Save the frame to the dashboard directory to serve as a live feed
            display_frame = cv2.resize(frame, (1280, 720)) if frame.shape[1] > 1280 else frame
            cv2.imwrite(f"dashboard/feed_{camera_id}.jpg", display_frame)

            prev_frame_visitors = current_frame_visitors

            # Periodic cleanup
            if frame_number % (int(fps) * 60) == 0:  # Every minute
                tracker.cleanup_stale()

        # Flush remaining events
        emitter.flush()

    finally:
        cap.release()
        if jsonl_file:
            jsonl_file.close()

    logger.info(
        "detect.complete",
        frames_processed=frame_number,
        total_events=total_events,
    )
    return total_events


def _determine_direction(y_positions: list[float]) -> str:
    """
    Determine movement direction from recent Y positions.
    In most CCTV entry cameras, moving DOWN (increasing Y) = entering.
    Returns 'inbound', 'outbound', or 'unknown'.
    """
    if len(y_positions) < 3:
        return "unknown"
    delta = y_positions[-1] - y_positions[0]
    if delta > 0.05:
        return "inbound"   # Moving down in frame → entering
    elif delta < -0.05:
        return "outbound"  # Moving up in frame → exiting
    return "unknown"


def _frames_to_ms(frames: int, fps: float) -> int:
    """Convert frame count to milliseconds."""
    if fps <= 0:
        return 0
    return int((frames / fps) * 1000)


def _count_billing_occupants(
    visitor_state: dict, zone_mapper: ZoneMapper
) -> int:
    """Count how many visitors are currently in the billing zone."""
    count = 0
    for state in visitor_state.values():
        zone_id = state.get("zone_id")
        if zone_id and zone_mapper.is_billing_zone(zone_id):
            count += 1
    return count


def _emit_and_log(emitter: EventEmitter, jsonl_file, event_kind: str, **kwargs) -> None:
    """Emit an entry event and optionally log to JSONL file."""
    emitter.emit_entry(**{k: v for k, v in kwargs.items() if k in [
        "visitor_id", "frame_number", "confidence", "is_staff", "is_reentry"
    ]})
    if jsonl_file:
        import json
        entry = {
            "event_kind": event_kind,
            "visitor_id": kwargs.get("visitor_id"),
            "frame_number": kwargs.get("frame_number"),
            "confidence": kwargs.get("confidence"),
            "is_staff": kwargs.get("is_staff"),
        }
        jsonl_file.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Store Intelligence CCTV Detection Pipeline"
    )
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--camera-id", required=True, help="Camera ID from store_layout")
    parser.add_argument("--store-id", required=True, help="Store ID")
    parser.add_argument(
        "--clip-start",
        required=True,
        help="Clip start UTC datetime (ISO-8601, e.g. 2026-04-10T12:00:00Z)",
    )
    parser.add_argument(
        "--layout",
        default="data/store_layout.json",
        help="Path to store_layout.json",
    )
    parser.add_argument(
        "--api-url",
        default=os.getenv("API_BASE_URL", "http://localhost:8000"),
        help="API base URL",
    )
    parser.add_argument(
        "--no-groq",
        action="store_true",
        help="Disable Groq Vision staff detection",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limit to N frames (for testing)",
    )
    parser.add_argument(
        "--output-jsonl",
        default=None,
        help="Also write events to this JSONL file",
    )
    return parser.parse_args()


if __name__ == "__main__":
    import logging
    import structlog

    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        logger_factory=structlog.PrintLoggerFactory(),
    )

    args = _parse_args()

    clip_start = datetime.fromisoformat(args.clip_start.replace("Z", "+00:00"))

    run_detection(
        video_path=args.video,
        camera_id=args.camera_id,
        store_id=args.store_id,
        clip_start_utc=clip_start,
        layout_path=args.layout,
        api_url=args.api_url,
        use_groq=not args.no_groq,
        max_frames=args.max_frames,
        output_jsonl=args.output_jsonl,
    )
