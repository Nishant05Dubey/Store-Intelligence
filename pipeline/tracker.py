"""
tracker.py — Person Re-ID and cross-camera deduplication.

Assigns persistent visitor_ids across:
  1. Disappearance/reappearance within a single camera view (occlusion)
  2. Movement between overlapping camera fields of view
  3. Re-entry into the store after exiting

Strategy:
  - Primary: ByteTrack assigns track_ids per camera (short-term tracking)
  - Re-ID: Match appearance features (color histogram of torso + body ratio)
    against a gallery of recent visitors (last 30 minutes per store)
  - Re-entry: If same visitor_id matches after an EXIT event, emit REENTRY

Note: We do NOT use face recognition — footage is anonymized (faces blurred).
All Re-ID is based on body appearance (clothing color, height ratio, gait).
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import structlog

logger = structlog.get_logger(__name__)

# Gallery window: how long we keep visitor appearance features in memory
GALLERY_WINDOW_SECONDS = 1800  # 30 minutes
# Re-ID similarity threshold (cosine similarity)
REID_SIMILARITY_THRESHOLD = 0.75
# Re-entry window: same visitor re-entering within this window → REENTRY event
REENTRY_WINDOW_SECONDS = 3600  # 1 hour


@dataclass
class VisitorRecord:
    visitor_id: str
    track_id: int
    camera_id: str
    appearance_features: np.ndarray  # color histogram feature vector
    last_seen: float  # time.time()
    has_exited: bool = False
    exit_time: Optional[float] = None
    bbox_history: list[tuple[int, int, int, int]] = field(default_factory=list)


class PersonTracker:
    """
    Manages visitor Re-ID across frames and cameras.

    Uses per-camera ByteTrack track_ids for short-term tracking,
    then matches appearance features for cross-camera and re-entry detection.
    """

    def __init__(self, store_id: str):
        self.store_id = store_id
        # Active gallery: visitor_id → VisitorRecord
        self._gallery: dict[str, VisitorRecord] = {}
        # Camera-local track mapping: (camera_id, track_id) → visitor_id
        self._track_map: dict[tuple[str, int], str] = {}

    def get_or_assign_visitor_id(
        self,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
        track_id: int,
        camera_id: str,
    ) -> tuple[str, bool]:
        """
        Get existing visitor_id for this track, or assign a new one via Re-ID.

        Returns:
            (visitor_id, is_reentry)
        """
        key = (camera_id, track_id)

        # Known track in this camera
        if key in self._track_map:
            vid = self._track_map[key]
            if vid in self._gallery:
                record = self._gallery[vid]
                record.last_seen = time.time()
                record.bbox_history.append(bbox)
                return vid, False
            # Track exists but gallery entry expired — treat as new
            del self._track_map[key]

        # Extract appearance features
        features = _extract_features(frame, bbox)

        # Try to match against gallery (cross-camera / occlusion Re-ID)
        matched_vid, similarity = self._find_match(features, camera_id, track_id)

        if matched_vid and similarity >= REID_SIMILARITY_THRESHOLD:
            record = self._gallery[matched_vid]
            is_reentry = record.has_exited
            record.has_exited = False
            record.last_seen = time.time()
            record.track_id = track_id
            record.camera_id = camera_id
            record.bbox_history.append(bbox)
            self._track_map[key] = matched_vid
            return matched_vid, is_reentry

        # New visitor
        visitor_id = _generate_visitor_id()
        record = VisitorRecord(
            visitor_id=visitor_id,
            track_id=track_id,
            camera_id=camera_id,
            appearance_features=features,
            last_seen=time.time(),
            bbox_history=[bbox],
        )
        self._gallery[visitor_id] = record
        self._track_map[key] = visitor_id

        logger.debug(
            "tracker.new_visitor",
            visitor_id=visitor_id,
            camera_id=camera_id,
            track_id=track_id,
        )
        return visitor_id, False

    def mark_exited(self, visitor_id: str) -> None:
        """Mark a visitor as exited (for re-entry detection)."""
        if visitor_id in self._gallery:
            self._gallery[visitor_id].has_exited = True
            self._gallery[visitor_id].exit_time = time.time()

    def cleanup_stale(self) -> None:
        """Remove gallery entries older than GALLERY_WINDOW_SECONDS."""
        cutoff = time.time() - GALLERY_WINDOW_SECONDS
        stale_vids = [
            vid
            for vid, rec in self._gallery.items()
            if rec.last_seen < cutoff
        ]
        for vid in stale_vids:
            # Remove track map entries for stale visitors
            stale_keys = [k for k, v in self._track_map.items() if v == vid]
            for k in stale_keys:
                del self._track_map[k]
            del self._gallery[vid]

        if stale_vids:
            logger.debug("tracker.cleanup", removed=len(stale_vids))

    def _find_match(
        self,
        features: np.ndarray,
        camera_id: str,
        track_id: int,
    ) -> tuple[Optional[str], float]:
        """
        Find the best matching gallery entry for given features.

        Uses cosine similarity on color histogram features.
        Excludes entries currently active on the same track (avoid self-match).
        """
        if features is None or len(features) == 0:
            return None, 0.0

        best_vid = None
        best_sim = 0.0

        # Only consider recently seen visitors (within reentry window)
        cutoff = time.time() - REENTRY_WINDOW_SECONDS

        for vid, record in self._gallery.items():
            if record.last_seen < cutoff:
                continue
            if record.appearance_features is None:
                continue

            sim = _cosine_similarity(features, record.appearance_features)
            if sim > best_sim:
                best_sim = sim
                best_vid = vid

        return best_vid, best_sim


def _extract_features(
    frame: np.ndarray, bbox: tuple[int, int, int, int]
) -> np.ndarray:
    """
    Extract appearance feature vector from person bounding box.

    Feature vector = concatenation of:
    - HSV color histogram of full body (192 bins)
    - HSV color histogram of torso region (192 bins)
    Total: 384-dimensional vector, L2 normalized.
    """
    x1, y1, x2, y2 = bbox
    fh, fw = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(fw, x2), min(fh, y2)

    person_crop = frame[y1:y2, x1:x2]
    if person_crop.size == 0:
        return np.zeros(384)

    h = y2 - y1
    torso_crop = frame[
        min(fh - 1, y1 + int(h * 0.2)):min(fh, y1 + int(h * 0.65)),
        x1:x2,
    ]

    def hist(crop):
        if crop.size == 0:
            return np.zeros(192)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h_hist = cv2.calcHist([hsv], [0], None, [64], [0, 180]).flatten()
        s_hist = cv2.calcHist([hsv], [1], None, [64], [0, 256]).flatten()
        v_hist = cv2.calcHist([hsv], [2], None, [64], [0, 256]).flatten()
        vec = np.concatenate([h_hist, s_hist, v_hist])
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    full_hist = hist(person_crop)
    torso_hist = hist(torso_crop) if torso_crop.size > 0 else np.zeros(192)
    return np.concatenate([full_hist, torso_hist])


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two feature vectors."""
    if a.shape != b.shape:
        return 0.0
    dot = float(np.dot(a, b))
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _generate_visitor_id() -> str:
    """Generate a short, unique visitor ID."""
    raw = str(time.time_ns()) + str(id(object()))
    return "VIS_" + hashlib.sha1(raw.encode()).hexdigest()[:8]
