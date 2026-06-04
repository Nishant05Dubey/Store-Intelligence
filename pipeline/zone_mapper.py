"""
zone_mapper.py — Map detected person bounding boxes to store zones.

Uses the store_layout.json zone definitions (relative bounding boxes 0.0-1.0)
to determine which zone a person's centroid falls into.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Zone:
    zone_id: str
    zone_name: str
    sku_zone: Optional[str]
    zone_type: str
    x1: float
    y1: float
    x2: float
    y2: float
    camera_ids: list[str]


class ZoneMapper:
    """
    Maps a (cx, cy) centroid in normalised [0,1] coordinates
    to the deepest matching zone.

    Zone bounding boxes in store_layout.json are defined as percentages
    of the camera frame (x1_pct, y1_pct, x2_pct, y2_pct).
    """

    def __init__(self, layout_path: str):
        self.zones: list[Zone] = []
        self._load(layout_path)

    def _load(self, path: str) -> None:
        with open(path) as f:
            layout = json.load(f)
        for z in layout.get("zones", []):
            bbox = z.get("bbox_pct", {})
            self.zones.append(
                Zone(
                    zone_id=z["zone_id"],
                    zone_name=z["zone_name"],
                    sku_zone=z.get("sku_zone"),
                    zone_type=z.get("type", "unknown"),
                    x1=float(bbox.get("x1", 0.0)),
                    y1=float(bbox.get("y1", 0.0)),
                    x2=float(bbox.get("x2", 1.0)),
                    y2=float(bbox.get("y2", 1.0)),
                    camera_ids=z.get("camera_ids", []),
                )
            )

    def get_zone(
        self, cx: float, cy: float, camera_id: Optional[str] = None
    ) -> Optional[Zone]:
        """
        Return the most specific zone containing the centroid (cx, cy).
        If camera_id is provided, filters to zones visible from that camera.
        'Most specific' = smallest area zone that contains the point.
        """
        candidates: list[Zone] = []
        for zone in self.zones:
            if camera_id and zone.camera_ids and camera_id not in zone.camera_ids:
                continue
            if zone.x1 <= cx <= zone.x2 and zone.y1 <= cy <= zone.y2:
                candidates.append(zone)

        if not candidates:
            return None

        # Return the smallest-area zone (most specific)
        return min(
            candidates,
            key=lambda z: (z.x2 - z.x1) * (z.y2 - z.y1),
        )

    def is_billing_zone(self, zone_id: Optional[str]) -> bool:
        """Check if a zone is a billing/cash counter zone."""
        if zone_id is None:
            return False
        for z in self.zones:
            if z.zone_id == zone_id and z.zone_type == "billing":
                return True
        return False

    def is_entry_zone(self, zone_id: Optional[str]) -> bool:
        if zone_id is None:
            return False
        for z in self.zones:
            if z.zone_id == zone_id and z.zone_type == "entry_exit":
                return True
        return False
