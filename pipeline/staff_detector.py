"""
staff_detector.py — Classify whether a detected person is store staff.

Strategy (layered):
1. Color histogram on torso region of the bounding box
   - Purplle stores use purple/violet uniforms (HSV hue range 120-160)
   - High confidence if dominant color matches uniform profile
2. Groq Vision LLM fallback for ambiguous cases
   - Sends a cropped frame snippet to llama-3.2-11b-vision-preview
   - Prompt asks: "Is this person wearing a store uniform?"
   - Used only when color confidence is below threshold

This ensures we don't make blanket assumptions —
each person's classification is based on actual visual features.
"""

from __future__ import annotations

import base64
import json
import os
from typing import Optional

import cv2
import numpy as np
import structlog

logger = structlog.get_logger(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_VISION_MODEL = "llama-3.2-11b-vision-preview"
COLOR_CONFIDENCE_THRESHOLD = 0.6  # below this, use Groq fallback
STAFF_COLOR_RATIO_THRESHOLD = 0.25  # fraction of pixels matching uniform color


# Purplle uniform HSV range (purple/violet)
STAFF_HSV_RANGES = [
    # Purple/Violet range
    {"h_low": 120, "h_high": 160, "s_low": 50, "s_high": 255, "v_low": 50, "v_high": 255},
    # Dark purple (dim lighting)
    {"h_low": 110, "h_high": 170, "s_low": 30, "s_high": 255, "v_low": 30, "v_high": 200},
]


def classify_staff(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    groq_client=None,
    use_groq: bool = True,
) -> tuple[bool, float]:
    """
    Classify whether the person in the bounding box is store staff.

    Args:
        frame: Full video frame (BGR numpy array)
        bbox: Bounding box (x1, y1, x2, y2) in pixel coordinates
        groq_client: Initialized Groq client (optional)
        use_groq: Whether to use Groq fallback for uncertain cases

    Returns:
        (is_staff, confidence) tuple
    """
    x1, y1, x2, y2 = bbox
    h = y2 - y1
    w = x2 - x1

    if h <= 0 or w <= 0:
        return False, 0.0

    # Extract torso region (middle 40% of the bounding box height)
    torso_y1 = y1 + int(h * 0.25)
    torso_y2 = y1 + int(h * 0.65)
    torso_x1 = x1 + int(w * 0.1)
    torso_x2 = x2 - int(w * 0.1)

    # Guard against out-of-bounds
    frame_h, frame_w = frame.shape[:2]
    torso_y1 = max(0, min(torso_y1, frame_h - 1))
    torso_y2 = max(0, min(torso_y2, frame_h - 1))
    torso_x1 = max(0, min(torso_x1, frame_w - 1))
    torso_x2 = max(0, min(torso_x2, frame_w - 1))

    torso_crop = frame[torso_y1:torso_y2, torso_x1:torso_x2]
    if torso_crop.size == 0:
        return False, 0.0

    # Color histogram approach
    color_ratio, color_confidence = _color_histogram_check(torso_crop)
    is_staff_color = color_ratio >= STAFF_COLOR_RATIO_THRESHOLD

    if color_confidence >= COLOR_CONFIDENCE_THRESHOLD:
        return is_staff_color, color_confidence

    # Groq Vision fallback for uncertain cases
    if use_groq and groq_client and GROQ_API_KEY:
        try:
            groq_result, groq_confidence = _groq_staff_classify(
                frame, bbox, groq_client
            )
            # Combine color and groq signals
            combined_confidence = (color_confidence + groq_confidence) / 2
            return groq_result, combined_confidence
        except Exception as exc:
            logger.warning(
                "staff_detector.groq_fallback_failed",
                error=str(exc),
            )

    # Default to color-based result with lower confidence
    return is_staff_color, color_confidence


def _color_histogram_check(
    torso_crop: np.ndarray,
) -> tuple[float, float]:
    """
    Check if the torso color matches the Purplle staff uniform (purple/violet).

    Returns:
        (staff_color_ratio, confidence)
        - staff_color_ratio: fraction of pixels matching uniform color
        - confidence: how certain we are about the color classification
    """
    hsv = cv2.cvtColor(torso_crop, cv2.COLOR_BGR2HSV)
    total_pixels = hsv.shape[0] * hsv.shape[1]

    if total_pixels == 0:
        return 0.0, 0.0

    # Count pixels matching staff uniform HSV ranges
    staff_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for hsv_range in STAFF_HSV_RANGES:
        lower = np.array([
            hsv_range["h_low"],
            hsv_range["s_low"],
            hsv_range["v_low"],
        ])
        upper = np.array([
            hsv_range["h_high"],
            hsv_range["s_high"],
            hsv_range["v_high"],
        ])
        mask = cv2.inRange(hsv, lower, upper)
        staff_mask = cv2.bitwise_or(staff_mask, mask)

    staff_pixels = int(np.sum(staff_mask > 0))
    staff_ratio = staff_pixels / total_pixels

    # Confidence is higher when ratio is clearly above or below threshold
    # If ratio is near 0.25 threshold, confidence is low
    distance_from_threshold = abs(staff_ratio - STAFF_COLOR_RATIO_THRESHOLD)
    confidence = min(distance_from_threshold * 4, 1.0)  # scale to [0, 1]

    return staff_ratio, confidence


def _groq_staff_classify(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    groq_client,
) -> tuple[bool, float]:
    """
    Use Groq Vision API (llama-3.2-11b-vision-preview) to classify staff.

    Sends a cropped image of the person to the model with a targeted prompt.
    Documents the prompt in DESIGN.md as an AI-assisted decision.
    """
    x1, y1, x2, y2 = bbox
    # Add 20px padding around person
    pad = 20
    frame_h, frame_w = frame.shape[:2]
    crop = frame[
        max(0, y1 - pad): min(frame_h, y2 + pad),
        max(0, x1 - pad): min(frame_w, x2 + pad),
    ]

    if crop.size == 0:
        return False, 0.0

    # Encode to JPEG base64
    _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 75])
    img_b64 = base64.b64encode(buf.tobytes()).decode()

    # Prompt engineered to detect staff uniforms in Indian beauty stores
    # AI-Assisted Decision: Used Groq llama vision to handle ambiguous uniform colors
    prompt = (
        "You are analyzing a CCTV frame from a Purplle beauty retail store in India. "
        "The image shows a person detected by a computer vision system. "
        "Purplle store staff wear distinctive purple/violet uniforms or aprons. "
        "Answer ONLY with a JSON object: "
        "{\"is_staff\": true/false, \"confidence\": 0.0-1.0, \"reasoning\": \"brief\"} "
        "Base your answer on visible clothing color and style. "
        "If the image is blurry or the person's face is blurred (anonymized), "
        "focus only on clothing."
    )

    response = groq_client.chat.completions.create(
        model=GROQ_VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_b64}"
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        max_tokens=150,
        temperature=0.1,  # Low temp for deterministic classification
    )

    content = response.choices[0].message.content.strip()

    # Parse JSON response
    try:
        # Extract JSON from response (model sometimes adds text around it)
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            result = json.loads(content[start:end])
            return bool(result.get("is_staff", False)), float(
                result.get("confidence", 0.5)
            )
    except json.JSONDecodeError:
        pass

    # Fallback: look for keywords
    lower = content.lower()
    is_staff = "true" in lower or "staff" in lower or "uniform" in lower
    return is_staff, 0.5
