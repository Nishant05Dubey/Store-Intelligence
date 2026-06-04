#!/bin/bash
# run.sh — Process all CCTV clips and feed events into the Store Intelligence API
#
# Usage: ./run.sh [OPTIONS]
# Options:
#   --no-groq       Disable Groq Vision staff detection (faster, less accurate)
#   --max-frames N  Process only first N frames per clip (for testing)
#   --dry-run       Write events to JSONL only, don't POST to API
#
# Environment variables:
#   GROQ_API_KEY    Required for Groq Vision staff detection
#   API_BASE_URL    API endpoint (default: http://localhost:8000)
#   FOOTAGE_DIR     Directory containing CAM*.mp4 files

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

API_URL="${API_BASE_URL:-http://localhost:8000}"
FOOTAGE_DIR="${FOOTAGE_DIR:-$PROJECT_DIR/../CCTV Footage}"
LAYOUT_PATH="$PROJECT_DIR/data/store_layout.json"
MAX_FRAMES="${MAX_FRAMES:-}"
NO_GROQ=""
DRY_RUN=""

# Parse args
for arg in "$@"; do
  case $arg in
    --no-groq) NO_GROQ="--no-groq" ;;
    --dry-run) DRY_RUN="1" ;;
    --max-frames=*) MAX_FRAMES="${arg#*=}" ;;
  esac
done

echo "=== Store Intelligence Detection Pipeline ==="
echo "API URL:     $API_URL"
echo "Footage dir: $FOOTAGE_DIR"
echo "Layout:      $LAYOUT_PATH"
echo ""

# Wait for API to be available
echo "Waiting for API..."
for i in {1..30}; do
  if curl -sf "$API_URL/health" > /dev/null 2>&1; then
    echo "API is up!"
    break
  fi
  sleep 2
  if [ $i -eq 30 ]; then
    echo "ERROR: API not available after 60s"
    exit 1
  fi
done

# Camera configuration
# Maps camera files to camera IDs and clip start times
# Clip start time: Brigade Bangalore store opens ~11:00 AM on April 10, 2026
STORE_ID="STORE_BLR_001"
CLIP_START="2026-04-10T11:00:00Z"

declare -A CAMERA_MAP
CAMERA_MAP["CAM 1.mp4"]="CAM_ENTRY_01"
CAMERA_MAP["CAM 2.mp4"]="CAM_FLOOR_01"
CAMERA_MAP["CAM 3.mp4"]="CAM_BILLING_01"
CAMERA_MAP["CAM 4.mp4"]="CAM_FLOOR_02"
CAMERA_MAP["CAM 5.mp4"]="CAM_FLOOR_03"

TOTAL_EVENTS=0
PROCESSED=0
FAILED=0

for cam_file in "CAM 1.mp4" "CAM 2.mp4" "CAM 3.mp4" "CAM 4.mp4" "CAM 5.mp4"; do
  VIDEO_PATH="$FOOTAGE_DIR/$cam_file"
  CAMERA_ID="${CAMERA_MAP[$cam_file]}"

  if [ ! -f "$VIDEO_PATH" ]; then
    echo "WARNING: Video not found: $VIDEO_PATH (skipping)"
    continue
  fi

  echo ""
  echo "--- Processing: $cam_file → $CAMERA_ID ---"

  # Build command
  CMD="python $SCRIPT_DIR/detect.py \
    --video \"$VIDEO_PATH\" \
    --camera-id $CAMERA_ID \
    --store-id $STORE_ID \
    --clip-start $CLIP_START \
    --layout $LAYOUT_PATH \
    --api-url $API_URL \
    $NO_GROQ"

  if [ -n "$MAX_FRAMES" ]; then
    CMD="$CMD --max-frames $MAX_FRAMES"
  fi

  if [ -n "$DRY_RUN" ]; then
    OUTPUT_JSONL="$PROJECT_DIR/data/${CAMERA_ID}_events.jsonl"
    CMD="$CMD --output-jsonl $OUTPUT_JSONL"
    echo "DRY RUN: Writing events to $OUTPUT_JSONL"
  fi

  if eval $CMD; then
    echo "✓ $cam_file processed successfully"
    PROCESSED=$((PROCESSED + 1))
  else
    echo "✗ $cam_file processing FAILED"
    FAILED=$((FAILED + 1))
  fi
done

echo ""
echo "=== Pipeline Complete ==="
echo "Processed: $PROCESSED clips"
echo "Failed:    $FAILED clips"
