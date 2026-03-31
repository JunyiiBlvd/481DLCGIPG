#!/bin/bash
# download_watchdog.sh — run a download_images.py until no missing images remain.
#
# Usage (run from project root):
#   bash download_watchdog.sh ja  [extra args]   # James Allen
#   bash download_watchdog.sh be  [extra args]   # Brilliant Earth natural
#   bash download_watchdog.sh be  --lab          # Brilliant Earth lab
#
# Examples:
#   bash download_watchdog.sh ja            > /tmp/ja_dl.log 2>&1 &
#   bash download_watchdog.sh be  --lab     > /tmp/be_lab_dl.log 2>&1 &

SITE=$1
shift
EXTRA_ARGS="$@"

case "$SITE" in
    ja)  SCRIPT_DIR="ja_scraper" ;;
    be)  SCRIPT_DIR="be_scraper" ;;
    *)   echo "Usage: $0 ja|be [extra args]"; exit 1 ;;
esac

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV_PYTHON="$PROJECT_ROOT/venv/bin/python"
SCRIPT="download_images.py"
FAILED_LOG="output/download_failures.csv"

# cd into site dir so relative Path("output") resolves correctly
cd "$PROJECT_ROOT/$SCRIPT_DIR" || { echo "ERROR: cannot cd to $SCRIPT_DIR"; exit 1; }
MAX_RESTARTS=5
PAUSE=60   # seconds between restarts — lets CDN rate limits cool

count_missing() {
    # Count rows in failures log (minus header)
    if [ -f "$FAILED_LOG" ]; then
        echo $(( $(wc -l < "$FAILED_LOG") - 1 ))
    else
        echo 0
    fi
}

echo "[dl-watchdog] site=$SITE  script=$SCRIPT  extra='$EXTRA_ARGS'"
echo "[dl-watchdog] Will restart up to $MAX_RESTARTS times on failures."

# First pass — full download
echo "[dl-watchdog] === Pass 1 (full) — $(date) ==="
"$VENV_PYTHON" "$SCRIPT" --workers 8 $EXTRA_ARGS

MISSING=$(count_missing)
echo "[dl-watchdog] Pass 1 done. Failures: $MISSING"

# Subsequent passes — missing only
for i in $(seq 2 $((MAX_RESTARTS + 1))); do
    if [ "$MISSING" -eq 0 ]; then
        echo "[dl-watchdog] No failures. All done."
        exit 0
    fi

    echo "[dl-watchdog] Waiting ${PAUSE}s before retry pass $i..."
    sleep $PAUSE

    echo "[dl-watchdog] === Pass $i (--missing) — $(date) ==="
    "$VENV_PYTHON" "$SCRIPT" --workers 8 --missing $EXTRA_ARGS

    MISSING=$(count_missing)
    echo "[dl-watchdog] Pass $i done. Remaining failures: $MISSING"
done

if [ "$MISSING" -gt 0 ]; then
    echo "[dl-watchdog] $MISSING images still failed after $MAX_RESTARTS retries."
    echo "[dl-watchdog] Check $FAILED_LOG for details."
    exit 1
fi

echo "[dl-watchdog] All images downloaded successfully."
