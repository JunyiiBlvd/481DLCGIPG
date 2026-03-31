#!/bin/bash
# watchdog.sh — run scrape_v3.py until all shape checkpoints are complete.
# Restarts after exit (up to MAX_RESTARTS) to collect any bands that failed
# during a stale-session window.
#
# Usage:
#   bash watchdog.sh natural oval,cushion,pear,... /tmp/ja_nat.log
#   bash watchdog.sh lab     oval,cushion,pear,... /tmp/ja_lab.log

PREFIX=$1      # "natural" or "lab"
SHAPES=$2      # comma-separated shape list
LOGFILE=$3

VENV_PYTHON="../venv/bin/python"
MAX_RESTARTS=6
PAUSE_BETWEEN=45   # seconds to wait before restarting (let rate limits cool)
TOTAL_BANDS=575    # 0.25–6.00 in 0.01ct steps

all_done() {
    python3 - <<PYEOF
import json, sys
from pathlib import Path

OUTPUT_DIR = Path("output")
prefix  = "$PREFIX"
shapes  = "$SHAPES".split(",")
total   = $TOTAL_BANDS

incomplete = []
for shape in shapes:
    ckpt = OUTPUT_DIR / f"scrape_v3_{prefix}_{shape}_checkpoint.json"
    if not ckpt.exists():
        incomplete.append(f"{shape}(no ckpt)")
        continue
    done = len(json.load(open(ckpt))["completed_bands"])
    if done < total:
        incomplete.append(f"{shape}({done}/{total})")

if incomplete:
    print("INCOMPLETE: " + ", ".join(incomplete))
    sys.exit(1)
else:
    print("ALL DONE")
    sys.exit(0)
PYEOF
}

echo "[watchdog] Starting. prefix=$PREFIX shapes=$SHAPES logfile=$LOGFILE"
echo "[watchdog] Will restart up to $MAX_RESTARTS times until all shapes complete."

for i in $(seq 1 $MAX_RESTARTS); do
    echo "" >> "$LOGFILE"
    echo "[watchdog] ===== Run $i/$MAX_RESTARTS — $(date) =====" >> "$LOGFILE"

    "$VENV_PYTHON" scrape_v3.py --shapes "$SHAPES" $([ "$PREFIX" = "lab" ] && echo "--lab") >> "$LOGFILE" 2>&1

    STATUS=$(all_done)
    echo "[watchdog] Run $i done. $STATUS" | tee -a "$LOGFILE"

    if all_done > /dev/null 2>&1; then
        echo "[watchdog] All shapes complete. Exiting." | tee -a "$LOGFILE"
        exit 0
    fi

    if [ $i -lt $MAX_RESTARTS ]; then
        echo "[watchdog] Incomplete bands remain. Waiting ${PAUSE_BETWEEN}s then restarting..." | tee -a "$LOGFILE"
        sleep $PAUSE_BETWEEN
    fi
done

echo "[watchdog] Reached max restarts ($MAX_RESTARTS). Check log for remaining failures." | tee -a "$LOGFILE"
