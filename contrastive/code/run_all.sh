#!/bin/bash
# Run all contrastive experiments with a configurable model.
# Usage: MODEL=microsoft/phi-4 bash contrastive/code/run_all.sh
#   or:  bash contrastive/code/run_all.sh          # defaults to phi-2

set -e

export MODEL="${MODEL:-microsoft/phi-2}"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export CUDA_LAUNCH_BLOCKING=0

LOGDIR="/workspace/contrastive_logs"
mkdir -p "$LOGDIR"

SCRIPTS=(
    explore_cases.py
    explore_triplets.py
    explore_2x2.py
    explore_2x2_v2.py
    explore_minimal.py
    direct_readout.py
    explore_current_token.py
    integration_dynamics.py
    syntactic_axes.py
    poster_cases.py
    replicate_landmarks.py
    per_head_analysis.py
)

echo "=============================================="
echo "  Contrastive trajectory suite"
echo "  Model: $MODEL"
echo "  Logs:  $LOGDIR"
echo "=============================================="

PASS=0
FAIL=0
SKIP=0

for script in "${SCRIPTS[@]}"; do
    SPATH="contrastive/code/$script"
    LOGFILE="$LOGDIR/${script%.py}.log"

    if [ -f "$LOGFILE" ] && grep -q "^=== DONE ===" "$LOGFILE" 2>/dev/null; then
        echo "[SKIP] $script (already completed)"
        SKIP=$((SKIP + 1))
        continue
    fi

    echo -n "[RUN]  $script ... "
    if python3 -u "$SPATH" > "$LOGFILE" 2>&1; then
        echo "=== DONE ===" >> "$LOGFILE"
        echo "OK ($(wc -l < "$LOGFILE") lines)"
        PASS=$((PASS + 1))
    else
        echo "FAIL (exit $?), see $LOGFILE"
        FAIL=$((FAIL + 1))
    fi
done

echo ""
echo "=============================================="
echo "  Results: $PASS passed, $FAIL failed, $SKIP skipped"
echo "=============================================="
