#!/bin/bash
# Engram AutoResearch - 单次实验运行脚本
# 用法: ./run_experiment.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Engram AutoResearch Experiment ==="
echo "Time: $(date)"
echo "Branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'not a git repo')"
echo "Commit: $(git rev-parse --short HEAD 2>/dev/null || echo 'N/A')"
echo ""

# 运行训练
echo "Starting training..."
python train.py > run.log 2>&1
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "FAILED (exit code: $EXIT_CODE)"
    echo "--- Last 50 lines of run.log ---"
    tail -n 50 run.log
    exit 1
fi

# 提取结果
echo ""
echo "=== Results ==="
grep "^recall_score:\|^val_ppl:\|^peak_vram_mb:\|^training_seconds:\|^engram_params_M:\|^base_model:" run.log

echo ""
echo "Full log saved to run.log"
