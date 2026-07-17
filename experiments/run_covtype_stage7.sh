#!/bin/bash
# ステージ7: Covertype 本実験 (3 シード、10 エポック)
set -e
cd "$(dirname "$0")/.."

for seed in 42 43 44; do
    python experiments/run_comparison.py --dataset covtype \
        --optimizers adam eagle2 eagle3 \
        --epochs 10 --lr 1e-3 --batch-size 256 --hidden 128 --eval-every 500 \
        --seed $seed --name st7_covtype_s$seed
done

echo "STAGE7 DONE"
