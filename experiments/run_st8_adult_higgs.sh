#!/bin/bash
# ステージ8: 表形式分類の横展開 (Adult / HIGGS、3 シード、20 エポック)
set -e
cd "$(dirname "$0")/.."

OPTS="adam eagle2 eagle3"

for seed in 42 43 44; do
    python experiments/run_comparison.py --dataset adult --optimizers $OPTS \
        --epochs 20 --lr 1e-3 --batch-size 256 --hidden 128 --eval-every 100 \
        --seed $seed --name st8_adult_s$seed
    python experiments/run_comparison.py --dataset higgs --optimizers $OPTS \
        --epochs 20 --lr 1e-3 --batch-size 256 --hidden 128 --eval-every 200 \
        --seed $seed --name st8_higgs_s$seed
done

echo "STAGE8 DONE"
