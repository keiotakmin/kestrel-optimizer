#!/bin/bash
# ステージ9: SNR 発火ゲート (G1/G2) の検証 — adult/higgs の中終盤失速の解消を狙う
set -e
cd "$(dirname "$0")/.."

OPTS="adam eagle3 eagle3-g1 eagle3-g2-025 eagle3-g2-05"

for seed in 42 43 44; do
    python experiments/run_comparison.py --dataset adult --optimizers $OPTS \
        --epochs 20 --lr 1e-3 --batch-size 256 --hidden 128 --eval-every 100 \
        --seed $seed --name st9_adult_s$seed
    python experiments/run_comparison.py --dataset higgs --optimizers $OPTS \
        --epochs 20 --lr 1e-3 --batch-size 256 --hidden 128 --eval-every 200 \
        --seed $seed --name st9_higgs_s$seed
done

echo "STAGE9 DONE"
