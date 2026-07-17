#!/bin/bash
# B1 (trust region) / B2 (SNR ゲート) のハイパラスイープ
# iris / wine × 3 シード。ベースライン: adam, eagle-orig
set -e
cd "$(dirname "$0")/.."

OPTS="adam eagle-orig eagle-b2-025 eagle-b2-05 eagle-b2-1 eagle-b1-10 eagle-b1-50 eagle-b1-200"

for seed in 42 43 44; do
    python experiments/run_comparison.py --dataset iris --optimizers $OPTS \
        --epochs 100 --lr 0.01 --hidden 25 --seed $seed --name sweep_iris_s$seed
    python experiments/run_comparison.py --dataset wine --optimizers $OPTS \
        --epochs 100 --lr 0.01 --hidden 15 --seed $seed --name sweep_wine_s$seed
done

echo "SWEEP DONE"
