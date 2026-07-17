#!/bin/bash
# ステージ10: G2 ゲートのリグレッション確認 (eagle3 が既に強いドメインを損なわないか)
set -e
cd "$(dirname "$0")/.."

OPTS="adam eagle3 eagle3-g2-025 eagle3-g2-05"

for seed in 42 43 44; do
    python experiments/run_comparison.py --dataset iris --optimizers $OPTS \
        --epochs 100 --lr 0.01 --hidden 25 --seed $seed --name st10_iris_s$seed
    python experiments/run_comparison.py --dataset wine --optimizers $OPTS \
        --epochs 100 --lr 0.01 --hidden 15 --seed $seed --name st10_wine_s$seed
    python experiments/run_comparison.py --dataset cancer --optimizers $OPTS \
        --epochs 100 --lr 0.01 --hidden 25 --seed $seed --name st10_cancer_s$seed
    python experiments/run_comparison.py --dataset covtype --optimizers $OPTS \
        --epochs 10 --lr 1e-3 --batch-size 256 --hidden 128 --eval-every 500 \
        --seed $seed --name st10_covtype_s$seed
done

echo "STAGE10 DONE"
