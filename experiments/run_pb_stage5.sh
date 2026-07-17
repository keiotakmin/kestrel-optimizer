#!/bin/bash
# ステージ5: ペアバッチ割線の実データ検証
set -e
cd "$(dirname "$0")/.."

OPTS="adam eagle-orig eagle2 eagle-pb eagle2-pb"

for seed in 42 43 44; do
    python experiments/run_comparison.py --dataset iris --optimizers $OPTS \
        --epochs 100 --lr 0.01 --hidden 25 --seed $seed --name st5_iris_s$seed
    python experiments/run_comparison.py --dataset wine --optimizers $OPTS \
        --epochs 100 --lr 0.01 --hidden 15 --seed $seed --name st5_wine_s$seed
    python experiments/run_comparison.py --dataset cancer --optimizers $OPTS \
        --epochs 100 --lr 0.01 --hidden 25 --seed $seed --name st5_cancer_s$seed
    python experiments/run_comparison.py --dataset mnist --arch mlp --hidden 100 \
        --optimizers $OPTS --epochs 5 --lr 1e-3 --eval-every 100 \
        --seed $seed --name st5_mnist-mlp_s$seed
    python experiments/run_comparison.py --dataset mnist --arch cnn \
        --optimizers $OPTS --epochs 3 --lr 1e-3 --eval-every 100 \
        --seed $seed --name st5_mnist-cnn_s$seed
done

echo "STAGE5 DONE"
