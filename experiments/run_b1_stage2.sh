#!/bin/bash
# ステージ2: B1 (trust region κ=50/200) の全データセット・複数シード検証
set -e
cd "$(dirname "$0")/.."

OPTS="adam eagle-orig eagle-b1-50 eagle-b1-200"

# 表形式 5 シード
for seed in 42 43 44 45 46; do
    python experiments/run_comparison.py --dataset iris --optimizers $OPTS \
        --epochs 100 --lr 0.01 --hidden 25 --seed $seed --name st2_iris_s$seed
    python experiments/run_comparison.py --dataset wine --optimizers $OPTS \
        --epochs 100 --lr 0.01 --hidden 15 --seed $seed --name st2_wine_s$seed
    python experiments/run_comparison.py --dataset cancer --optimizers $OPTS \
        --epochs 100 --lr 0.01 --hidden 25 --seed $seed --name st2_cancer_s$seed
done

# MNIST 3 シード
for seed in 42 43 44; do
    python experiments/run_comparison.py --dataset mnist --arch mlp --hidden 100 \
        --optimizers $OPTS --epochs 5 --lr 1e-3 --eval-every 100 \
        --seed $seed --name st2_mnist-mlp_s$seed
    python experiments/run_comparison.py --dataset mnist --arch cnn \
        --optimizers $OPTS --epochs 3 --lr 1e-3 --eval-every 100 \
        --seed $seed --name st2_mnist-cnn_s$seed
done

echo "STAGE2 DONE"
