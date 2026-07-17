#!/bin/bash
# ステージ6: 間欠ペア + 曲率 EMA (eagle3) の検証
# 比較: adam / eagle2 / eagle2-pb / eagle3。MNIST は seed 42 のみ
set -e
cd "$(dirname "$0")/.."

OPTS="adam eagle2 eagle2-pb eagle3"

for seed in 42 43 44; do
    python experiments/run_comparison.py --dataset iris --optimizers $OPTS \
        --epochs 100 --lr 0.01 --hidden 25 --seed $seed --name st6_iris_s$seed
    python experiments/run_comparison.py --dataset wine --optimizers $OPTS \
        --epochs 100 --lr 0.01 --hidden 15 --seed $seed --name st6_wine_s$seed
    python experiments/run_comparison.py --dataset cancer --optimizers $OPTS \
        --epochs 100 --lr 0.01 --hidden 25 --seed $seed --name st6_cancer_s$seed
done

python experiments/run_comparison.py --dataset mnist --arch mlp --hidden 100 \
    --optimizers $OPTS --epochs 5 --lr 1e-3 --eval-every 100 \
    --seed 42 --name st6_mnist-mlp_s42
python experiments/run_comparison.py --dataset mnist --arch cnn \
    --optimizers $OPTS --epochs 3 --lr 1e-3 --eval-every 100 \
    --seed 42 --name st6_mnist-cnn_s42

echo "STAGE6 DONE"
