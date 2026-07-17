#!/bin/bash
# 最終図用: st6 の MNIST に seed 43, 44 を追加 (3 シード帯にするため)
set -e
cd "$(dirname "$0")/.."

OPTS="adam eagle2 eagle2-pb eagle3"

for seed in 43 44; do
    python experiments/run_comparison.py --dataset mnist --arch mlp --hidden 100 \
        --optimizers $OPTS --epochs 5 --lr 1e-3 --eval-every 100 \
        --seed $seed --name st6_mnist-mlp_s$seed
    python experiments/run_comparison.py --dataset mnist --arch cnn \
        --optimizers $OPTS --epochs 3 --lr 1e-3 --eval-every 100 \
        --seed $seed --name st6_mnist-cnn_s$seed
done

echo "MNIST SEEDS DONE"
