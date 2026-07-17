#!/bin/bash
# ステージ4: B3 の MNIST 検証 (adam / orig / tr / tr-cd5 / tr-cd20)
set -e
cd "$(dirname "$0")/.."

OPTS="adam eagle-orig eagle-tr eagle-tr-cd5 eagle-tr-cd20"

for seed in 42 43 44; do
    python experiments/run_comparison.py --dataset mnist --arch mlp --hidden 100 \
        --optimizers $OPTS --epochs 5 --lr 1e-3 --eval-every 100 \
        --seed $seed --name st4_mnist-mlp_s$seed
    python experiments/run_comparison.py --dataset mnist --arch cnn \
        --optimizers $OPTS --epochs 3 --lr 1e-3 --eval-every 100 \
        --seed $seed --name st4_mnist-cnn_s$seed
done

echo "STAGE4 DONE"
