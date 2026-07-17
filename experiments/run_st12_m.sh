#!/bin/bash
# ステージ12: eagle4-m (分子 m̂ 化) の残りドメイン検証
set -e
cd "$(dirname "$0")/.."

OPTS="adam eagle4 eagle4-m"

for seed in 42 43 44; do
    python experiments/run_comparison.py --dataset cancer --optimizers $OPTS \
        --epochs 100 --lr 0.01 --hidden 25 --seed $seed --name st12_cancer_s$seed
    python experiments/run_comparison.py --dataset covtype --optimizers $OPTS \
        --epochs 10 --lr 1e-3 --batch-size 256 --hidden 128 --eval-every 500 \
        --seed $seed --name st12_covtype_s$seed
    python experiments/run_comparison.py --dataset adult --optimizers $OPTS \
        --epochs 20 --lr 1e-3 --batch-size 256 --hidden 128 --eval-every 100 \
        --seed $seed --name st12_adult_s$seed
done

echo "STAGE12 DONE"
