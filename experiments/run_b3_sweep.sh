#!/bin/bash
# ステージ3: B3 (失敗判定クールダウン) + アニーリング対照のスイープ
# ベースライン: adam, eagle-orig, eagle-tr (= orig + trust region κ=50)
set -e
cd "$(dirname "$0")/.."

OPTS="adam eagle-orig eagle-tr eagle-tr-cd5 eagle-tr-cd20 eagle-tr-anneal"

for seed in 42 43 44; do
    python experiments/run_comparison.py --dataset iris --optimizers $OPTS \
        --epochs 100 --lr 0.01 --hidden 25 --seed $seed --name st3_iris_s$seed
    python experiments/run_comparison.py --dataset wine --optimizers $OPTS \
        --epochs 100 --lr 0.01 --hidden 15 --seed $seed --name st3_wine_s$seed
    python experiments/run_comparison.py --dataset cancer --optimizers $OPTS \
        --epochs 100 --lr 0.01 --hidden 25 --seed $seed --name st3_cancer_s$seed
done

echo "STAGE3 DONE"
