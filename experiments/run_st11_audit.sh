#!/bin/bash
# ステージ11: 監査候補の検証 — 曲率信頼度ゲート (conf) と分子 m̂ 化 (-m)
# 判別データセット: iris (終盤ジャンプ有効) / higgs (ノイズ床) / wine (中間)
set -e
cd "$(dirname "$0")/.."

OPTS="adam eagle4 eagle4-m eagle3-conf05 eagle3-conf1 eagle3-conf1-m"

for seed in 42 43 44; do
    python experiments/run_comparison.py --dataset iris --optimizers $OPTS \
        --epochs 100 --lr 0.01 --hidden 25 --seed $seed --name st11_iris_s$seed
    python experiments/run_comparison.py --dataset wine --optimizers $OPTS \
        --epochs 100 --lr 0.01 --hidden 15 --seed $seed --name st11_wine_s$seed
    python experiments/run_comparison.py --dataset higgs --optimizers $OPTS \
        --epochs 20 --lr 1e-3 --batch-size 256 --hidden 128 --eval-every 200 \
        --seed $seed --name st11_higgs_s$seed
done

echo "STAGE11 DONE"
