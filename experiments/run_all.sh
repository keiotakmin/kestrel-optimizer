#!/bin/bash
# 卒論期の全実験を順次実行する
set -e
cd "$(dirname "$0")/.."

OPTS="eagle eagle-s eagle-orig adam sgd"

# 表形式データ: 卒論設定に合わせ lr=0.01, 100 エポック, 隠れ層は旧実装に対応
python experiments/run_comparison.py --dataset iris   --optimizers $OPTS \
    --epochs 100 --lr 0.01 --hidden 25 --name iris_full

python experiments/run_comparison.py --dataset wine   --optimizers $OPTS \
    --epochs 100 --lr 0.01 --hidden 15 --name wine_full

python experiments/run_comparison.py --dataset cancer --optimizers $OPTS \
    --epochs 100 --lr 0.01 --hidden 25 --name cancer_full

# MNIST: lr=1e-3, 100 ステップごとに記録
python experiments/run_comparison.py --dataset mnist --arch mlp --hidden 100 \
    --optimizers $OPTS --epochs 5 --lr 1e-3 --eval-every 100 --name mnist_mlp

python experiments/run_comparison.py --dataset mnist --arch cnn \
    --optimizers $OPTS --epochs 3 --lr 1e-3 --eval-every 100 --name mnist_cnn

# 損失地形分析: EAGLE / Adam / SGD で学習した解の比較 (Iris)
for opt in eagle adam sgd; do
    python experiments/run_landscape.py --dataset iris --optimizer $opt \
        --epochs 100 --lr 0.01 --samples-per-layer 20 --n-points 200 \
        --name landscape_iris_$opt
done

echo "ALL EXPERIMENTS DONE"
