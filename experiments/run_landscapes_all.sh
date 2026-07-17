#!/bin/bash
# 全データセット/バックボーンの損失地形を統一設定で取得する
# (固定評価サブセット版。EAGLE / Adam / SGD の解を比較)
set -e
cd "$(dirname "$0")/.."

COMMON="--samples-per-layer 20 --n-points 200 --param-range 5.0 --max-batches 10"

for opt in eagle adam sgd; do
    python experiments/run_landscape.py --dataset iris   --optimizer $opt \
        --epochs 100 --lr 0.01 --hidden 25 $COMMON --name ls2_iris_$opt
    python experiments/run_landscape.py --dataset wine   --optimizer $opt \
        --epochs 100 --lr 0.01 --hidden 15 $COMMON --name ls2_wine_$opt
    python experiments/run_landscape.py --dataset cancer --optimizer $opt \
        --epochs 100 --lr 0.01 --hidden 25 $COMMON --name ls2_cancer_$opt
done

for opt in eagle adam; do
    python experiments/run_landscape.py --dataset mnist --arch mlp --hidden 100 \
        --optimizer $opt --epochs 3 --lr 1e-3 $COMMON --name ls2_mnistmlp_$opt
    python experiments/run_landscape.py --dataset mnist --arch cnn \
        --optimizer $opt --epochs 2 --lr 1e-3 $COMMON --name ls2_mnistcnn_$opt
done

# fused カーネルでの実学習時間の確認 (MNIST CNN 1 エポック)
python experiments/run_comparison.py --dataset mnist --arch cnn \
    --optimizers eagle adam --epochs 1 --lr 1e-3 --eval-every 200 \
    --name mnist_cnn_speedcheck

echo "ALL LANDSCAPES DONE"
