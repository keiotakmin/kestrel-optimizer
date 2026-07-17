#!/bin/bash
# 最終図用の不足データ: eagle-orig を covtype/adult/higgs で実行 (設定は st7/st8 と同一)
set -e
cd "$(dirname "$0")/.."

for seed in 42 43 44; do
    python experiments/run_comparison.py --dataset covtype --optimizers eagle-orig \
        --epochs 10 --lr 1e-3 --batch-size 256 --hidden 128 --eval-every 500 \
        --seed $seed --name fill_covtype_s$seed
    python experiments/run_comparison.py --dataset adult --optimizers eagle-orig \
        --epochs 20 --lr 1e-3 --batch-size 256 --hidden 128 --eval-every 100 \
        --seed $seed --name fill_adult_s$seed
    python experiments/run_comparison.py --dataset higgs --optimizers eagle-orig \
        --epochs 20 --lr 1e-3 --batch-size 256 --hidden 128 --eval-every 200 \
        --seed $seed --name fill_higgs_s$seed
done

echo "FILL DONE"
