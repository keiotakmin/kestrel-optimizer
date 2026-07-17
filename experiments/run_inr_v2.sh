#!/bin/bash
# INR 本実験 (プロトコル v2): SIREN 画像フィッティング、
# lr グリッド × 3 シード × 2 画像 × {adam, eagle3, eagle4-aj, eagle-dqn, lbfgs}。
#
# 注意: wall-clock の公正性のため必ず単一ジョブで逐次実行すること
# (複数グリッドの並列実行は資源競合で time 基準を汚染する — パイロットの教訓)。
# 各ランは計測前ウォームアップ済み (pilot_inr.py --warmup-steps)。
#
# 使い方: bash experiments/run_inr_v2.sh
# 集計:   python experiments/analyze_inr.py --prefix inrv2
set -e
cd "$(dirname "$0")/.."

OPTS="${INR_OPTS:-adam eagle3 eagle4-aj eagle-dqn lbfgs}"
LRS="${INR_LRS:-1e-4 3e-4 1e-3 3e-3 1e-2}"
STEPS="${INR_STEPS:-2000}"

for img in camera astronaut; do
    for seed in 42 43 44; do
        python experiments/pilot_inr.py --image "$img" --seed "$seed" \
            --steps "$STEPS" --prefix inrv2 \
            --optimizers $OPTS --lrs $LRS --lbfgs-lrs 0.1 0.3 1.0
    done
done

echo "INR V2 DONE"
