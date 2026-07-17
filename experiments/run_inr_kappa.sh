#!/bin/bash
# ① κ 緩和による eagle-dqn 上振れの安定化実験 (INR、プロトコル v2)。
# 中間点 4 種を camera/astronaut × 3 シード × lr 5 点で走査し、
# inrv2 のベースライン (adam/eagle3/eagle4-aj/eagle-dqn/lbfgs) と比較する:
#   eagle-aj-k200 / eagle-aj-k1000: aj の trust region を緩める
#   eagle-dqn-cd: 保険の分解 — cooldown のみ (大ジャンプ許可 + 失敗座標ベンチ)
#   eagle-dqn-k50: 保険の分解 — κ のみ (クリップするがベンチしない)
#
# 集計: python experiments/analyze_inr.py --prefix inrv2 inrk
set -e
cd "$(dirname "$0")/.."

OPTS="${KAPPA_OPTS:-eagle-aj-k200 eagle-aj-k1000 eagle-dqn-cd eagle-dqn-k50}"
LRS="${KAPPA_LRS:-1e-4 3e-4 1e-3 3e-3 1e-2}"

for img in camera astronaut; do
    for seed in 42 43 44; do
        python experiments/pilot_inr.py --image "$img" --seed "$seed" \
            --steps 2000 --prefix inrk --optimizers $OPTS --lrs $LRS
    done
done

echo "INR KAPPA DONE"
