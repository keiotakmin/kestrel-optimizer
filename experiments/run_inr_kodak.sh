#!/bin/bash
# ② INR 画像セット拡大: Kodak 24 枚 (768×512、data/kodak/)。
# 統計単位は画像 (24 枚) なのでシードは 42 固定、lr は inrv2 の
# エンベロープ位置に基づく 3 点に絞る (adam/eagle 系の最良は
# 3e-4〜3e-3 に収まることを inrv2 で確認済み)。
#
# 集計: python experiments/analyze_inr.py --prefix kodak \
#         --images kodim01 ... kodim24 --seeds 42
set -e
cd "$(dirname "$0")/.."

OPTS="${KODAK_OPTS:-adam eagle3 eagle-dqn lbfgs}"
LRS="${KODAK_LRS:-3e-4 1e-3 3e-3}"

for i in $(seq -w 1 24); do
    python experiments/pilot_inr.py --image "kodim$i" --seed 42 \
        --steps 2000 --prefix kodak --optimizers $OPTS --lrs $LRS \
        --lbfgs-lrs 0.3 1.0
done

echo "INR KODAK DONE"
