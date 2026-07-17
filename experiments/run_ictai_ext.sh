#!/bin/bash
# Stage A の lr グリッド端対応: エンベロープが格子の端に張り付いた家族の
# 上方拡張 (「エンベロープが端 = チューニング不足」という査読批判の芽を摘む)。
#
# 本計測ジョブ (run_ictai_baselines.sh) の完了マーカーを待ってから走る
# (並列実行は資源競合で Stage B/C の time 計測を汚染するため)。
#
# 起動: setsid bash experiments/run_ictai_ext.sh \
#           > /var/tmp/fujimoto/ictai_ext.log 2>&1 < /dev/null &
set -e
cd "$(dirname "$0")/.."
LOG=/var/tmp/fujimoto/ictai_baselines.log

echo "[chain-wait] 本計測ジョブの完了を待機 (5 分間隔で確認)..."
while ! grep -q "ICTAI BASELINES DONE" "$LOG" 2>/dev/null; do sleep 300; done
echo "[chain-wait] 完了を確認、グリッド拡張を開始 ($(date '+%F %T'))"

# adabelief: Stage A で 3 データセットすべて最良 lr = 1e-1 (グリッド上端)
PROTO_PREFIX=protoe PROTO_OPTS="adabelief" PROTO_LRS="3e-1 1" \
    bash experiments/run_protocol.sh california concrete energy
# adahessian: 最良 lr = 3e-1 (専用グリッド上端、california/concrete)
PROTO_PREFIX=protoeh PROTO_OPTS="adahessian" PROTO_LRS="1 3" \
    bash experiments/run_protocol.sh california concrete energy

echo "ICTAI EXT DONE ($(date '+%F %T'))"
