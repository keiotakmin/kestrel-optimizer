#!/bin/bash
# ICTAI 2026 ベースライン一斉計測 (ICMLA 査読 R5–R8 対応、2026-07-16)。
#
# 全手法 (新ベースライン 5 + adam/lbfgs/eagle3/eagle-dqn-cd) を
# **単一ジョブで逐次実行**し、同一負荷条件のクリーンな wall-clock を取る。
# eagle-dqn-cd は fused カーネル対応後の再計測を兼ねる (既存 kodakcd_*/
# protod_* の time はフォールバック実装 = Adam 比 +41%〜x1.7 の
# オーバーヘッド込みなので、time 基準の図はこのジョブの結果で描くこと)。
#
# GPU アイドル待ち: 開始前に GPU 使用率が IDLE_UTIL% 以下で IDLE_CHECKS 回
# 連続するまで待つ (melon は共用サーバのため)。IDLE_WAIT=0 で無効。
#
# 起動例 (セッション死に耐える分離起動、確立済みの運用):
#   mkdir -p /var/tmp/fujimoto
#   setsid bash experiments/run_ictai_baselines.sh \
#       > /var/tmp/fujimoto/ictai_baselines.log 2>&1 < /dev/null &
#
# ステージ選択: STAGES="A" bash experiments/run_ictai_baselines.sh
#   A = 回帰 3 データセット (protoe_* + protoeh_*)   [~2-4h]
#   B = INR camera/astronaut × 3 シード (inrv3_*)     [~6-8h]
#   C = Kodak-24 エンベロープ (kodakb_*)              [~10-14h]
#
# 集計:
#   python experiments/analyze_protocol.py --dataset california concrete energy \
#       --prefix protoe protoeh
#   python experiments/analyze_inr.py --prefix inrv3
#   python experiments/analyze_inr.py --prefix kodakb --seeds 42 \
#       --images kodim01 ... kodim24
set -e
cd "$(dirname "$0")/.."

STAGES="${STAGES:-A B C}"

# 新ベースライン + 主要既存手法 (adahessian は lr スケールが違うので別走)。
# bb は発散を含めて記録する対照 (「大域割線も保険を要する」証拠)、
# bb-stab が強いベースライン
MAIN_OPTS="adam adamw adam-cos adabelief bb bb-stab lbfgs eagle3 eagle-dqn-cd"
AH_LRS_REG="3e-3 1e-2 3e-2 1e-1 3e-1"   # 回帰用 adahessian グリッド
AH_LRS_INR="1e-2 3e-2 1e-1 3e-1"        # INR 用 (論文既定 0.1 を中心に)

wait_gpu_idle() {
    [ "${IDLE_WAIT:-1}" = "0" ] && return 0
    local thresh="${IDLE_UTIL:-10}" need="${IDLE_CHECKS:-5}" \
          interval="${IDLE_INTERVAL:-60}" ok=0 util
    echo "[idle-wait] GPU 使用率 <= ${thresh}% が ${need} 回連続するまで待機 " \
         "(${interval}s 間隔)..."
    while [ "$ok" -lt "$need" ]; do
        util=$(nvidia-smi --query-gpu=utilization.gpu \
               --format=csv,noheader,nounits | head -1)
        if [ "$util" -le "$thresh" ]; then
            ok=$((ok + 1))
        else
            ok=0
        fi
        echo "[idle-wait] $(date '+%F %T') util=${util}% (連続 ${ok}/${need})"
        [ "$ok" -lt "$need" ] && sleep "$interval"
    done
    echo "[idle-wait] GPU アイドルを確認、計測を開始する"
}

wait_gpu_idle

if [[ "$STAGES" == *A* ]]; then
    echo "=== Stage A: 回帰 3 データセット ($(date '+%F %T')) ==="
    PROTO_PREFIX=protoe PROTO_OPTS="$MAIN_OPTS" \
        bash experiments/run_protocol.sh california concrete energy
    PROTO_PREFIX=protoeh PROTO_OPTS="adahessian" PROTO_LRS="$AH_LRS_REG" \
        bash experiments/run_protocol.sh california concrete energy
fi

if [[ "$STAGES" == *B* ]]; then
    echo "=== Stage B: INR camera/astronaut ($(date '+%F %T')) ==="
    for img in camera astronaut; do
        for seed in 42 43 44; do
            python experiments/pilot_inr.py --image "$img" --seed "$seed" \
                --steps 2000 --prefix inrv3 \
                --optimizers $MAIN_OPTS adahessian \
                --lrs 1e-4 3e-4 1e-3 3e-3 1e-2 \
                --lbfgs-lrs 0.1 0.3 1.0 \
                --adahessian-lrs $AH_LRS_INR
        done
    done
fi

if [[ "$STAGES" == *C* ]]; then
    echo "=== Stage C: Kodak-24 ($(date '+%F %T')) ==="
    # 統計単位 = 画像 (24 枚)、シード 42 固定、lr 3 点エンベロープ
    # (既存 kodak_*/kodakcd_* と同じプロトコル)。lbfgs は既存 kodak_* で
    # 全敗 (0/24) 確定済み + steps/grad_evals 基準は競合の影響を受けない
    # ため再走しない (time 図に載せる場合のみ KODAK_OPTS で追加)
    for i in $(seq -w 1 24); do
        python experiments/pilot_inr.py --image "kodim$i" --seed 42 \
            --steps 2000 --prefix kodakb \
            --optimizers ${KODAK_OPTS:-adam adamw adam-cos adabelief bb-stab eagle-dqn-cd adahessian} \
            --lrs 3e-4 1e-3 3e-3 \
            --adahessian-lrs $AH_LRS_INR
    done
fi

echo "ICTAI BASELINES DONE ($(date '+%F %T'))"
