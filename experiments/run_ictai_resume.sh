#!/bin/bash
# 2026-07-17 12:52 の全ジョブ死亡 (原因未特定、セッション再起動と時間相関)
# からの復旧ジョブ。旧 ext/ext2/ext3 チェーンをこの 1 本に統合する。
#
# 再開安全: 全ユニットが metrics.json の存在チェックでスキップされるため、
# 何度殺されても失うのは実行中の 1 ラン分だけ。
#
# 重要な修正: 旧 ext2/ext3 は PROTO_PREFIX=protoe で走る設計だったが、
# これは Stage A の protoe_*/metrics.json を上書き破壊するバグ (run_comparison
# は同名 dir に書く)。オプティマイザ追加は別プレフィックス (protoe2/protoe3)
# にし、gen_macros / analyze_protocol --prefix でマージする。
#
# 起動: nohup setsid bash experiments/run_ictai_resume.sh \
#           > /var/tmp/fujimoto/ictai_resume.log 2>&1 < /dev/null &
set -e
cd "$(dirname "$0")/.."

AH_LRS_REG="3e-3 1e-2 3e-2 1e-1 3e-1"
AH_LRS_INR="1e-2 3e-2 1e-1 3e-1"

echo "=== resume: Stage C 残り (kodakb) ($(date '+%F %T')) ==="
for i in $(seq -w 1 24); do
    [ -f "results/kodakb_kodim${i}_s42/metrics.json" ] && continue
    python experiments/pilot_inr.py --image "kodim$i" --seed 42 \
        --steps 2000 --prefix kodakb \
        --optimizers adam adamw adam-cos adabelief bb-stab eagle-dqn-cd adahessian \
        --lrs 3e-4 1e-3 3e-3 --adahessian-lrs $AH_LRS_INR
done

echo "=== resume: グリッド端拡張 (adabelief/adahessian) ($(date '+%F %T')) ==="
PROTO_PREFIX=protoe PROTO_OPTS="adabelief" PROTO_LRS="3e-1 1" \
    bash experiments/run_protocol.sh california concrete energy
PROTO_PREFIX=protoeh PROTO_OPTS="adahessian" PROTO_LRS="1 3" \
    bash experiments/run_protocol.sh california concrete energy

echo "=== resume: 旧 EAGLE ベースライン (protoe2) ($(date '+%F %T')) ==="
PROTO_PREFIX=protoe2 PROTO_OPTS="eagle" \
    bash experiments/run_protocol.sh california concrete energy

echo "=== resume: kestrel-cos (protoe3 / inrv3c / kodakc) ($(date '+%F %T')) ==="
PROTO_PREFIX=protoe3 PROTO_OPTS="kestrel-cos" \
    bash experiments/run_protocol.sh california concrete energy
for img in camera astronaut; do
    for seed in 42 43 44; do
        [ -f "results/inrv3c_${img}_s${seed}/metrics.json" ] && continue
        python experiments/pilot_inr.py --image "$img" --seed "$seed" \
            --steps 2000 --prefix inrv3c --optimizers kestrel-cos \
            --lrs 1e-4 3e-4 1e-3 3e-3 1e-2
    done
done
for i in $(seq -w 1 24); do
    [ -f "results/kodakc_kodim${i}_s42/metrics.json" ] && continue
    python experiments/pilot_inr.py --image "kodim$i" --seed 42 \
        --steps 2000 --prefix kodakc --optimizers kestrel-cos \
        --lrs 3e-4 1e-3 3e-3
done

echo "ICTAI RESUME DONE ($(date '+%F %T'))"
