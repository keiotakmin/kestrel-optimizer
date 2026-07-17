#!/bin/bash
# 評価プロトコル v2: lr スイープ × 3 シードのグリッドを走らせる。
#
# 敵対的レビュー (2026-07-12) の教訓に基づく公正な比較プロトコル:
#   - Adam と EAGLE 系を同一の lr グリッドでスイープし、家族エンベロープ
#     (家族内最良 lr) 同士で比較する (単一 lr の比較は artifact を生む)
#   - 進捗指標は running train loss でなく固定 train 評価サブセット
#     (train_eval_loss) と test_loss (run_comparison の既定で記録される)
#   - 各オプティマイザは計測前ウォームアップ済み (wall-clock の実行順交絡除去)
#
# 使い方: bash experiments/run_protocol.sh [dataset ...]   (既定: iris covtype)
#          PROTO_OPTS="adam eagle4" bash experiments/run_protocol.sh wine
#          PROTO_PREFIX=protoa で結果ディレクトリの prefix を変更
#          (既存 proto_* の上書きを防ぐ。集計は --prefix でマージ)
# 集計:   python experiments/analyze_protocol.py --dataset <dataset>
set -e
cd "$(dirname "$0")/.."

OPTS="${PROTO_OPTS:-adam eagle4 eagle4-m}"
PREFIX="${PROTO_PREFIX:-proto}"
SEEDS="42 43 44"

for ds in "${@:-iris covtype}"; do
    case $ds in
        iris)    LRS="0.003 0.01 0.03 0.1"; ARGS="--epochs 100 --hidden 25";;
        wine)    LRS="0.003 0.01 0.03 0.1"; ARGS="--epochs 100 --hidden 15";;
        cancer)  LRS="0.003 0.01 0.03 0.1"; ARGS="--epochs 100 --hidden 25";;
        covtype) LRS="3e-4 1e-3 3e-3 1e-2"
                 ARGS="--epochs 10 --batch-size 256 --hidden 128 --eval-every 500";;
        adult)   LRS="3e-4 1e-3 3e-3 1e-2"
                 ARGS="--epochs 20 --batch-size 256 --hidden 128 --eval-every 100";;
        higgs)   LRS="3e-4 1e-3 3e-3 1e-2"
                 ARGS="--epochs 20 --batch-size 256 --hidden 128 --eval-every 200";;
        # 回帰 (フルバッチ + tanh = 割線法の理論レジーム σ→0・滑らか・MSE)。
        # フルバッチでは 1 エポック = 1 ステップ。L-BFGS も回すこと:
        #   PROTO_OPTS="adam eagle4 eagle4-m lbfgs" bash run_protocol.sh california
        california) LRS="1e-3 3e-3 1e-2 3e-2 1e-1"
                 ARGS="--epochs 2000 --full-batch --activation tanh --hidden 64 --eval-every 20";;
        concrete) LRS="1e-3 3e-3 1e-2 3e-2 1e-1"
                 ARGS="--epochs 3000 --full-batch --activation tanh --hidden 64 --eval-every 20";;
        energy)  LRS="1e-3 3e-3 1e-2 3e-2 1e-1"
                 ARGS="--epochs 3000 --full-batch --activation tanh --hidden 64 --eval-every 20";;
        *) echo "unknown dataset: $ds" >&2; exit 1;;
    esac
    # PROTO_LRS で lr グリッドを上書き (adahessian など lr スケールが
    # Adam と桁違いの手法用。プレフィックスを分けて走らせること)
    LRS="${PROTO_LRS:-$LRS}"
    for seed in $SEEDS; do
        for lr in $LRS; do
            name="${PREFIX}_${ds}_lr${lr}_s${seed}"
            # 再開安全化: 完了済み (metrics.json あり) はスキップ。
            # 注意: run_comparison は同名 dir の metrics.json を上書きする
            # ため、既存グリッドへのオプティマイザ追加は必ず新しい PREFIX
            # で走らせ、集計側 (--prefix / gen_macros) でマージすること
            if [ -f "results/${name}/metrics.json" ]; then
                echo "skip ${name} (metrics.json exists)"
                continue
            fi
            python experiments/run_comparison.py --dataset "$ds" \
                --optimizers $OPTS $ARGS --lr "$lr" --seed "$seed" \
                --name "$name"
        done
    done
done

echo "PROTOCOL DONE"
