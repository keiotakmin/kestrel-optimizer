"""オプティマイザ比較実験。

例:
    python experiments/run_comparison.py --dataset iris --optimizers eagle adam sgd --epochs 100
    python experiments/run_comparison.py --dataset mnist --arch cnn --optimizers eagle adam --epochs 3 --eval-every 100
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch.nn as nn
import torch.optim as optim

from eagle import (EAGLE, build_model, convergence_speedup, get_dataloaders,
                   get_device, make_train_eval_loader, plot_comparison,
                   set_seed, train_model, warmup_optimizer)
from eagle.baselines import BB, AdamCosine, KestrelCosine
from eagle.data import DATASET_INFO, dataset_task


def _build_adabelief(params, lr):
    # 著者公式パッケージ (adabelief-pytorch)。論文推奨既定 (eps=1e-16,
    # weight_decouple=True, rectify=True) をそのまま使う
    from adabelief_pytorch import AdaBelief
    return AdaBelief(params, lr=lr, eps=1e-16, print_change_log=False)


def _build_adahessian(params, lr):
    # pytorch_optimizer 実装。Hutchinson HVP (rademacher, num_samples=1) で
    # 対角ヘッセを推定する = 毎ステップ追加 backward 1 回。
    # 学習ループ側の対応が 2 つ必要: loss.backward(create_graph=True)
    # (NEEDS_CREATE_GRAPH) と grad_evals の 2/step 計上 (GRAD_EVALS_PER_STEP)
    from pytorch_optimizer import AdaHessian
    return AdaHessian(params, lr=lr)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

# 名前 -> オプティマイザの対応 (params, lr, total_steps)。
# eagle-orig は卒論オリジナル設定
OPTIMIZER_BUILDERS = {
    "adam": lambda params, lr, T=None: optim.Adam(params, lr=lr),
    "sgd": lambda params, lr, T=None: optim.SGD(params, lr=lr, momentum=0.9),
    # 回帰・フルバッチ土俵の王者ベースライン。1 記録ステップ = step() 1 回で、
    # strong Wolfe line search の複数回の関数評価はステップ数に現れない
    # → L-BFGS との計算量公平な比較は time 基準で行う
    "lbfgs": lambda params, lr, T=None: optim.LBFGS(
        params, lr=lr, max_iter=1, history_size=10,
        line_search_fn="strong_wolfe"),
    "eagle": lambda params, lr, T=None: EAGLE(
        params, lr=lr, base="adam", adaptive_threshold=True,
        use_lr_in_eagle_update=True),
    "eagle-s": lambda params, lr, T=None: EAGLE(
        params, lr=lr, base="sgd", adaptive_threshold=True,
        use_lr_in_eagle_update=True),
    "eagle-orig": lambda params, lr, T=None: EAGLE(
        params, lr=lr, base="adam", adaptive_threshold=False,
        threshold=5e-4, use_lr_in_eagle_update=False),
}


def _orig_variant(anneal=False, **kw):
    """eagle-orig をベースに改良を足したバリアント。anneal=True なら
    EAGLE 係数を総ステップ数で 1→lr に減衰させる。"""
    def build(params, lr, T=None):
        if anneal:
            kw["coeff_anneal_steps"] = T
        return EAGLE(params, lr=lr, base="adam", adaptive_threshold=False,
                     threshold=5e-4, use_lr_in_eagle_update=False, **kw)
    return build


# B2 (SNR ゲート) / B1 (trust region) / B3 (クールダウン) / アニーリング対照
OPTIMIZER_BUILDERS.update({
    "eagle-b2-025": _orig_variant(snr_gate=0.25),
    "eagle-b2-05": _orig_variant(snr_gate=0.5),
    "eagle-b2-1": _orig_variant(snr_gate=1.0),
    "eagle-b1-10": _orig_variant(trust_kappa=10),
    "eagle-b1-50": _orig_variant(trust_kappa=50),
    "eagle-b1-200": _orig_variant(trust_kappa=200),
    "eagle-b1b2": _orig_variant(snr_gate=0.5, trust_kappa=50),
    # 採用版: eagle-orig + trust region κ=50 (ステージ2で全データセット検証済み)
    "eagle-tr": _orig_variant(trust_kappa=50),
    # B3: 失敗判定つきクールダウン (eagle-tr に追加)
    "eagle-tr-cd5": _orig_variant(trust_kappa=50, cooldown_steps=5),
    "eagle-tr-cd20": _orig_variant(trust_kappa=50, cooldown_steps=20),
    # 対照: 係数アニーリング 1→lr (eagle-tr に追加)
    "eagle-tr-anneal": _orig_variant(trust_kappa=50, anneal=True),
    # 最終採用版 (2026-07-11): orig + trust region κ=50 + クールダウン K=20。
    # 全 5 データセット × 複数シードで「全進捗域で Adam 以上か同等」を確認済み
    "eagle2": _orig_variant(trust_kappa=50, cooldown_steps=20),
    # ペアバッチ割線 (同一バッチを 2 ステップ連続使用、偶数ステップのみ EAGLE)
    "eagle-pb": _orig_variant(paired_secant=True),
    "eagle2-pb": _orig_variant(trust_kappa=50, cooldown_steps=20,
                               paired_secant=True),
    # 間欠ペア + 曲率 EMA (5 バッチに 1 回クリーン測定、ジャンプは g/h_ema)
    "eagle3": _orig_variant(trust_kappa=50, cooldown_steps=20,
                            curvature_ema=0.8),
    # eagle3 + SNR 発火ゲート (ノイズ床での自動停止を狙う)
    "eagle3-g1": _orig_variant(trust_kappa=50, cooldown_steps=20,
                               curvature_ema=0.8, snr_gate=0.5),
    "eagle3-g2-025": _orig_variant(trust_kappa=50, cooldown_steps=20,
                                   curvature_ema=0.8, signal_gate=0.25),
    "eagle3-g2-05": _orig_variant(trust_kappa=50, cooldown_steps=20,
                                  curvature_ema=0.8, signal_gate=0.5),
    # 採用版 (2026-07-12): eagle3 + G2 発火ゲート c=0.25。
    # st10 で全ドメイン無リグレッション + covtype 初期/wine 中盤を改善
    "eagle4": _orig_variant(trust_kappa=50, cooldown_steps=20,
                            curvature_ema=0.8, signal_gate=0.25),
    # 監査候補の検証: 曲率信頼度ゲート (G2 の代理変数性を解消する本命)
    "eagle3-conf05": _orig_variant(trust_kappa=50, cooldown_steps=20,
                                   curvature_ema=0.8, conf_gate=0.5),
    "eagle3-conf1": _orig_variant(trust_kappa=50, cooldown_steps=20,
                                  curvature_ema=0.8, conf_gate=1.0),
    # 監査候補の検証: ジャンプ分子の m̂ 化 (ノイズ平均された標的)
    "eagle4-m": _orig_variant(trust_kappa=50, cooldown_steps=20,
                              curvature_ema=0.8, signal_gate=0.25,
                              momentum_jump=True),
    "eagle3-conf1-m": _orig_variant(trust_kappa=50, cooldown_steps=20,
                                    curvature_ema=0.8, conf_gate=1.0,
                                    momentum_jump=True),
    # 採用版 (2026-07-12): eagle4 + ジャンプ分子 m̂ 化 (= eagle4-m の昇格)。
    # 密・連続特徴で一貫改善。疎 one-hot (adult 型) では eagle4 を推奨
    "eagle5": _orig_variant(trust_kappa=50, cooldown_steps=20,
                            curvature_ema=0.8, signal_gate=0.25,
                            momentum_jump=True),
    # --- フルバッチ回帰レジームでの機構アブレーション (2026-07-13) ---
    # 機構スタック (EMA/ゲート/保険/振動切替) は分類×ミニバッチ時代の
    # 物差しで選別されており、生き残ったレジームでの必要性は未検証だった。
    # H2: 曲率の鮮度 — EMA を切り、毎クリーン測定をそのまま使う
    "eagle4-bh0": _orig_variant(trust_kappa=50, cooldown_steps=20,
                                curvature_ema=0.0, signal_gate=0.25),
    # H3: 振動切替は必要か — 曲率が有効な全要素で常時ジャンプ
    # (= 保険つき対角準ニュートン)
    "eagle4-aj": _orig_variant(trust_kappa=50, cooldown_steps=20,
                               curvature_ema=0.8, signal_gate=0.25,
                               always_jump=True),
    "eagle4-aj-bh0": _orig_variant(trust_kappa=50, cooldown_steps=20,
                                   curvature_ema=0.0, signal_gate=0.25,
                                   always_jump=True),
    # H4: 保険 (trust region + クールダウン) は必要か
    "eagle4-noins": _orig_variant(curvature_ema=0.8, signal_gate=0.25),
    # H1 の補助: G2 ゲートは決定論レジームで意味があるか (eagle3 = ゲートなし)
    # H3×H4 交互作用: 切替もゲートも保険もない「素の保護なし対角準ニュートン
    # (正値フィルタ + Adam フォールバックのみ)」と、その中間形
    "eagle-dqn": _orig_variant(curvature_ema=0.8, always_jump=True),
    "eagle-dqn-bh0": _orig_variant(curvature_ema=0.0, always_jump=True),
    "eagle4-aj-noins": _orig_variant(curvature_ema=0.8, signal_gate=0.25,
                                     always_jump=True),
    "eagle3-noins": _orig_variant(curvature_ema=0.8),
    # --- κ 緩和による eagle-dqn 上振れの安定化 (2026-07-13、INR 向け) ---
    # dqn の camera 圧勝 (+1.26 dB, x4.7) は無制限ジャンプ由来、astronaut
    # 崩壊は同じ無制限ジャンプ由来。中間点を静的パラメータで走査する:
    # κ を緩めた aj (200/1000)、保険の分解 (cooldown のみ / κ のみ)
    "eagle-aj-k200": _orig_variant(trust_kappa=200, cooldown_steps=20,
                                   curvature_ema=0.8, signal_gate=0.25,
                                   always_jump=True),
    "eagle-aj-k1000": _orig_variant(trust_kappa=1000, cooldown_steps=20,
                                    curvature_ema=0.8, signal_gate=0.25,
                                    always_jump=True),
    "eagle-dqn-cd": _orig_variant(cooldown_steps=20, curvature_ema=0.8,
                                  always_jump=True),
    "eagle-dqn-k50": _orig_variant(trust_kappa=50, curvature_ema=0.8,
                                   always_jump=True),
    # --- K / β_h 感度走査 (ICTAI Limitations の裏付け、2026-07-17) ---
    "eagle-dqn-cd-k5": _orig_variant(cooldown_steps=5, curvature_ema=0.8,
                                     always_jump=True),
    "eagle-dqn-cd-k80": _orig_variant(cooldown_steps=80, curvature_ema=0.8,
                                      always_jump=True),
    "eagle-dqn-cd-bh05": _orig_variant(cooldown_steps=20, curvature_ema=0.5,
                                       always_jump=True),
    "eagle-dqn-cd-bh095": _orig_variant(cooldown_steps=20,
                                        curvature_ema=0.95,
                                        always_jump=True),
    # --- ICTAI 2026 ベースライン (ICMLA 査読 R5–R8 対応、2026-07-16) ---
    # 各手法ファミリーの代表 1 つ: Adam 系標準 / lr スケジュール対照 /
    # 現代適応 1 次法 / DL 実用対角曲率 / 割線法の直系祖先。
    # Sophia (CE 損失依存の曲率推定器)・Shampoo/K-FAC (L-BFGS が既に占める
    # 密曲率枠)・Lion (lr×wd 同時チューニング前提の汎化狙い)・Adafactor
    # (メモリ節約が 10⁵〜10⁶ params で無意味) は Related Work 言及に留める
    "adamw": lambda params, lr, T=None: optim.AdamW(params, lr=lr),
    # Rprop (Riedmiller & Braun 1993): フルバッチ決定論設定の古典的
    # 座標ごと手法 (符号ベース・実質 lr フリー)。lr = 初期ステップ幅で
    # エンベロープ走査の対象 (ICTAI 査読 W1 対応)
    "rprop": lambda params, lr, T=None: optim.Rprop(params, lr=lr),
    "adam-cos": lambda params, lr, T=None: AdamCosine(params, lr=lr,
                                                      T=T or 1000),
    "adabelief": lambda params, lr, T=None: _build_adabelief(params, lr),
    # 注意: lr スケールが Adam と桁違い (論文既定 0.1)。専用 lr グリッドで
    # 回すこと (run_ictai_baselines.sh / pilot_inr.py --adahessian-lrs)
    "adahessian": lambda params, lr, T=None: _build_adahessian(params, lr),
    "bb": lambda params, lr, T=None: BB(params, lr=lr),
    # stabilized BB (Burdakov+ 2019): ステップノルム上限 Δ=1。素の bb は
    # tanh MLP 回帰で非単調発散する (非凸での既知の挙動) ため、強い
    # ベースラインとしてはこちらを使う
    "bb-stab": lambda params, lr, T=None: BB(params, lr=lr, stab_delta=1.0),
    # KESTREL (= eagle-dqn-cd) + フォールバック lr の cosine アニーリング。
    # INR で adam-cos が最終 PSNR 最高だったことへの回答 (到達×磨きの両取り)
    "kestrel-cos": lambda params, lr, T=None: KestrelCosine(
        params, lr=lr, T=T or 1000, base="adam", adaptive_threshold=False,
        threshold=5e-4, use_lr_in_eagle_update=False, cooldown_steps=20,
        curvature_ema=0.8, always_jump=True),
})

# Hutchinson HVP のため loss.backward(create_graph=True) が必要な手法
NEEDS_CREATE_GRAPH = {"adahessian"}
# 1 ステップの勾配評価数 (省略時 1)。adahessian は HVP が backward 相当 1 回分
GRAD_EVALS_PER_STEP = {"adahessian": 2}

# ペアバッチ割線を使うバリアント (train_model に repeat_batch=True を渡す)
PAIRED_OPTIMIZERS = {"eagle-pb", "eagle2-pb"}
# 間欠ペアを使うバリアント (train_model に pair_every を渡す)
PAIR_EVERY = {"eagle3": 5, "eagle3-g1": 5, "eagle3-g2-025": 5,
              "eagle3-g2-05": 5, "eagle4": 5, "eagle3-conf05": 5,
              "eagle3-conf1": 5, "eagle4-m": 5, "eagle3-conf1-m": 5,
              "eagle5": 5, "eagle4-bh0": 5, "eagle4-aj": 5,
              "eagle4-aj-bh0": 5, "eagle4-noins": 5, "eagle-dqn": 5,
              "eagle-dqn-bh0": 5, "eagle4-aj-noins": 5, "eagle3-noins": 5,
              "eagle-aj-k200": 5, "eagle-aj-k1000": 5, "eagle-dqn-cd": 5,
              "eagle-dqn-k50": 5, "kestrel-cos": 5,
              "eagle-dqn-cd-k5": 5, "eagle-dqn-cd-k80": 5,
              "eagle-dqn-cd-bh05": 5, "eagle-dqn-cd-bh095": 5}


def parse_args():
    parser = argparse.ArgumentParser(description="オプティマイザ比較実験")
    parser.add_argument("--dataset", default="iris",
                        choices=sorted(DATASET_INFO))
    parser.add_argument("--optimizers", nargs="+", default=["eagle", "adam"],
                        choices=list(OPTIMIZER_BUILDERS))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--arch", default="mlp", choices=["mlp", "cnn"])
    parser.add_argument("--hidden", type=int, nargs="+", default=[25],
                        help="MLP の隠れ層サイズ (複数指定で多層)")
    parser.add_argument("--activation", default="relu",
                        choices=["relu", "tanh", "gelu"],
                        help="MLP の活性化関数 (tanh/gelu は滑らかな地形用)")
    parser.add_argument("--full-batch", action="store_true",
                        help="全訓練データを 1 バッチにする (σ→0 レジーム)")
    parser.add_argument("--eval-every", type=int, default=None,
                        help="評価・記録の間隔 (ステップ数)。省略時は 1 エポックごと")
    parser.add_argument("--log-early", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="対数間隔の早期記録点を足す (--no-log-early で無効)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", default=None,
                        help="結果ディレクトリ名 (省略時は dataset_日時)")
    # プロトコル v2 (既定で有効。0 指定で旧挙動に戻る)
    parser.add_argument("--warmup-steps", type=int, default=20,
                        help="各オプティマイザの計測前ウォームアップ回数 "
                             "(JIT/cuDNN 初期化の実行順交絡を除去)。0 で無効")
    parser.add_argument("--train-eval-samples", type=int, default=10240,
                        help="固定 train 評価サブセットのサンプル数 "
                             "(不偏な進捗指標 train_eval_loss 用)。0 で無効")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    print(f"Device: {device}")

    name = args.name or f"{args.dataset}_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir = RESULTS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    task = dataset_task(args.dataset)
    criterion = nn.MSELoss() if task == "regression" else nn.CrossEntropyLoss()
    histories = {}
    for opt_name in args.optimizers:
        print(f"\n=== {opt_name} ===")
        set_seed(args.seed)  # 全オプティマイザで同じ初期値・データ順にする
        train_loader, test_loader = get_dataloaders(
            args.dataset, batch_size=args.batch_size, seed=args.seed,
            full_batch=args.full_batch)
        model = build_model(args.dataset, arch=args.arch,
                            hidden_sizes=args.hidden,
                            activation=args.activation).to(device)
        total_steps = args.epochs * len(train_loader)
        builder = OPTIMIZER_BUILDERS[opt_name]
        optimizer = builder(model.parameters(), args.lr, total_steps)
        train_eval_loader = (
            make_train_eval_loader(train_loader, args.train_eval_samples)
            if args.train_eval_samples > 0 else None)
        needs_cg = opt_name in NEEDS_CREATE_GRAPH
        if args.warmup_steps > 0:
            # 固定バッチ (shuffle なしの test_loader 先頭) を使うので RNG を
            # 消費せず、学習の軌道には影響しない
            warmup_optimizer(model, lambda p: builder(p, args.lr, total_steps),
                             next(iter(test_loader)), criterion, device,
                             steps=args.warmup_steps, create_graph=needs_cg)
        # フルバッチでは全勾配が同一関数の勾配なのでペア繰り返しは不要。
        # clean_every_step で毎ステップ測定し、他手法とステップ数を揃える
        # (ペア繰り返しは 2 倍の勾配評価 = 予算の交絡になる)
        pair = PAIR_EVERY.get(opt_name)
        clean_all = args.full_batch and pair is not None
        histories[opt_name] = train_model(
            model, optimizer, train_loader, test_loader, criterion,
            epochs=args.epochs, device=device, eval_every=args.eval_every,
            desc=opt_name, repeat_batch=opt_name in PAIRED_OPTIMIZERS,
            pair_every=None if clean_all else pair,
            clean_every_step=clean_all,
            train_eval_loader=train_eval_loader, log_early=args.log_early,
            create_graph=needs_cg,
            grad_evals_per_step=GRAD_EVALS_PER_STEP.get(opt_name, 1))
        h = histories[opt_name]
        metric = "test_rmse" if task == "regression" else "test_acc"
        print(f"final: train_loss={h['train_loss'][-1]:.4f} "
              f"{metric}={h['test_acc'][-1]:.3f} time={h['time'][-1]:.1f}s")

    # 収束速度分析: EAGLE 系それぞれを非 EAGLE 系ベースラインと比較
    eagle_names = [n for n in histories if n.startswith("eagle")]
    baseline_names = [n for n in histories if not n.startswith("eagle")]
    speedup = {t: convergence_speedup(histories, t, baseline_names)
               for t in eagle_names if baseline_names}

    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"args": vars(args), "histories": histories,
                   "speedup": speedup}, f, indent=2)
    plot_comparison(histories, out_dir / "comparison.png",
                    title=f"{args.dataset} / lr={args.lr} / seed={args.seed}",
                    metric_label="RMSE" if task == "regression"
                    else "Accuracy (%)")

    print("\n=== Convergence speedup ===")
    for target, per_baseline in speedup.items():
        for baseline, s in per_baseline.items():
            if s["target_reached_step"] == 0:
                # step 0 で既に到達 = ベースラインの最終 loss が初期 loss 以上
                # (発散ベースライン)。speedup_ratio は定義できない (None)
                print(f"{target} は {baseline} の最終 loss "
                      f"({s['baseline_final_loss']:.4f}) に step 0 で既に到達")
            elif s["target_reached_step"] is not None:
                print(f"{target} は {baseline} の最終 loss "
                      f"({s['baseline_final_loss']:.4f} @ step {s['baseline_final_step']}) に "
                      f"step {s['target_reached_step']} で到達 "
                      f"(speedup x{s['speedup_ratio']:.2f})")
            else:
                print(f"{target} は {baseline} の最終 loss "
                      f"({s['baseline_final_loss']:.4f}) に未到達")

    print(f"\n結果を保存: {out_dir}")


if __name__ == "__main__":
    main()
