"""損失地形 (パラメータ感度) 分析。

学習済みモデルの weight を 1 要素ずつ動かして損失の変化を可視化する
(archive/proj/lossfunction の iris25.py などの統合版)。

例:
    python experiments/run_landscape.py --dataset iris --optimizer eagle --epochs 100 --samples-per-layer 50
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from eagle import (build_model, evaluate, get_dataloaders, get_device,
                   parameter_landscape, plot_landscape, set_seed, train_model)
from run_comparison import OPTIMIZER_BUILDERS

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def parse_args():
    parser = argparse.ArgumentParser(description="損失地形分析")
    parser.add_argument("--dataset", default="iris",
                        choices=["iris", "wine", "cancer", "mnist", "covtype", "adult", "higgs"])
    parser.add_argument("--optimizer", default="eagle",
                        choices=list(OPTIMIZER_BUILDERS))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--arch", default="mlp", choices=["mlp", "cnn"])
    parser.add_argument("--hidden", type=int, nargs="+", default=[25])
    parser.add_argument("--samples-per-layer", type=int, default=20,
                        help="各 weight 行列からサンプリングする要素数")
    parser.add_argument("--param-range", type=float, default=5.0,
                        help="学習値から ± この範囲を走査")
    parser.add_argument("--n-points", type=int, default=100,
                        help="走査する点数")
    parser.add_argument("--max-batches", type=int, default=10,
                        help="損失計算に使う最大バッチ数")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    print(f"Device: {device}")

    name = args.name or (f"landscape_{args.dataset}_{args.optimizer}_"
                         f"{datetime.now():%Y%m%d_%H%M%S}")
    out_dir = RESULTS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    train_loader, test_loader = get_dataloaders(args.dataset, seed=args.seed)
    model = build_model(args.dataset, arch=args.arch,
                        hidden_sizes=args.hidden).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = OPTIMIZER_BUILDERS[args.optimizer](model.parameters(), args.lr)

    print(f"=== {args.optimizer} で学習 ({args.epochs} epochs) ===")
    history = train_model(model, optimizer, train_loader, test_loader,
                          criterion, epochs=args.epochs, device=device,
                          desc=args.optimizer)

    # 地形評価は毎回同じデータで行う必要があるため、訓練データの先頭から
    # 固定サブセットを切り出す (shuffle された train_loader をそのまま使うと
    # 評価点ごとに異なるバッチが選ばれ、曲線がノイズに埋もれる)
    batch_size = train_loader.batch_size
    n_eval = min(args.max_batches * batch_size, len(train_loader.dataset))
    eval_loader = DataLoader(
        Subset(train_loader.dataset, list(range(n_eval))),
        batch_size=batch_size, shuffle=False)

    baseline_loss, _ = evaluate(model, eval_loader, criterion, device)
    print(f"学習後の train loss (地形評価サブセット): {baseline_loss:.4f}")

    print("=== 損失地形を分析 ===")
    landscape = parameter_landscape(
        model, eval_loader, criterion, device,
        samples_per_layer=args.samples_per_layer,
        param_range=args.param_range, n_points=args.n_points,
        max_batches=None, seed=args.seed)

    with open(out_dir / "landscape.json", "w") as f:
        json.dump({"args": vars(args), "baseline_loss": baseline_loss,
                   "history": history, "landscape": landscape}, f)
    plot_landscape(
        landscape, baseline_loss, history["train_loss"],
        out_dir / "landscape.png",
        title=f"Parameter Impact ({args.optimizer}, {args.dataset}, "
              f"hidden={args.hidden})")

    print(f"結果を保存: {out_dir}")


if __name__ == "__main__":
    main()
