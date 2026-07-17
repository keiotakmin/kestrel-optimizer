"""EAGLE 系の高速収束条件の解析。

Part 1: 学習初期の収束速度
    各比較実験の train loss 履歴から、損失減少の進捗率 f (50%/80%/90%/95%) に
    到達するまでのステップ数を求め、EAGLE 系の speedup (vs Adam / SGD) を
    マイルストーンごとに計算する。最終性能ではなく初期の速さを測る。

Part 2: 損失地形の形状指標
    ls2_* の損失地形 (1 要素スライス) から形状指標を計算する:
    - convexity_frac: 2 階差分が非負の点の割合 (1.0 = 完全に凸)
    - ruggedness: 正規化した曲線の総変動 (V 字なら ~2、ギザギザだと大きい)
    - n_minima: スライス内の局所解の個数 (1 = 単峰)
    - curvature: 学習値での曲率 (2 階差分 / h^2)
    - flat_frac: 正規化損失が 0.05 以下の点の割合 (谷底の平坦さ)
    - dist_to_min: 学習値とスライス内最小値の距離 / param_range

Part 3: 両者の突き合わせ (speedup vs 形状指標の散布図と表)

実行: python experiments/analyze_results.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT = RESULTS / "analysis"

COMPARISON_RUNS = {
    "iris": "iris_full",
    "wine": "wine_full",
    "cancer": "cancer_full",
    "mnist-mlp": "mnist_mlp",
    "mnist-cnn": "mnist_cnn",
}
LANDSCAPE_DS = {  # dataset 表示名 -> ls2_ ディレクトリの接頭辞
    "iris": "ls2_iris",
    "wine": "ls2_wine",
    "cancer": "ls2_cancer",
    "mnist-mlp": "ls2_mnistmlp",
    "mnist-cnn": "ls2_mnistcnn",
}
FRACS = [0.5, 0.8, 0.9, 0.95]


# ---------------------------------------------------------------- Part 1

def steps_to(steps, losses, target):
    """train loss が target に到達する最初のステップ (線形補間)。未到達は None。"""
    for i, loss in enumerate(losses):
        if loss <= target:
            if i == 0:
                return float(steps[0])
            prev_l, prev_s = losses[i - 1], steps[i - 1]
            if prev_l > target and loss < prev_l:
                t = (prev_l - target) / (prev_l - loss)
                return float(prev_s + t * (steps[i] - prev_s))
            return float(steps[i])
    return None


def early_convergence():
    """データセットごとのマイルストーン到達ステップと speedup を計算。"""
    out = {}
    for ds, run in COMPARISON_RUNS.items():
        d = json.load(open(RESULTS / run / "metrics.json"))
        hists = d["histories"]
        loss0 = max(h["train_loss"][0] for h in hists.values())
        loss_star = min(min(h["train_loss"]) for h in hists.values())

        milestones = {}
        for f in FRACS:
            target = loss0 - f * (loss0 - loss_star)
            reach = {name: steps_to(h["steps"], h["train_loss"], target)
                     for name, h in hists.items()}
            speedup = {}
            for e in [n for n in hists if n.startswith("eagle")]:
                for b in ["adam", "sgd"]:
                    if reach.get(e) and reach.get(b):
                        speedup[f"{e}_vs_{b}"] = reach[b] / reach[e]
                    else:
                        speedup[f"{e}_vs_{b}"] = None
            milestones[f] = {"target_loss": target, "steps": reach,
                             "speedup": speedup}

        usage = {}
        for name, h in hists.items():
            r = [x for x in h["eagle_ratio"] if x is not None]
            if r:
                k = max(1, len(r) // 4)
                usage[name] = {"first": r[0], "early_mean": float(np.mean(r[:k])),
                               "final": r[-1]}
        out[ds] = {"loss0": loss0, "loss_star": loss_star,
                   "milestones": milestones, "eagle_usage": usage}
    return out


# ---------------------------------------------------------------- Part 2

def slice_metrics(entry, param_range):
    values = np.asarray(entry["param_values"])
    losses = np.asarray(entry["losses"])
    h = values[1] - values[0]
    rng = losses.max() - losses.min()
    ln = (losses - losses.min()) / (rng + 1e-12)

    d1 = np.diff(ln)
    ruggedness = float(np.abs(d1).sum())

    # 局所解の個数: 傾きの符号列 (ノイズ除去のため閾値付き) の -→+ 遷移数
    eps = 1e-3
    signs = np.sign(d1) * (np.abs(d1) > eps)
    signs = signs[signs != 0]
    n_minima = int(((signs[:-1] < 0) & (signs[1:] > 0)).sum()) if len(signs) > 1 else 0
    if len(signs) and signs[0] > 0:
        n_minima += 1  # 左端が谷

    d2 = np.diff(losses, 2)
    convexity = float((d2 >= 0).mean())

    c = int(np.argmin(np.abs(values - entry["original_value"])))
    c = min(max(c, 1), len(losses) - 2)
    curvature = float((losses[c - 1] - 2 * losses[c] + losses[c + 1]) / h ** 2)

    flat_frac = float((ln <= 0.05).mean())
    dist_to_min = float(abs(values[np.argmin(losses)] - entry["original_value"])
                        / param_range)

    return dict(ruggedness=ruggedness, n_minima=n_minima, convexity=convexity,
                curvature=curvature, flat_frac=flat_frac, dist_to_min=dist_to_min)


def landscape_metrics():
    """ls2_* の各 run について、全スライスの形状指標の平均を計算。"""
    out = {}
    for ds, prefix in LANDSCAPE_DS.items():
        for run_dir in sorted(RESULTS.glob(f"{prefix}_*")):
            opt = run_dir.name.rsplit("_", 1)[-1]
            path = run_dir / "landscape.json"
            if not path.exists():
                continue
            d = json.load(open(path))
            param_range = d["args"]["param_range"]
            per_slice = [slice_metrics(e, param_range)
                         for entries in d["landscape"].values()
                         for e in entries]
            agg = {}
            for key in per_slice[0]:
                vals = np.array([m[key] for m in per_slice], dtype=float)
                agg[key] = float(vals.mean())
                agg[key + "_std"] = float(vals.std())
            agg["n_slices"] = len(per_slice)
            agg["baseline_loss"] = d["baseline_loss"]
            out.setdefault(ds, {})[opt] = agg
    return out


# ---------------------------------------------------------------- Part 3

def plot_early_curves(conv):
    fig, axes = plt.subplots(1, len(COMPARISON_RUNS), figsize=(22, 4))
    for ax, (ds, run) in zip(axes, COMPARISON_RUNS.items()):
        d = json.load(open(RESULTS / run / "metrics.json"))
        for name, h in d["histories"].items():
            if name in ("eagle", "eagle-orig", "adam", "sgd"):
                ax.plot(h["steps"], h["train_loss"], label=name, linewidth=1.2)
        for f in [0.8, 0.95]:
            ax.axhline(conv[ds]["milestones"][f]["target_loss"],
                       color="gray", linestyle=":", alpha=0.7)
        ax.set_yscale("log")
        ax.set_title(ds)
        ax.set_xlabel("Steps")
        ax.grid(True, linestyle=":", alpha=0.4)
    axes[0].set_ylabel("Train Loss (log)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Early convergence (dotted: 80% / 95% progress)")
    fig.tight_layout()
    fig.savefig(OUT / "early_convergence.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_speed_vs_shape(conv, land):
    """speedup (f=0.8, eagle) と地形指標の散布図。"""
    metrics = ["convexity", "ruggedness", "n_minima", "flat_frac"]
    fig, axes = plt.subplots(2, len(metrics), figsize=(5 * len(metrics), 9))
    for row, baseline in enumerate(["sgd", "adam"]):
        for col, met in enumerate(metrics):
            ax = axes[row][col]
            for ds in COMPARISON_RUNS:
                if ds not in land or "eagle" not in land[ds]:
                    continue
                sp = conv[ds]["milestones"][0.8]["speedup"].get(f"eagle_vs_{baseline}")
                if sp is None:
                    continue
                x = land[ds]["eagle"][met]
                ax.scatter(x, sp, s=60)
                ax.annotate(ds, (x, sp), fontsize=8,
                            xytext=(4, 4), textcoords="offset points")
            ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
            ax.set_xlabel(met)
            ax.set_ylabel(f"speedup vs {baseline} (f=0.8)")
            ax.grid(True, linestyle=":", alpha=0.4)
    fig.suptitle("Early speedup (f=0.8) vs loss-landscape shape metrics")
    fig.tight_layout()
    fig.savefig(OUT / "speed_vs_shape.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_example_slices(land_dirs=("ls2_iris_eagle", "ls2_wine_eagle",
                                   "ls2_cancer_eagle", "ls2_mnistmlp_eagle",
                                   "ls2_mnistcnn_eagle")):
    """各データセットのスライス例 (正規化) を並べて形状を目視比較できる図。"""
    fig, axes = plt.subplots(len(land_dirs), 6, figsize=(18, 3 * len(land_dirs)))
    for row, run in enumerate(land_dirs):
        path = RESULTS / run / "landscape.json"
        if not path.exists():
            continue
        d = json.load(open(path))
        entries = [e for v in d["landscape"].values() for e in v]
        idx = np.linspace(0, len(entries) - 1, 6).astype(int)
        for col, i in enumerate(idx):
            e = entries[i]
            v = np.asarray(e["param_values"])
            l = np.asarray(e["losses"])
            ln = (l - l.min()) / (l.max() - l.min() + 1e-12)
            ax = axes[row][col]
            ax.plot(v - e["original_value"], ln, "b-", linewidth=1)
            ax.axvline(0, color="r", linestyle="--", alpha=0.5)
            ax.set_ylim(-0.05, 1.05)
            ax.tick_params(labelsize=6)
            if col == 0:
                ax.set_ylabel(run.replace("ls2_", "").replace("_eagle", ""),
                              fontsize=10)
            ax.grid(True, linestyle=":", alpha=0.4)
    fig.suptitle("Example landscape slices (normalized, x = offset from trained value)")
    fig.tight_layout()
    fig.savefig(OUT / "example_slices.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)

    conv = early_convergence()
    land = landscape_metrics()

    with open(OUT / "analysis.json", "w") as f:
        json.dump({"early_convergence": conv, "landscape_metrics": land},
                  f, indent=2)

    # ---- 表: 初期収束 speedup ----
    print("=" * 100)
    print("Part 1: 学習初期の収束速度 (進捗率 f への到達ステップ比, >1 で EAGLE が速い)")
    print("=" * 100)
    header = f"{'dataset':<11}" + "".join(
        f"{'vs adam f=' + str(f):>15}" for f in FRACS) + "".join(
        f"{'vs sgd f=' + str(f):>14}" for f in FRACS)
    for variant in ["eagle", "eagle-orig"]:
        print(f"\n[{variant}]")
        print(header)
        for ds in COMPARISON_RUNS:
            row = f"{ds:<11}"
            for b in ["adam", "sgd"]:
                for f in FRACS:
                    sp = conv[ds]["milestones"][f]["speedup"].get(f"{variant}_vs_{b}")
                    row += f"{('x%.2f' % sp) if sp else '---':>14} "
            print(row)

    print("\nEAGLE 更新使用率 (early = 序盤 1/4 の平均):")
    for ds in COMPARISON_RUNS:
        u = conv[ds]["eagle_usage"]
        row = f"{ds:<11}"
        for name in ["eagle", "eagle-orig"]:
            if name in u:
                row += (f"  {name}: first={u[name]['first']:.2f} "
                        f"early={u[name]['early_mean']:.2f} "
                        f"final={u[name]['final']:.2f}")
        print(row)

    # ---- 表: 地形指標 ----
    print("\n" + "=" * 100)
    print("Part 2: 損失地形の形状指標 (スライス平均)")
    print("=" * 100)
    print(f"{'dataset':<11}{'opt':<7}{'convexity':>10}{'ruggedness':>11}"
          f"{'n_minima':>9}{'curvature':>11}{'flat_frac':>10}{'dist2min':>9}")
    for ds, opts in land.items():
        for opt, m in opts.items():
            print(f"{ds:<11}{opt:<7}{m['convexity']:>10.3f}{m['ruggedness']:>11.2f}"
                  f"{m['n_minima']:>9.2f}{m['curvature']:>11.3f}"
                  f"{m['flat_frac']:>10.3f}{m['dist_to_min']:>9.3f}")

    plot_early_curves(conv)
    plot_speed_vs_shape(conv, land)
    plot_example_slices()
    print(f"\n図と JSON を保存: {OUT}")


if __name__ == "__main__":
    main()
