"""REPORT_regression.md の図を生成する。

- reg_<dataset>.png: 家族エンベロープ代表 (最終床が最良の lr) の学習曲線。
  縦軸は best-so-far (その時点までの最良 train loss)。高 lr のフルバッチ
  曲線は振動が激しく生値では読めないため。マイルストーン分析 (最初の到達
  ステップ) と意味が一致する。
  左 = ステップ基準、右 = wall-clock 基準 (L-BFGS の line search コストが見える)。
  線 = 3 シード平均、帯 = シード間 min–max。y は log。
- reg_floors.png: 最終到達床のエンベロープ比 (adam = 1.0、シードごとの
  エンベロープ比の mean±SD)。

実行: python experiments/plot_regression_report.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT = RESULTS / "analysis"
SEEDS = (42, 43, 44)
DATASETS = ["california", "concrete", "energy"]
KEY = "train_eval_loss"

# 系列色: リファレンスパレットのカテゴリカル スロット 1〜4 (固定順)
COLORS = {"adam": "#2a78d6", "eagle4": "#1baf7a",
          "eagle4-m": "#eda100", "lbfgs": "#008300"}
FAMILIES = ["adam", "eagle4", "eagle4-m", "lbfgs"]

# チャート基調色 (光沢のない紙面調)
SURFACE = "#fcfcfb"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"


# 後の prefix が同一セルを上書きする (protoc = clean_every_step 版の eagle 系、
# protol = grad_evals 記録版の lbfgs。予算公平化後のデータが正)
PREFIXES = ("proto", "protoc", "protol")


def collect(dataset):
    """{family: {lr: {seed: history}}}"""
    data = {}
    for prefix in PREFIXES:
        for path in sorted(
                RESULTS.glob(f"{prefix}_{dataset}_lr*_s*/metrics.json")):
            body = path.parent.name[len(f"{prefix}_{dataset}_lr"):]
            lr, seed = body.rsplit("_s", 1)
            with open(path) as f:
                d = json.load(f)
            for opt, h in d["histories"].items():
                data.setdefault(opt, {}).setdefault(lr, {})[int(seed)] = h
    return data


def best_lr(cells):
    """家族内で最終床 (シード平均の min loss) が最良の lr を返す。"""
    return min(cells, key=lambda lr: np.mean(
        [min(h[KEY]) for h in cells[lr].values()]))


def style_axis(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=1.0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    ax.title.set_color(INK)


def plot_curves(dataset):
    data = collect(dataset)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.patch.set_facecolor(SURFACE)

    picked = {}
    for fam in FAMILIES:
        if fam not in data:
            continue
        lr = best_lr(data[fam])
        picked[fam] = lr
        seeds = data[fam][lr]
        # 3 シードで記録グリッドは共通 (同一設定) — 最短長に切り揃える
        n = min(len(h[KEY]) for h in seeds.values())
        # best-so-far に変換 (単調減少・マイルストーン分析と同じ意味)
        losses = np.minimum.accumulate(
            np.array([h[KEY][:n] for h in seeds.values()]), axis=1)
        steps = next(iter(seeds.values()))["steps"][:n]
        times = np.array([h["time"][:n] for h in seeds.values()]).mean(0)

        for ax, x in ((axes[0], steps), (axes[1], times)):
            x = np.asarray(x, dtype=float)
            mask = x > 0 if ax is axes[1] else np.ones_like(x, bool)
            label = f"{fam} (lr={lr})"
            ax.plot(x[mask], losses.mean(0)[mask], color=COLORS[fam],
                    linewidth=2, solid_capstyle="round", label=label)
            ax.fill_between(x[mask], losses.min(0)[mask], losses.max(0)[mask],
                            color=COLORS[fam], alpha=0.10, linewidth=0)

    axes[0].set_xlabel("Steps")
    axes[1].set_xlabel("Wall-clock time (s)")
    axes[1].set_xscale("log")
    for ax in axes:
        ax.set_yscale("log")
        ax.set_ylabel("Best train loss so far (fixed eval subset)")
        style_axis(ax)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK2)
    fig.suptitle(
        f"{dataset}: full-batch MSE regression, tanh MLP "
        "(family-best lr, mean of 3 seeds, band = min–max)",
        fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = OUT / f"reg_{dataset}.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}  (picked lr: {picked})")


def plot_floors():
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    fig.patch.set_facecolor(SURFACE)

    fams = ["eagle4", "eagle4-m", "lbfgs"]
    width = 0.16
    for j, fam in enumerate(fams):
        means, sds = [], []
        for ds in DATASETS:
            data = collect(ds)
            ratios = []
            for s in SEEDS:
                fam_floor = min(min(data[fam][lr][s][KEY])
                                for lr in data[fam] if s in data[fam][lr])
                adam_floor = min(min(data["adam"][lr][s][KEY])
                                 for lr in data["adam"] if s in data["adam"][lr])
                ratios.append(fam_floor / adam_floor)
            means.append(np.mean(ratios))
            sds.append(np.std(ratios))
        xs = np.arange(len(DATASETS)) + (j - 1) * (width + 0.04)
        ax.bar(xs, means, width, color=COLORS[fam], label=fam, zorder=3)
        ax.errorbar(xs, means, yerr=sds, fmt="none", ecolor=INK2,
                    elinewidth=1, capsize=2, zorder=4)
        for x, m, sd in zip(xs, means, sds):
            ax.text(x, m + sd + 0.035, f"{m:.2f}", ha="center", fontsize=8.5,
                    color=INK2)

    ax.axhline(1.0, color=AXIS, linewidth=1, zorder=2)
    ax.text(0.5, 1.035, "adam envelope = 1.0", fontsize=8.5,
            color=MUTED, ha="center")
    ax.set_xticks(np.arange(len(DATASETS)))
    ax.set_xticklabels(DATASETS)
    ax.set_ylabel("Floor ratio vs tuned Adam\n(lower is better)")
    ax.set_ylim(0, 1.30)
    style_axis(ax)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, ncols=3,
              loc="upper center", bbox_to_anchor=(0.5, 1.02))
    ax.set_title("Envelope-vs-envelope final floors (per-seed ratio, mean ± SD)",
                 fontsize=11, pad=28)
    fig.tight_layout()
    out = OUT / "reg_floors.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


FRACS = [0.5, 0.8, 0.9, 0.95]
# f (進捗率) は順序量 → ブルーの順序ランプ (ordinal、浅→深)
FRAC_COLORS = {0.5: "#86b6ef", 0.8: "#5598e7", 0.9: "#256abf", 0.95: "#104281"}


def _xs(h, x_key, family):
    """到達比較の横軸系列。grad_evals 未記録の旧ランは lbfgs 以外なら
    steps で代用できる (lbfgs のみ line search で steps ≠ grad_evals)。"""
    if x_key == "grad_evals":
        g = h.get("grad_evals")
        if g is None:
            return h["steps"] if family != "lbfgs" else None
        return g
    return h[x_key]


def _reach(h, key, target, xs):
    losses = h[key]
    for i, loss in enumerate(losses):
        if loss <= target:
            if i == 0:
                return float(xs[0]) or None
            prev_l, prev_x = losses[i - 1], xs[i - 1]
            if prev_l > target and loss < prev_l:
                frac = (prev_l - target) / (prev_l - loss)
                return prev_x + frac * (xs[i] - prev_x)
            return float(xs[i])
    return None


def plot_milestones(fam="eagle4", baseline="adam",
                    bases=(("steps", "steps basis"),
                           ("time", "wall-clock basis")),
                    outname="reg_milestones.png"):
    """中終盤マイルストーンのエンベロープ比 (vs baseline)。等予算の主結果。"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    fig.patch.set_facecolor(SURFACE)

    for ax, (x_key, label) in zip(axes, bases):
        width = 0.17
        for j, f in enumerate(FRACS):
            means, sds = [], []
            for ds in DATASETS:
                data = collect(ds)
                ratios = []
                for s in SEEDS:
                    # 共通の loss0/loss* (このシードの全セル)
                    all_h = [seeds[s] for fm in data for lr, seeds
                             in data[fm].items() if s in seeds]
                    loss0 = max(h[KEY][0] for h in all_h)
                    lstar = min(min(h[KEY]) for h in all_h)
                    target = loss0 - f * (loss0 - lstar)

                    def env(family):
                        vals = []
                        for lr, seeds in data[family].items():
                            if s not in seeds:
                                continue
                            xs = _xs(seeds[s], x_key, family)
                            if xs is not None:
                                vals.append(
                                    _reach(seeds[s], KEY, target, xs))
                        vals = [v for v in vals if v]
                        return min(vals) if vals else None
                    a, e = env(baseline), env(fam)
                    ratios.append(a / e if (a and e) else None)
                ok = [r for r in ratios if r is not None]
                means.append(np.mean(ok) if len(ok) == len(SEEDS) else np.nan)
                sds.append(np.std(ok) if len(ok) == len(SEEDS) else 0)
            xs = np.arange(len(DATASETS)) + (j - 1.5) * (width + 0.03)
            ax.bar(xs, means, width, color=FRAC_COLORS[f], label=f"f={f}",
                   zorder=3)
            ax.errorbar(xs, means, yerr=sds, fmt="none", ecolor=INK2,
                        elinewidth=1, capsize=2, zorder=4)
        ax.axhline(1.0, color=AXIS, linewidth=1, zorder=2)
        ax.set_xticks(np.arange(len(DATASETS)))
        ax.set_xticklabels(DATASETS)
        ax.set_ylabel(f"Reach speedup vs tuned {baseline}\n"
                      "(envelope vs envelope)")
        ax.set_title(label, fontsize=10)
        style_axis(ax)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK2, ncols=4,
                   loc="lower center", bbox_to_anchor=(0.5, 1.06))
    fig.suptitle(
        f"{fam} vs tuned {baseline}: milestone reach speedup "
        "(mean ± SD, 3 seeds)", fontsize=11, color=INK, y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = OUT / outname
    fig.savefig(out, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    for ds in DATASETS:
        plot_curves(ds)
    plot_floors()
    plot_milestones()
    # vs L-BFGS: steps 基準は line search の複数評価を数えないため不公平。
    # 公平な基準 = 勾配評価数 (grad_evals) と wall-clock
    plot_milestones(baseline="lbfgs",
                    bases=(("grad_evals", "gradient-evaluations basis"),
                           ("time", "wall-clock basis")),
                    outname="reg_milestones_vs_lbfgs.png")
