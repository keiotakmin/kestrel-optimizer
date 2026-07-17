"""REPORT_ablation_regime.md の図を生成する。

abl_milestones.png: データセット 4 面 × バリアント (系譜の固定順) の
マイルストーン到達 speedup (エンベロープ比 vs tuned adam、steps 基準、
train_eval_loss)。f = 0.8 / 0.9 / 0.95 を順序ランプの 3 点で示す。
1.0 の水平線 = tuned adam と同速。エラーバー = 3 シードの SD。

実行: python experiments/plot_ablation_regime.py
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
KEY = "train_eval_loss"
DATASETS = ["california", "concrete", "energy"]
PREFIXES = ("proto", "protoc", "protol", "protoa", "protob", "protod")

# 系譜の固定順 (左 = 古い)。物語の核となる 8 変種のみ
VARIANTS = ["eagle-orig", "eagle2", "eagle3", "eagle4", "eagle4-m",
            "eagle4-noins", "eagle4-aj", "eagle-dqn"]
FRACS = [0.8, 0.9, 0.95]
# f (進捗率) は順序量 → ブルーの順序ランプ (plot_regression_report と共通)
FRAC_COLORS = {0.8: "#5598e7", 0.9: "#256abf", 0.95: "#104281"}

SURFACE = "#fcfcfb"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"


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


def _reach(h, target):
    losses, xs = h[KEY], h["steps"]
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


def style_axis(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, axis="y")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.yaxis.label.set_color(INK2)
    ax.title.set_color(INK)


def main():
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), sharey=True)
    fig.patch.set_facecolor(SURFACE)

    for ax, ds in zip(axes.flat, DATASETS):
        data = collect(ds)
        present = [v for v in VARIANTS if v in data]
        for j, f in enumerate(FRACS):
            means, sds, xs_pos = [], [], []
            for i, fam in enumerate(present):
                ratios = []
                for s in SEEDS:
                    all_h = [seeds[s] for fm in data
                             for lr, seeds in data[fm].items() if s in seeds]
                    loss0 = max(h[KEY][0] for h in all_h)
                    lstar = min(min(h[KEY]) for h in all_h)
                    target = loss0 - f * (loss0 - lstar)

                    def env(family):
                        vals = [_reach(seeds[s], target)
                                for lr, seeds in data[family].items()
                                if s in seeds]
                        vals = [v for v in vals if v]
                        return min(vals) if vals else None

                    a, e = env("adam"), env(fam)
                    ratios.append(a / e if (a and e) else None)
                ok = [r for r in ratios if r is not None]
                if not ok:
                    continue
                means.append(np.mean(ok))
                sds.append(np.std(ok))
                xs_pos.append(i + (j - 1) * 0.22)
            ax.errorbar(xs_pos, means, yerr=sds, fmt="o", markersize=5,
                        color=FRAC_COLORS[f], ecolor=FRAC_COLORS[f],
                        elinewidth=1, capsize=2, label=f"f={f}", zorder=3)
        ax.axhline(1.0, color=AXIS, linewidth=1, zorder=2)
        ax.set_xticks(range(len(present)))
        ax.set_xticklabels(present, rotation=28, ha="right", color=INK2)
        ax.set_title(ds, fontsize=10.5)
        style_axis(ax)
    axes[0].set_ylabel("Reach speedup vs tuned adam\n(steps basis, envelope)")
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK2, ncols=3,
                   loc="upper left")
    fig.suptitle(
        "Mechanism ablation in the surviving regime "
        "(full-batch regression, mean ± SD over 3 seeds)",
        fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = OUT / "abl_milestones.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
