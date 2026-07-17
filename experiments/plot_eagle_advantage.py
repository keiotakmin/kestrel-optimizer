"""EAGLE の優位性を示す旗艦図 2 枚を生成する (2026-07-13 再アブレーション後、
チャンピオン = eagle3)。

- adv_regression.png: フルバッチ回帰 4 データセット。
  左 = eagle3 vs tuned Adam のマイルストーン到達 speedup (steps 基準、
  等予算、f=0.8/0.9/0.95)。右 = eagle3 vs tuned L-BFGS (勾配評価数基準 =
  line search の隠れコストを数える装置非依存の公平な予算)。
  どちらも家族 lr エンベロープ同士、3 シード mean±SD。
- adv_inr.png: INR パイロット (SIREN 512², フルバッチ 262k 画素、
  ~200k params)。best-so-far PSNR vs steps (家族最良 lr)。
  L-BFGS がこのサイズ帯で失速し、eagle3 が tuned Adam を上回る様子。

実行: python experiments/plot_eagle_advantage.py
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
FRACS = [0.8, 0.9, 0.95]
FRAC_COLORS = {0.8: "#5598e7", 0.9: "#256abf", 0.95: "#104281"}

# 系列色 (固定順のカテゴリカル: 青 / 緑 / 琥珀 / 灰)
C_ADAM = "#2a78d6"
C_EAGLE = "#1baf7a"
C_LBFGS = "#eda100"
C_DQN = "#898781"

SURFACE = "#fcfcfb"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"


def collect(dataset):
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


def _xs(h, x_key, family):
    if x_key == "grad_evals":
        g = h.get("grad_evals")
        if g is None:
            return h["steps"] if family != "lbfgs" else None
        return g
    return h[x_key]


def _reach(h, target, xs):
    losses = h[KEY]
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
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.label.set_color(INK2)
    ax.title.set_color(INK)


def speedup_ratios(data, fam, baseline, x_key):
    """f ごとの [seed ごとの envelope 比 (baseline 到達 / fam 到達)]。"""
    out = {f: [] for f in FRACS}
    for s in SEEDS:
        all_h = [seeds[s] for fm in data
                 for lr, seeds in data[fm].items() if s in seeds]
        loss0 = max(h[KEY][0] for h in all_h)
        lstar = min(min(h[KEY]) for h in all_h)
        for f in FRACS:
            target = loss0 - f * (loss0 - lstar)

            def env(family):
                vals = []
                for lr, seeds in data[family].items():
                    if s not in seeds:
                        continue
                    xs = _xs(seeds[s], x_key, family)
                    if xs is not None:
                        vals.append(_reach(seeds[s], target, xs))
                vals = [v for v in vals if v]
                return min(vals) if vals else None

            b, e = env(baseline), env(fam)
            out[f].append(b / e if (b and e) else None)
    return out


PANELS_BUDGET = [
    ("adam", "steps", "vs tuned Adam  (steps basis, equal budget)"),
    ("lbfgs", "grad_evals", "vs tuned L-BFGS  (gradient-evaluations basis)")]
PANELS_TIME = [
    ("adam", "time", "vs tuned Adam  (wall-clock basis)"),
    ("lbfgs", "time", "vs tuned L-BFGS  (wall-clock basis)")]


FAMS_ADV = [("eagle3", C_EAGLE), ("eagle-dqn-cd", "#7c5cd6")]


def _shade(base, pos):
    """マイルストーンの深さ (0/1/2 = 浅い/中間/深い) を明度にエンコード。"""
    import matplotlib.colors as mcolors
    rgb = np.array(mcolors.to_rgb(base))
    if pos == 0:
        rgb = rgb + (1 - rgb) * 0.45   # 白へ 45%
    elif pos == 2:
        rgb = rgb * 0.65               # 黒へ 35%
    return tuple(rgb)


def plot_regression(panels=PANELS_BUDGET, outname="adv_regression.png",
                    note=None):
    """回帰の旗艦図: eagle3 と eagle-dqn-cd (統一既定) を並べて、
    マイルストーン f=0.8/0.9/0.95 の到達 speedup を示す。
    色は他の図と同じ系列恒等 (緑 = eagle3、紫 = eagle-dqn-cd)。"""
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.4), sharey=False)
    fig.patch.set_facecolor(SURFACE)

    group_w, width = 4.0, 0.38
    for ax, (baseline, x_key, title) in zip(axes, panels):
        xticks, xticklabels = [], []
        for di, ds in enumerate(DATASETS):
            data = collect(ds)
            ratios_by_fam = {fam: speedup_ratios(data, fam, baseline, x_key)
                             for fam, _ in FAMS_ADV}
            for fi, f in enumerate(FRACS):
                x0 = di * group_w + fi
                xticks.append(x0)
                xticklabels.append(f"{f}")
                for j, (fam, color) in enumerate(FAMS_ADV):
                    ok = [r for r in ratios_by_fam[fam][f] if r is not None]
                    if not ok:
                        continue
                    m, sd = np.mean(ok), np.std(ok)
                    x = x0 + (j - 0.5) * (width + 0.04)
                    ax.bar(x, m, width, color=_shade(color, fi), zorder=3,
                           label=f"{fam} (f=0.8→0.95)"
                           if (di == 0 and fi == 1) else None)
                    ax.errorbar(x, m, yerr=sd, fmt="none", ecolor=INK2,
                                elinewidth=1, capsize=2, zorder=4)
                    ax.text(x, 0.07, f"{m:.2f}", ha="center", va="bottom",
                            fontsize=7, color="#ffffff", zorder=5,
                            rotation=90)
            # データセット名 (f ラベルの下の 2 段目)
            ax.text(di * group_w + 1, -0.14, ds, ha="center", fontsize=9.5,
                    color=INK2, transform=ax.get_xaxis_transform())
        ax.axhline(1.0, color=AXIS, linewidth=1, zorder=2)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, fontsize=8)
        ax.tick_params(axis="x", length=0)
        ax.set_title(title, fontsize=10.5)
        style_axis(ax)
    axes[0].set_ylabel("Reach speedup at milestone f\n"
                       "(family-envelope ratio)")
    axes[0].text(0.02, 0.94, "1.0 = same speed as the tuned baseline",
                 transform=axes[0].transAxes, fontsize=8.5, color=MUTED)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, labelcolor=INK2,
               ncols=2, loc="upper right", bbox_to_anchor=(0.995, 0.965))
    fig.suptitle(
        "EAGLE on full-batch regression: eagle-dqn-cd (unified default) and "
        "eagle3 reach mid-to-late milestones faster than tuned baselines "
        "(3 seeds, mean ± SD)",
        fontsize=11.5, color=INK)
    if note:
        fig.text(0.5, 0.005, note, ha="center", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03 if note else 0, 1, 0.93))
    out = OUT / outname
    fig.savefig(out, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


REG_FAMS = [("adam", C_ADAM, 2.0),
            ("eagle3", C_EAGLE, 2.2),
            ("eagle-dqn-cd", "#7c5cd6", 2.2),
            ("lbfgs", C_LBFGS, 2.0)]


def plot_regression_curves():
    """INR 図と同形式の回帰学習曲線: データセット 3 面 × 4 手法。
    y = best-so-far train_eval_loss (log)、x = 勾配評価数
    (lbfgs 以外は = steps。lbfgs は line search の評価を含む実測値で、
    等予算 = 非 lbfgs の総ステップ数で切り揃える)。
    線 = 家族最良 lr (最終床基準) の 3 シード平均、帯 = min–max。"""
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.4))
    fig.patch.set_facecolor(SURFACE)

    for ax, ds in zip(axes, DATASETS):
        data = collect(ds)
        # マイルストーン標的 (analyze_protocol と同じ per-seed 定義)
        seed_targets = {f: {} for f in FRACS}
        for s in SEEDS:
            all_h = [seeds[s] for fm in data
                     for lr, seeds in data[fm].items() if s in seeds]
            loss0 = max(h[KEY][0] for h in all_h)
            lstar = min(min(h[KEY]) for h in all_h)
            for f in FRACS:
                seed_targets[f][s] = loss0 - f * (loss0 - lstar)
        targets = {f: np.mean(list(seed_targets[f].values())) for f in FRACS}
        # 等予算 = 非 lbfgs のステップ数 (lbfgs はこの評価数で切り揃える)
        budget = max(h["steps"][-1] for cells in data["adam"].values()
                     for h in cells.values())
        grid = np.geomspace(1, budget, 400)
        for fam, color, lw in REG_FAMS:
            if fam not in data:
                continue
            # 家族 lr エンベロープ曲線: シードごとに全 lr セルの
            # best-so-far loss の各時点最小 (共通 log グリッドに補間して
            # 合成)。棒グラフ (adv_regression) と同じ「家族の最善」の軌跡
            curves = []
            for s in SEEDS:
                cell_curves = []
                for lr2, cells in data[fam].items():
                    if s not in cells:
                        continue
                    h = cells[s]
                    xc = _xs(h, "grad_evals", fam)
                    if xc is None:
                        continue
                    xs = np.asarray(xc, dtype=float)
                    ys = np.minimum.accumulate(np.asarray(h[KEY]))
                    mask = xs >= 1
                    cell_curves.append(np.interp(grid, xs[mask], ys[mask]))
                if cell_curves:
                    curves.append(np.array(cell_curves).min(0))
            if not curves:
                continue
            curves = np.array(curves)
            mean = curves.mean(0)
            ax.plot(grid, mean, color=color, linewidth=lw,
                    solid_capstyle="round", label=f"{fam} (lr envelope)")
            ax.fill_between(grid, curves.min(0), curves.max(0), color=color,
                            alpha=0.12, linewidth=0)
            # ドット = 平均エンベロープ曲線が標的水準を初めて下回る点
            for f in FRACS:
                t = targets[f]
                below = np.flatnonzero(mean <= t)
                if len(below) == 0:
                    continue
                i = below[0]
                if i == 0:
                    x_cross = grid[0]
                else:
                    frac = (mean[i - 1] - t) / (mean[i - 1] - mean[i])
                    x_cross = grid[i - 1] + frac * (grid[i] - grid[i - 1])
                ax.plot(x_cross, t, "o", markersize=5.5, color=color,
                        markeredgecolor="#ffffff", markeredgewidth=0.8,
                        zorder=5, clip_on=False)
        for f in FRACS:
            ax.axhline(targets[f], color=GRID, linewidth=0.8, zorder=1)
            ax.text(budget * 0.95, targets[f], f"f={f}", fontsize=7.5,
                    color=MUTED, va="bottom", ha="right")
        ax.set_xscale("log")
        ax.set_xlim(1, budget)
        ax.set_yscale("log")
        ax.set_xlabel("Gradient evaluations (= steps for all but L-BFGS, "
                      "log)", color=INK2, fontsize=9)
        ax.set_title(ds, fontsize=10.5)
        style_axis(ax)
        ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2,
                  loc="upper right")
    axes[0].set_ylabel("Best train loss so far\n(fixed eval subset, log)")
    fig.suptitle(
        "Full-batch regression at equal gradient-evaluation budget — "
        "curves = family lr-envelope (pointwise best-so-far over the lr "
        "grid; same statistic as adv_regression.png), mean of 3 seeds, "
        "band = min–max; dots = first crossing of the f=0.8/0.9/0.95 levels",
        fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = OUT / "adv_regression_curves.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


# ---- INR ----

C_DQNCD2 = "#7c5cd6"
INR_FAMS = [("adam", C_ADAM, "-", 2.0),
            ("eagle3", C_EAGLE, "-", 2.2),
            ("eagle-dqn-cd", C_DQNCD2, "-", 2.2),
            ("lbfgs", C_LBFGS, "-", 2.0)]


INR_PREFIXES = ("inrv2", "inrk")  # inrk = κ 安定化変種 (eagle-dqn-cd 等)
INR_SEEDS = (42, 43, 44)
INR_THRS = (28, 30, 32)


def _env_dot(ax, x, thr, color):
    ax.plot(x, thr, "o", markersize=5.5, color=color,
            markeredgecolor="#ffffff", markeredgewidth=0.8, zorder=5,
            clip_on=False)


def _kodak_mean_curves():
    """Kodak 24 枚の平均 lr エンベロープ曲線 {family: (steps, mean)}。
    画像ごとに全 lr セルの best-so-far PSNR の各時点最大 (家族エンベロープ)
    を取り、画像間で平均する。全ランが同一の記録グリッドを持つ前提。"""
    import glob
    per_img = []
    for f in sorted(glob.glob(str(RESULTS / "kodak_kodim*_s42/metrics.json"))):
        h = json.load(open(f))["histories"]
        try:
            h.update(json.load(
                open(f.replace("kodak_", "kodakcd_")))["histories"])
        except FileNotFoundError:
            pass
        per_img.append(h)
    out = {}
    for fam, *_ in INR_FAMS:
        curves, steps = [], None
        for h in per_img:
            cells = [v for k, v in h.items() if k.split("@")[0] == fam]
            if not cells:
                continue
            n = min(len(c["psnr"]) for c in cells)
            env = np.array([np.maximum.accumulate(c["psnr"][:n])
                            for c in cells]).max(0)
            curves.append(env)
            steps = cells[0]["steps"][:n]
        if curves:
            n = min(len(c) for c in curves)
            out[fam] = (np.asarray(steps[:n], dtype=float),
                        np.array([c[:n] for c in curves]).mean(0))
    return len(per_img), out, per_img


def plot_inr():
    images = ["camera", "astronaut"]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.4), sharey=True)
    fig.patch.set_facecolor(SURFACE)

    for ax, img in zip(axes[:2], images):
        # {seed: {"opt@lr": history}} (inrv2 + inrk をマージ)
        per_seed = {}
        for s in INR_SEEDS:
            for prefix in INR_PREFIXES:
                path = RESULTS / f"{prefix}_{img}_s{s}" / "metrics.json"
                if path.exists():
                    with open(path) as f:
                        per_seed.setdefault(s, {}).update(
                            json.load(f)["histories"])
        seeds = sorted(per_seed)
        finals = []
        for fam, color, ls, lw in INR_FAMS:
            # 家族 lr エンベロープ曲線: シードごとに全 lr セルの
            # best-so-far PSNR の各時点最大を取り、シード間で平均する。
            # 棒グラフ (adv_inr_milestones) と同じ「家族の最善」の軌跡
            curves, steps = [], None
            for s in seeds:
                cells = [v for k, v in per_seed[s].items()
                         if k.split("@")[0] == fam]
                if not cells:
                    continue
                n = min(len(c["psnr"]) for c in cells)
                env = np.array([np.maximum.accumulate(c["psnr"][:n])
                                for c in cells]).max(0)
                curves.append(env)
                steps = np.asarray(cells[0]["steps"][:n], dtype=float)
            if not curves:
                continue
            n = min(len(c) for c in curves)
            curves = np.array([c[:n] for c in curves])
            steps = steps[:n]
            mean = curves.mean(0)
            ax.plot(steps, mean, color=color, linestyle=ls, linewidth=lw,
                    solid_capstyle="round", label=f"{fam} (lr envelope)")
            ax.fill_between(steps, curves.min(0), curves.max(0), color=color,
                            alpha=0.12, linewidth=0)
            # ドット = 平均エンベロープ曲線がしきい値を初めて超える点
            # (線形補間、曲線上かつしきい値線上)
            for thr in INR_THRS:
                above = np.flatnonzero(mean >= thr)
                if len(above) == 0:
                    continue
                i = above[0]
                if i == 0:
                    x_cross = steps[0]
                else:
                    frac = (thr - mean[i - 1]) / (mean[i] - mean[i - 1])
                    x_cross = steps[i - 1] + frac * (steps[i] - steps[i - 1])
                _env_dot(ax, x_cross, thr, color)
            finals.append([steps[-1], mean[-1], color])
        # 最終値ラベルの衝突回避 (下から詰めて最小 0.7 dB の間隔を確保)
        finals.sort(key=lambda t: t[1])
        y_prev = None
        for x, y, color in finals:
            y_disp = y if y_prev is None else max(y, y_prev + 0.7)
            y_prev = y_disp
            ax.annotate(f"{y:.1f} dB", (x, y_disp),
                        xytext=(4, 0), textcoords="offset points",
                        fontsize=8, color=color, va="center")
        for thr in INR_THRS:
            ax.axhline(thr, color=GRID, linewidth=0.8, zorder=1)
            ax.text(2280, thr, f"{thr} dB", fontsize=7.5, color=MUTED,
                    va="bottom", ha="right")
        ax.set_xlim(0, 2300)
        ax.set_xlabel("Steps (= gradient evaluations for all but L-BFGS)",
                      color=INK2, fontsize=9)
        ax.set_title(f"{img} 512²", fontsize=10.5)
        style_axis(ax)
        ax.grid(False, axis="x")
    # 第 3 パネル: Kodak 24 枚の平均曲線 (seed 42、画像ごとに家族最良 lr)
    ax = axes[2]
    n_img, kodak, per_img = _kodak_mean_curves()
    finals = []
    for fam, color, ls, lw in INR_FAMS:
        if fam not in kodak:
            continue
        steps, mean = kodak[fam]
        ax.plot(steps, mean, color=color, linestyle=ls, linewidth=lw,
                solid_capstyle="round", label=f"{fam} (lr envelope)")
        for thr in INR_THRS:
            above = np.flatnonzero(mean >= thr)
            if len(above) == 0:
                continue
            i = above[0]
            if i == 0:
                x_cross = steps[0]
            else:
                frac = (thr - mean[i - 1]) / (mean[i] - mean[i - 1])
                x_cross = steps[i - 1] + frac * (steps[i] - steps[i - 1])
            _env_dot(ax, x_cross, thr, color)
        finals.append([steps[-1], mean[-1], color])
    finals.sort(key=lambda t: t[1])
    y_prev = None
    for x, y, color in finals:
        y_disp = y if y_prev is None else max(y, y_prev + 0.7)
        y_prev = y_disp
        ax.annotate(f"{y:.1f} dB", (x, y_disp), xytext=(4, 0),
                    textcoords="offset points", fontsize=8, color=color,
                    va="center")
    for thr in INR_THRS:
        ax.axhline(thr, color=GRID, linewidth=0.8, zorder=1)
        ax.text(2280, thr, f"{thr} dB", fontsize=7.5, color=MUTED,
                va="bottom", ha="right")
    ax.set_xlim(0, 2300)
    ax.set_xlabel("Steps (= gradient evaluations for all but L-BFGS)",
                  color=INK2, fontsize=9)
    ax.set_title(f"Kodak-24 mean 768×512 (n={n_img} images, seed 42)",
                 fontsize=10.5)
    style_axis(ax)
    ax.grid(False, axis="x")

    axes[0].set_ylabel("Best-so-far PSNR (dB)")
    axes[0].set_ylim(20, 35.5)
    for ax in axes:
        ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2,
                  loc="lower right")
    fig.suptitle(
        "INR main experiment (protocol v2): SIREN image fitting "
        "(~200k params, full-batch MSE) — curves = family lr-envelope "
        "(pointwise best over the lr grid; same statistic as "
        "adv_inr_milestones); dots = first crossing of 28/30/32 dB",
        fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = OUT / "adv_inr.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


C_DQNCD = "#7c5cd6"


def plot_kodak():
    """Kodak 24 枚の統計図: 左 = 最終 PSNR の per-image 差 (vs tuned adam)、
    右 = 到達 speedup。データ = kodak_* (adam/eagle3/eagle-dqn/lbfgs) +
    kodakcd_* (eagle-dqn-cd)、seed 42、lr 3 点エンベロープ。"""
    import glob

    def env_best(h, fam):
        cells = [v for k, v in h.items() if k.split("@")[0] == fam]
        return max(max(c["psnr"]) for c in cells) if cells else None

    def env_reach(h, fam, thr):
        best = None
        for k, v in h.items():
            if k.split("@")[0] != fam:
                continue
            for s, p in zip(v["steps"], v["psnr"]):
                if p >= thr:
                    best = s if best is None else min(best, s)
                    break
        return best

    rows = []
    for f in sorted(glob.glob(str(RESULTS / "kodak_kodim*_s42/metrics.json"))):
        h = json.load(open(f))["histories"]
        try:
            h.update(json.load(
                open(f.replace("kodak_", "kodakcd_")))["histories"])
        except FileNotFoundError:
            pass
        rows.append(h)
    n = len(rows)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.4))
    fig.patch.set_facecolor(SURFACE)

    # 左: per-image ΔPSNR (vs tuned adam)
    ax = axes[0]
    fams = [("eagle-dqn-cd", C_DQNCD), ("eagle3", C_EAGLE),
            ("lbfgs", C_LBFGS)]
    rng = np.random.default_rng(0)
    for i, (fam, color) in enumerate(fams):
        d = np.array([env_best(h, fam) - env_best(h, "adam") for h in rows])
        x = i + rng.uniform(-0.13, 0.13, len(d))
        ax.scatter(x, d, s=18, color=color, alpha=0.55, linewidths=0,
                   zorder=3)
        ax.errorbar([i + 0.32], [d.mean()], yerr=[d.std()], fmt="D",
                    markersize=6, color=color, ecolor=color, elinewidth=1.6,
                    capsize=3, zorder=4)
        win = (d > 0).sum()
        ax.text(i, ax.get_ylim()[0] * 0 - 2.55, f"win {win}/{n}",
                ha="center", fontsize=8.5, color=INK2)
    ax.axhline(0, color=AXIS, linewidth=1, zorder=2)
    ax.set_xticks(range(len(fams)))
    ax.set_xticklabels([f for f, _ in fams], color=INK2)
    ax.set_ylabel("Final PSNR − tuned Adam (dB)\nper image")
    ax.set_ylim(-2.8, 2.2)
    ax.set_title(f"Final quality vs tuned Adam (Kodak, n={n})",
                 fontsize=10.5)
    style_axis(ax)

    # 右: 到達 speedup (28/30/32 dB)
    ax = axes[1]
    thrs = [28, 30, 32]
    width = 0.3
    for j, (fam, color) in enumerate([("eagle-dqn-cd", C_DQNCD),
                                      ("eagle3", C_EAGLE)]):
        means, sds, ns = [], [], []
        for thr in thrs:
            sp = []
            for h in rows:
                a, e = env_reach(h, "adam", thr), env_reach(h, fam, thr)
                if a and e:
                    sp.append(a / e)
            means.append(np.mean(sp))
            sds.append(np.std(sp))
            ns.append(len(sp))
        xs = np.arange(len(thrs)) + (j - 0.5) * (width + 0.04)
        ax.bar(xs, means, width, color=color, label=fam, zorder=3)
        ax.errorbar(xs, means, yerr=sds, fmt="none", ecolor=INK2,
                    elinewidth=1, capsize=2, zorder=4)
        for x, m, k in zip(xs, means, ns):
            ax.text(x, 0.06, f"x{m:.2f} (n={k})", ha="center", va="bottom",
                    fontsize=7.5, color="#ffffff", rotation=90, zorder=5)
    ax.axhline(1.0, color=AXIS, linewidth=1, zorder=2)
    ax.set_xticks(np.arange(len(thrs)))
    ax.set_xticklabels([f"{t} dB" for t in thrs], color=INK2)
    ax.set_ylabel("Reach speedup vs tuned Adam\n(steps basis)")
    ax.set_title("Steps to reach PSNR thresholds (envelope ratio)",
                 fontsize=10.5)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
    style_axis(ax)

    fig.suptitle(
        "Kodak-24 SIREN fitting: eagle-dqn-cd (always-jump + cooldown-only) "
        "beats tuned Adam on 22/24 images (+0.59 dB, paired-t p<1e-4)",
        fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = OUT / "adv_kodak.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def _inr_reach(h, fam, thr, basis):
    """家族エンベロープ: fam の全 lr セルで PSNR >= thr の初到達 x の最小。"""
    best = None
    for k, v in h.items():
        if k.split("@")[0] != fam:
            continue
        for x, p in zip(v[basis], v["psnr"]):
            if p >= thr:
                best = x if best is None else min(best, x)
                break
    return best


def plot_inr_milestones():
    """INR 版のマイルストーン到達 speedup (adv_regression.png と同文法)。
    しきい値 = 28/30/32 dB (淡→濃)。camera/astronaut = 3 シード、
    Kodak-24 = 24 枚 (seed 42、SD は画像間)。"""
    import glob
    thrs = [28, 30, 32]

    # データ収集: {group: [per-unit histories]} (unit = seed or image)
    groups = {}
    for img in ("camera", "astronaut"):
        units = []
        for s in INR_SEEDS:
            h = {}
            for prefix in INR_PREFIXES:
                path = RESULTS / f"{prefix}_{img}_s{s}" / "metrics.json"
                if path.exists():
                    h.update(json.load(open(path))["histories"])
            if h:
                units.append(h)
        groups[img] = units
    units = []
    for f in sorted(glob.glob(str(RESULTS / "kodak_kodim*_s42/metrics.json"))):
        h = json.load(open(f))["histories"]
        try:
            h.update(json.load(
                open(f.replace("kodak_", "kodakcd_")))["histories"])
        except FileNotFoundError:
            pass
        units.append(h)
    groups["kodak-24"] = units

    panels = [("adam", "steps", "vs tuned Adam  (steps basis)"),
              ("lbfgs", "grad_evals",
               "vs tuned L-BFGS  (gradient-evaluations basis)")]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.4), sharey=False)
    fig.patch.set_facecolor(SURFACE)

    group_w, width = 4.0, 0.38
    for ax, (baseline, basis, title) in zip(axes, panels):
        xticks, xticklabels = [], []
        for di, (gname, units) in enumerate(groups.items()):
            for ti, thr in enumerate(thrs):
                x0 = di * group_w + ti
                xticks.append(x0)
                xticklabels.append(f"{thr}")
                for j, (fam, color) in enumerate(FAMS_ADV):
                    sp = []
                    for h in units:
                        b = _inr_reach(h, baseline, thr, basis)
                        e = _inr_reach(h, fam, thr, basis)
                        if b and e:
                            sp.append(b / e)
                    if not sp:
                        continue
                    m, sd = np.mean(sp), np.std(sp)
                    x = x0 + (j - 0.5) * (width + 0.04)
                    ax.bar(x, m, width, color=_shade(color, ti), zorder=3,
                           label=f"{fam} (28→32 dB)"
                           if (di == 0 and ti == 1) else None)
                    ax.errorbar(x, m, yerr=sd, fmt="none", ecolor=INK2,
                                elinewidth=1, capsize=2, zorder=4)
                    note = f"x{m:.2f}" + (f" (n={len(sp)})"
                                          if len(sp) < len(units) else "")
                    ax.text(x, 0.12, note, ha="center", va="bottom",
                            fontsize=6.8, color="#ffffff", zorder=5,
                            rotation=90)
            ax.text(di * group_w + 1, -0.14, gname, ha="center",
                    fontsize=9.5, color=INK2,
                    transform=ax.get_xaxis_transform())
        ax.axhline(1.0, color=AXIS, linewidth=1, zorder=2)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, fontsize=8)
        ax.tick_params(axis="x", length=0)
        ax.set_title(title, fontsize=10.5)
        style_axis(ax)
    axes[0].set_ylabel("Reach speedup at PSNR threshold\n"
                       "(family-envelope ratio)")
    axes[0].text(0.02, 0.94, "1.0 = same speed as the tuned baseline",
                 transform=axes[0].transAxes, fontsize=8.5, color=MUTED)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, labelcolor=INK2,
               ncols=2, loc="upper right", bbox_to_anchor=(0.995, 0.965))
    fig.suptitle(
        "INR (SIREN fitting): milestone reach speedup — camera/astronaut = "
        "mean ± SD over 3 seeds, Kodak-24 = mean ± SD over 24 images "
        "(n shown when some units miss the threshold)",
        fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = OUT / "adv_inr_milestones.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    plot_regression()
    # wall-clock 基準版。ステップ時間は CUDA 同期込み・ウォームアップ済みだが、
    # 各グリッドは別ジョブの並列実行下で計測されており資源競合を含む。
    # 正式な予算基準は steps / grad_evals (adv_regression.png)
    plot_regression(
        panels=PANELS_TIME, outname="adv_regression_time.png",
        note="Wall-clock measured under partial GPU contention "
             "(grids ran alongside other jobs); indicative only — "
             "steps / gradient-evaluations basis is authoritative.")
    plot_regression_curves()
    plot_inr()
    plot_inr_milestones()
    plot_kodak()
