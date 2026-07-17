"""ICTAI 2026 論文図の生成 (結果データのみから、GPU 不要)。

- fig_regression.pdf: 回帰 3 データセットの収束曲線 (家族エンベロープ最良 lr、
  3 シード平均、train_eval_loss、log-y)
- fig_inr.pdf: (a) camera / (b) astronaut の PSNR 曲線、(c) Kodak-24 の
  画像ごと散布 (30dB 到達 speedup × ΔPSNR)

色は dataviz スキルの検証済みリファレンスパレット (light 列) をスロット
固定順で使用。同一エンティティは図間で同色 (KESTREL 系 = slot6 橙、
変種は線種/マーカーで区別)。

実行: python ictai/paper_figs.py   (bachelor/ ルートから)
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGS = Path(__file__).resolve().parent / "paper" / "figs"

sys.path.insert(0, str(ROOT / "experiments"))
from analyze_protocol import collect  # noqa: E402

# dataviz 検証済みパレット (light)。slot 固定順、entity に固定割当
C = {
    "adam": "#2a78d6",       # slot1 blue
    "adam-cos": "#008300",   # slot2 green
    "bb-stab": "#e87ba4",    # slot3 magenta
    "adahessian": "#eda100", # slot4 yellow
    "lbfgs": "#1baf7a",      # slot5 aqua
    "kestrel": "#eb6834",    # slot6 orange (ours: dqn-cd / +cos)
    "eagle-arxiv": "#4a3aa7" # slot7 violet
}
INK = "#333333"

plt.rcParams.update({
    "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.4,
    "lines.linewidth": 1.3,
})

SEEDS = (42, 43, 44)
KEY = "train_eval_loss"

# (表示名, 家族名, 色, 線種, 太さ)
REG_SERIES = [
    ("Adam (tuned)", "adam", C["adam"], "-", 1.3),
    ("Adam+cosine", "adam-cos", C["adam-cos"], "-", 1.3),
    ("BB (stabilized)", "bb-stab", C["bb-stab"], "-", 1.3),
    ("AdaHessian", "adahessian", C["adahessian"], "-", 1.3),
    # L-BFGS は line search の勾配評価が steps に現れないため、
    # プロファイルは grad-evals 基準で描く (予算監査ポリシーと整合)
    ("L-BFGS [grad-evals]", "lbfgs", C["lbfgs"], "-", 1.3),
    ("EAGLE (arXiv)", "eagle", C["eagle-arxiv"], "-", 1.3),
    ("KESTREL (ours)", "eagle-dqn-cd", C["kestrel"], "-", 2.0),
]


def mean_curve(hists, key=KEY, x="steps"):
    """記録グリッドが共通である前提でシード平均。"""
    xs = hists[min(hists)][x]
    ys = np.array([h[key] for h in hists.values() if len(h[key]) == len(xs)])
    return np.asarray(xs, float), ys.mean(0)


def x_to(xs, losses, target):
    for i, v in enumerate(losses):
        if v <= target:
            if i == 0:
                return float(xs[0])
            pl, px = losses[i - 1], xs[i - 1]
            if pl > target and v < pl:
                return px + (pl - target) / (pl - v) * (xs[i] - px)
            return float(xs[i])
    return None


def speedup_profile(data, fam, fracs, key=KEY, basis="steps"):
    """進捗率 f ごとの家族エンベロープ到達 speedup vs adam (シード平均)。

    マイルストーン定義は analyze_protocol / gen_macros と同一
    (全セル共通の loss0 / lstar)。basis="grad_evals" は lbfgs 用
    (line search の勾配評価を含む真の予算基準)。"""
    prof = []
    for f in fracs:
        sp = []
        for s in data:
            hists = data[s]
            loss0 = max(h[key][0] for h in hists.values())
            lstar = min(min(h[key]) for h in hists.values())
            tgt = loss0 - f * (loss0 - lstar)

            def env(family, b):
                vals = []
                for c, h in hists.items():
                    if c.split("@")[0] != family:
                        continue
                    xs = h.get(b) or h["steps"]
                    v = x_to(xs, h[key], tgt)
                    if v is not None:
                        vals.append(v)
                return min(vals) if vals else None

            a, e = env("adam", basis), env(fam, basis)
            if a and e:
                sp.append(a / e)
        prof.append(np.mean(sp) if len(sp) == len(data) else np.nan)
    return np.array(prof)


def fig_regression():
    """到達 speedup プロファイル: x = 進捗率 f、y = 対 tuned Adam speedup。
    Table I の主張 (マイルストーン到達の速さ) を全進捗域で可視化する。"""
    dss = [("california", "California"), ("concrete", "Concrete"),
           ("energy", "Energy")]
    fracs = np.concatenate([np.arange(0.50, 0.96, 0.05), [0.97]])
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.1), sharey=True)
    for ax, (ds, title) in zip(axes, dss):
        data = collect(ds, ("protoe", "protoeh", "protoe2", "protoe3"))
        for label, fam, color, ls, lw in REG_SERIES:
            if fam == "adam":
                continue
            basis = "grad_evals" if fam == "lbfgs" else "steps"
            prof = speedup_profile(data, fam, fracs, basis=basis)
            ax.plot(fracs, prof, ls, color=color, lw=lw, label=label,
                    solid_capstyle="round")
        ax.axhline(1.0, color=C["adam"], lw=1.0, ls="-", alpha=0.9)
        ax.text(0.505, 1.04, "tuned Adam", color=C["adam"], fontsize=6.5)
        ax.set_yscale("log", base=2)
        ax.set_yticks([0.25, 0.5, 1, 2])
        ax.set_yticklabels(["0.25×", "0.5×", "1×", "2×"])
        ax.set_ylim(0.12, 2.9)
        ax.set_title(title, color=INK)
        ax.set_xlabel("progress fraction $f$")
    axes[0].set_ylabel("reach speedup vs. tuned Adam")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=6, loc="upper center",
               bbox_to_anchor=(0.5, 1.14), frameon=False,
               columnspacing=1.0, handlelength=1.6)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_regression.pdf", bbox_inches="tight")
    plt.close(fig)


def collect_inr(prefixes, image):
    out = {}
    for prefix in prefixes:
        for path in sorted(RESULTS.glob(f"{prefix}_{image}_s*/metrics.json")):
            seed = int(path.parent.name.rsplit("_s", 1)[1])
            with open(path) as f:
                out.setdefault(seed, {}).update(json.load(f)["histories"])
    return out


INR_SERIES = [
    ("Adam (tuned)", "adam", C["adam"], "-", 1.3),
    ("Adam+cosine", "adam-cos", C["adam-cos"], "-", 1.3),
    ("KESTREL", "eagle-dqn-cd", C["kestrel"], "-", 2.0),
    ("KESTREL+cos", "kestrel-cos", C["kestrel"], "--", 2.0),
]


def inr_family_best(data, fam):
    """PSNR 最大で家族最良 lr を選び {seed: hist}。"""
    cells = sorted({c for s in data for c in data[s]
                    if c.split("@")[0] == fam})
    lrs = sorted({c.split("@")[1] for c in cells})
    best_lr, best_psnr = None, -np.inf
    for lr in lrs:
        vals = [max(data[s][f"{fam}@{lr}"]["psnr"]) for s in data
                if f"{fam}@{lr}" in data[s]]
        if vals and np.mean(vals) > best_psnr:
            best_psnr, best_lr = np.mean(vals), lr
    if best_lr is None:
        return None
    return {s: data[s][f"{fam}@{best_lr}"] for s in data
            if f"{fam}@{best_lr}" in data[s]}


def env_best(hists, fam):
    v = [max(h["psnr"]) for k, h in hists.items() if k.split("@")[0] == fam]
    return max(v) if v else None


def env_reach(hists, fam, thr=30.0):
    out = None
    for k, h in hists.items():
        if k.split("@")[0] != fam:
            continue
        for s, p in zip(h["steps"], h["psnr"]):
            if p >= thr:
                out = s if out is None else min(out, s)
                break
    return out


def fig_inr():
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.2))

    for ax, image, title in [(axes[0], "camera", "camera"),
                             (axes[1], "astronaut", "astronaut")]:
        data = collect_inr(("inrv3", "inrv3c"), image)
        for label, fam, color, ls, lw in INR_SERIES:
            hists = inr_family_best(data, fam)
            if not hists:
                continue
            xs, ys = mean_curve(hists, key="psnr")
            ax.plot(xs, ys, ls, color=color, lw=lw, label=label)
        ax.axhline(30, color=INK, lw=0.5, ls=":", alpha=0.6)
        ax.set_title(f"SIREN / {title}", color=INK)
        ax.set_xlabel("steps")
        ax.set_ylim(20, 36.5)
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].legend(loc="lower right", frameon=False)

    # (c) Kodak-24 散布: x = 30dB 到達 speedup vs adam (log2), y = ΔPSNR
    ax = axes[2]
    pts = {"eagle-dqn-cd": [], "kestrel-cos": [], "adam-cos": []}
    for i in range(1, 25):
        img = f"kodim{i:02d}"
        h = {}
        for pre in ("kodakb", "kodakc"):
            p = RESULTS / f"{pre}_{img}_s42" / "metrics.json"
            if p.exists():
                h.update(json.load(open(p))["histories"])
        a_best, a_reach = env_best(h, "adam"), env_reach(h, "adam")
        for fam in pts:
            b, r = env_best(h, fam), env_reach(h, fam)
            if None not in (a_best, a_reach, b, r):
                pts[fam].append((a_reach / r, b - a_best))
    style = {"eagle-dqn-cd": dict(marker="o", color=C["kestrel"],
                                  label="KESTREL"),
             "kestrel-cos": dict(marker="^", color=C["kestrel"],
                                 facecolors="none", label="KESTREL+cos"),
             "adam-cos": dict(marker="D", color=C["adam-cos"],
                              label="Adam+cosine")}
    for fam, kw in style.items():
        xy = np.array(pts[fam])
        if len(xy) == 0:
            continue
        fc = kw.pop("facecolors", kw["color"])
        ax.scatter(xy[:, 0], xy[:, 1], s=14, lw=1.0,
                   facecolors=fc, edgecolors=kw["color"],
                   marker=kw["marker"], label=kw["label"], alpha=0.85)
    ax.axvline(1.0, color=INK, lw=0.5, ls=":", alpha=0.6)
    ax.axhline(0.0, color=INK, lw=0.5, ls=":", alpha=0.6)
    ax.set_xscale("log", base=2)
    ax.set_xticks([0.5, 1, 2, 4])
    ax.set_xticklabels(["0.5×", "1×", "2×", "4×"])
    ax.set_title("Kodak-24 (per image)", color=INK)
    ax.set_xlabel("30 dB reach speedup vs. tuned Adam")
    ax.set_ylabel("ΔPSNR (dB)")
    ax.legend(loc="upper left", frameon=False, handletextpad=0.2)

    fig.tight_layout()
    fig.savefig(FIGS / "fig_inr.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    FIGS.mkdir(parents=True, exist_ok=True)
    fig_regression()
    fig_inr()
    print(f"figures -> {FIGS}")
