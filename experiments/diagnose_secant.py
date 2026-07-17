"""割線ジャンプの実装検証と限界の切り分け。

疑問: 損失形状が綺麗な放物線なのに eagle2 の初期加速が僅かなのはなぜか。
実装が想定通りの割線ジャンプなら、厳密な 2 次関数 + ノイズなし勾配では
Δg/Δp = h (真の曲率) が厳密に成り立ち、ジャンプは 1 発で最小点に着地する。

Part A: 理想条件 (対角 2 次関数、決定的勾配)
    → ジャンプの着地精度 (contraction = 跳んだ後の距離 / 跳ぶ前の距離) を実測。
      contraction ≈ 0 なら実装は正しい割線法。
Part B: 勾配ノイズを注入 (ミニバッチの模擬)
    → contraction がノイズ量とともにどう壊れるか。
Part C: 非対角ヘッセ行列 (座標間干渉)
    → 対角割線近似が座標結合でどう壊れるか。

実行: python experiments/diagnose_secant.py
出力: results/analysis/secant_diagnosis.png + 標準出力の表
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eagle.optim import EAGLE  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT = RESULTS / "analysis"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N = 200          # パラメータ次元
LR = 0.01
STEPS = 500

# dataviz 参照パレット (categorical slot 1-3 / sequential blue ramp)
COLORS = {"adam": "#2a78d6", "eagle-orig": "#1baf7a", "eagle2": "#eda100"}
SEQ_BLUE = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab"]
INK = "#333333"


def make_problem(seed=0):
    g = torch.Generator().manual_seed(seed)
    h = 10 ** (torch.rand(N, generator=g) * 2 - 1)  # 曲率 0.1〜10
    pstar = torch.randn(N, generator=g) * 2.0
    p0 = pstar + torch.randn(N, generator=g) * 2.0  # 初期距離 ~2
    return h.to(DEVICE), pstar.to(DEVICE), p0.to(DEVICE)


def build_optimizer(name, params):
    if name == "adam":
        return torch.optim.Adam(params, lr=LR)
    if name == "eagle-orig":
        return EAGLE(params, lr=LR, base="adam", threshold=5e-4,
                     use_lr_in_eagle_update=False)
    if name == "eagle2":
        return EAGLE(params, lr=LR, base="adam", threshold=5e-4,
                     use_lr_in_eagle_update=False, trust_kappa=50,
                     cooldown_steps=20)
    if name == "eagle-pb":
        return EAGLE(params, lr=LR, base="adam", threshold=5e-4,
                     use_lr_in_eagle_update=False, paired_secant=True)
    if name == "eagle2-pb":
        return EAGLE(params, lr=LR, base="adam", threshold=5e-4,
                     use_lr_in_eagle_update=False, trust_kappa=50,
                     cooldown_steps=20, paired_secant=True)
    raise ValueError(name)


def eagle_mask_from_state(opt, p, grad):
    """orig の切替マスクを状態から再現する (カーネルと同一の式)。"""
    st = opt.state[p]
    group = opt.param_groups[0]
    if group["paired_secant"] and (st["step"] + 1) % 2 == 1:
        return torch.zeros_like(grad, dtype=torch.bool)  # 奇数ステップは全ベース
    prev_g = st["prev_grad"]
    dg = grad - prev_g
    stable = (prev_g * grad >= 0) & (grad * dg >= 0)
    base = stable | (dg.abs() < st["threshold_t"])
    return ~base


def run(name, h, pstar, p0, noise=0.0, A=None, steps=STEPS, seed=0,
        paired_noise=False):
    """1 設定を実行し、loss 履歴とジャンプ品質統計を返す。

    A が None なら対角 2 次関数 (勾配 h*(p-p*))、
    A があれば結合 2 次関数 (勾配 A @ (p-p*))。
    paired_noise=True なら、ノイズ (= ミニバッチの模擬) を 2 ステップ連続で
    同じ実現値にする (同一バッチを 2 回使うペアバッチ方式の模擬)。
    """
    torch.manual_seed(seed)
    p = torch.nn.Parameter(p0.clone())
    opt = build_optimizer(name, [p])
    is_eagle = isinstance(opt, EAGLE)
    has_jumped_flag = is_eagle and opt.param_groups[0]["cooldown_steps"]

    losses = []
    contractions = []   # ジャンプした座標の |p-p*| の縮小率
    jump_steps = []     # ジャンプが起きたステップ
    eps = None
    for t in range(steps):
        diff = p - pstar
        if A is None:
            loss = 0.5 * (h * diff ** 2).sum()
        else:
            loss = 0.5 * diff @ (A @ diff)
        opt.zero_grad()
        loss.backward()
        if noise > 0:
            if not (paired_noise and t % 2 == 1):
                eps = noise * torch.randn_like(p.grad)  # 新しい「バッチ」
            p.grad += eps
        losses.append(loss.item())

        d_before = (p.detach() - pstar).abs()
        predicted = None
        if is_eagle and t >= 1 and not has_jumped_flag:
            predicted = eagle_mask_from_state(opt, p, p.grad)
            count_before = int(opt.state[p]["eagle_count"])

        opt.step()

        if is_eagle and t >= 1:
            if has_jumped_flag:
                jumped = opt.state[p]["jumped"].bool()
            else:
                jumped = predicted
                # 再現マスクがカーネルの判定と一致するか検証
                delta = int(opt.state[p]["eagle_count"]) - count_before
                assert delta == int(jumped.sum()), "マスク再現がカーネルと不一致"
            if jumped.any():
                d_after = (p.detach() - pstar).abs()
                sel = jumped & (d_before > 1e-3)
                if sel.any():
                    c = (d_after[sel] / d_before[sel]).cpu().numpy()
                    contractions.extend(c.tolist())
                    jump_steps.extend([t] * int(sel.sum()))
    return dict(losses=losses, contractions=np.array(contractions),
                jump_steps=np.array(jump_steps))


def coupled_matrix(h, rho, seed=0):
    """スペクトルを保ったまま座標を混ぜたヘッセ行列。rho=0 で対角。"""
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(N, N, generator=g))
    Q = Q.to(DEVICE)
    D = torch.diag(h)
    return (1 - rho) * D + rho * (Q @ D @ Q.T)


def summarize(tag, res):
    c = res["contractions"]
    if len(c) == 0:
        print(f"{tag:<28} ジャンプなし")
        return
    good = float((c < 0.5).mean())
    bad = float((c > 1.5).mean())
    print(f"{tag:<28} jumps={len(c):>6}  contraction中央値={np.median(c):.4f}  "
          f"良(<0.5)={good * 100:５.1f}%  有害(>1.5)={bad * 100:5.1f}%  "
          f"最終loss={res['losses'][-1]:.3e}")


def main():
    OUT.mkdir(exist_ok=True)
    h, pstar, p0 = make_problem()

    print("=" * 100)
    print("Part A: 理想条件 (対角 2 次関数・決定的勾配) — 割線ジャンプは 1 発で"
          "谷底に着地するはず")
    print("=" * 100)
    ideal = {}
    for name in ["adam", "eagle-orig", "eagle2"]:
        ideal[name] = run(name, h, pstar, p0)
        summarize(f"[ideal] {name}", ideal[name])

    print()
    print("=" * 100)
    print("Part B: 勾配ノイズ注入 (ミニバッチ模擬) — Δg/Δp の SNR 崩壊を実測")
    print("=" * 100)
    noise_levels = [0.0, 0.05, 0.5, 2.0]
    noisy = {}
    for sig in noise_levels:
        noisy[sig] = {name: run(name, h, pstar, p0, noise=sig)
                      for name in ["adam", "eagle-orig", "eagle2"]}
        for name in ["eagle-orig", "eagle2"]:
            summarize(f"[noise σ={sig}] {name}", noisy[sig][name])

    print()
    print("=" * 100)
    print("Part C: 座標結合 (非対角ヘッセ行列) — 対角割線近似の破れ")
    print("=" * 100)
    rhos = [0.0, 0.5, 1.0]
    coupled = {}
    for rho in rhos:
        A = coupled_matrix(h, rho)
        coupled[rho] = run("eagle-orig", h, pstar, p0, A=A)
        summarize(f"[coupled ρ={rho}] eagle-orig", coupled[rho])

    # 論文用サマリー (ictai/gen_macros.py が読む single-source データ)
    summary = {
        "ideal_contraction_median": float(
            np.median(ideal["eagle-orig"]["contractions"])),
        "coupled_contraction_median": {
            str(rho): float(np.median(coupled[rho]["contractions"]))
            for rho in rhos},
    }
    with open(OUT / "secant_diagnosis.json", "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 100)
    print("Part D: ペアバッチ割線 — 同一バッチの 2 連使用でノイズ相殺されるか")
    print("=" * 100)
    paired = {}
    for sig in noise_levels:
        paired[sig] = {}
        for name in ["eagle-pb", "eagle2-pb"]:
            paired[sig][name] = run(name, h, pstar, p0, noise=sig,
                                    paired_noise=True)
            summarize(f"[paired σ={sig}] {name}", paired[sig][name])

    # ---- 図 ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0][0]
    for name, res in ideal.items():
        ax.plot(res["losses"], color=COLORS[name], linewidth=1.8, label=name)
    ax.set_yscale("log")
    ax.set_title("A: Ideal diagonal quadratic (exact gradients)",
                 fontsize=11, color=INK)
    ax.set_xlabel("Steps")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=9)

    ax = axes[0][1]
    for sig, c in zip(noise_levels, SEQ_BLUE):
        ax.plot(noisy[sig]["eagle-orig"]["losses"], color=c, linewidth=1.8,
                label=f"eagle-orig σ={sig}")
    ax.plot(noisy[0.5]["adam"]["losses"], color=COLORS["adam"],
            linewidth=1.8, linestyle="--", label="adam σ=0.5")
    ax.plot(paired[0.5]["eagle-pb"]["losses"], color=COLORS["eagle-orig"],
            linewidth=1.8, linestyle="--", label="eagle-pb σ=0.5 (paired)")
    ax.set_yscale("log")
    ax.set_title("B: Gradient noise breaks the secant estimate",
                 fontsize=11, color=INK)
    ax.set_xlabel("Steps")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=8)

    ax = axes[1][0]
    for name in ["eagle-orig", "eagle2"]:
        med = [np.median(noisy[s][name]["contractions"])
               if len(noisy[s][name]["contractions"]) else np.nan
               for s in noise_levels]
        ax.plot(noise_levels, med, "o-", color=COLORS[name], linewidth=1.8,
                label=name)
    for name, base in [("eagle-pb", "eagle-orig"), ("eagle2-pb", "eagle2")]:
        med = [np.median(paired[s][name]["contractions"])
               if len(paired[s][name]["contractions"]) else np.nan
               for s in noise_levels]
        ax.plot(noise_levels, med, "s--", color=COLORS[base], linewidth=1.8,
                label=f"{name} (paired)")
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.6)
    ax.set_title("D: Median jump contraction vs noise\n(<1 = jump helps, "
                 "0 = perfect landing)", fontsize=11, color=INK)
    ax.set_xlabel("Gradient noise σ")
    ax.set_ylabel("Contraction")
    ax.legend(fontsize=8)

    ax = axes[1][1]
    med = [np.median(coupled[r]["contractions"])
           if len(coupled[r]["contractions"]) else np.nan for r in rhos]
    ax.plot(rhos, med, "o-", color=COLORS["eagle-orig"], linewidth=1.8)
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.6)
    ax.set_title("C: Median jump contraction vs Hessian coupling ρ\n"
                 "(diagonal secant assumption breaks)", fontsize=11, color=INK)
    ax.set_xlabel("Coupling ρ (0 = diagonal Hessian)")
    ax.set_ylabel("Contraction")

    for ax in axes.flat:
        ax.grid(True, linestyle=":", alpha=0.3)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=8, colors=INK)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    fig.suptitle("Secant-jump diagnosis: implementation is exact in the ideal "
                 "case; noise and coupling break it", fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = OUT / "secant_diagnosis.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"\n図を保存: {path}")


if __name__ == "__main__":
    main()
