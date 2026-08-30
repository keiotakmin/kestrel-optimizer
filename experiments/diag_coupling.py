"""結合二次問題での割線ジャンプ診断 (継続研究 TO DO 8)。

目的は 2 つある。

(1) **適用条件マップ (RQ1)**: 座標結合の強さ ‖D^{-1}E‖ を密に振り、
    ジャンプの収縮率がどこで壊れるかの境界を求める。

(2) **検出器の検証 (RQ2)**: 二次問題では最適点 θ* が既知なので、
    「そのジャンプが実際に悪かったか」を座標ごとの真ラベル
    (|θ_i - θ*_i| が増えたか) で定義できる。事後ベンチの発火条件
    (符号反転 かつ |g| 増大) をこの真ラベルに対して評価し、
    precision / recall と **sign-preserving な見逃し**の割合を出す。

(2) が TO DO 7 との接続点である。INR では θ* が未知なので TO DO 7 の
混同行列は |g| 増大を代理ラベルとして使うしかない。ここでその代理が
真ラベルをどれだけ再現するかを測っておけば、INR 側の数字を解釈できる。

問題:  f(θ) = ½ (θ-θ*)ᵀ A (θ-θ*),  A = (1-ρ)D + ρ QDQᵀ (スペクトル保存)
       勾配は決定的 (KESTREL が対象とするフルバッチ領域と同じ)。

実行例:
    python experiments/diag_coupling.py --out results/coupling
    python experiments/diag_coupling.py --rhos 0 0.5 1.0 --seeds 42 --quick
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from bench_probe import JumpProbe  # noqa: E402
from eagle.optim import EAGLE  # noqa: E402
from expand_grid_kodak import run_metadata  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

BASE_KW = dict(base="adam", adaptive_threshold=False, threshold=5e-4,
               use_lr_in_eagle_update=False, curvature_ema=0.8,
               always_jump=True)
OPTS = {
    "kestrel": dict(BASE_KW, cooldown_steps=20),
    "kestrel-nobench": dict(BASE_KW, cooldown_steps=None),
    "adam": None,
}
LRS = [1e-3, 1e-2, 1e-1]


class GTProbe(JumpProbe):
    """JumpProbe に「真ラベルとの照合」を足したもの。

    最適点が既知なので、ジャンプした座標が実際に θ* から遠ざかったかを
    真の失敗ラベルとして持ち、次ステップでベンチの発火条件と突き合わせる。
    """

    def __init__(self, device, cooldown_steps, theta_star):
        super().__init__(device, cooldown_steps=cooldown_steps)
        self.theta_star = theta_star
        self.jumped_now = None      # このステップで jump した座標
        self.d_before = None        # jump 直前の |θ-θ*|
        self.gt_harmful = None      # 前ステップの真の失敗ラベル
        self.cm = {k: 0 for k in ("tp", "fp", "fn", "tn",
                                  "harm_signflip", "harm_signpres")}
        self.contractions = []      # (rho ごとに集める) 収縮率

    def __call__(self, p, state, group, ctx):
        # 真ラベル (前ステップ分) と検出器の照合。probe は prev_grad 更新前に
        # 呼ばれるので、ここでの prev_grad = g_{t-1}, grad = g_t。
        if self.gt_harmful is not None:
            pg, g = ctx["prev_grad"], ctx["grad"]
            flip = pg * g < 0
            worse = g.abs() > pg.abs()
            # 母集団は「前ステップで jump した座標」だけ。ベンチもその座標に
            # しか作用しないので、ここを揃えないと fp を過大に数える。
            pj = self.jumped_now
            fired = flip & worse & pj     # ベンチの発火条件 (4)
            gt = self.gt_harmful & pj
            self.cm["tp"] += int((fired & gt).sum())
            self.cm["fp"] += int((fired & ~gt).sum())
            self.cm["fn"] += int((~fired & gt).sum())
            self.cm["tn"] += int((~fired & ~gt & pj).sum())
            self.cm["harm_signflip"] += int((gt & flip).sum())
            self.cm["harm_signpres"] += int((gt & ~flip).sum())
            self.cm["n_jumped"] = self.cm.get("n_jumped", 0) + int(pj.sum())
        super().__call__(p, state, group, ctx)
        self.jumped_now = (~ctx["base_mask"]).clone()
        self.d_before = (p.data - self.theta_star).abs().clone()

    def after_step(self, p):
        """optimizer.step() の後に呼ぶ。収縮率と真ラベルを確定する。"""
        d_after = (p.data - self.theta_star).abs()
        j = self.jumped_now
        ratio = d_after / (self.d_before + 1e-30)
        self.gt_harmful = j & (ratio > 1.0)
        c = ratio[j]
        if c.numel():
            self.contractions.append(c.detach().cpu().numpy())


def make_problem(n, cond, seed, device):
    """曲率 h を対数一様に張り、条件数 = cond の対角スペクトルを作る。"""
    g = torch.Generator().manual_seed(seed)
    e = torch.rand(n, generator=g) * 2 - 1                 # -1..1
    h = cond ** (0.5 * (e + 1)) / cond ** 0.5              # 1/√c .. √c
    h = h * cond ** 0.5                                    # 1 .. cond
    theta_star = torch.randn(n, generator=g) * 2.0
    theta0 = theta_star + torch.randn(n, generator=g) * 2.0
    return (h.to(device), theta_star.to(device), theta0.to(device),
            torch.Generator().manual_seed(seed + 1))


def coupled_matrix(h, rho, n, device, seed=0):
    """スペクトルを保ったまま座標を混ぜる。rho=0 で対角。"""
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(n, n, generator=g))
    Q = Q.to(device)
    D = torch.diag(h)
    return (1 - rho) * D + rho * (Q @ D @ Q.T)


def coupling_proxy(A):
    """理論側の結合度 ‖D^{-1}E‖₂ (D = diag(A), E = A - D)。"""
    d = torch.diagonal(A)
    E = A - torch.diag(d)
    return float(torch.linalg.matrix_norm(E / d.unsqueeze(1), ord=2))


def run(opt_name, lr, A, theta_star, theta0, steps, device, probe=True):
    theta = torch.nn.Parameter(theta0.clone())
    kw = OPTS[opt_name]
    if kw is None:
        optimizer = torch.optim.Adam([theta], lr=lr)
        jp = None
    else:
        optimizer = EAGLE([theta], lr=lr, fused=False, **kw)
        jp = (GTProbe(device, kw["cooldown_steps"], theta_star)
              if probe else None)
        if jp is not None:
            for g in optimizer.param_groups:
                g["_probe"] = jp

    losses = []
    for t in range(steps):
        optimizer.zero_grad(set_to_none=True)
        d = theta - theta_star
        loss = 0.5 * (d @ (A @ d))
        loss.backward()
        losses.append(float(loss))
        if kw is not None:
            optimizer.clean_step = True
        optimizer.step()
        if jp is not None:
            jp.flush(t, loss=losses[-1])
            jp.after_step(theta)
    with torch.no_grad():
        d = theta - theta_star
        losses.append(float(0.5 * (d @ (A @ d))))

    out = {"losses": losses, "final_loss": losses[-1],
           "min_loss": float(min(losses))}
    if jp is not None:
        cm = jp.cm
        tp, fp, fn = cm["tp"], cm["fp"], cm["fn"]
        harm = cm["harm_signflip"] + cm["harm_signpres"]
        c = (np.concatenate(jp.contractions) if jp.contractions
             else np.array([]))
        out.update({
            "confusion": cm,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "harmful_rate": harm / max(tp + fp + fn + cm["tn"], 1),
            # 検出器の盲点: 有害ジャンプのうち符号が反転しなかった割合
            "signpreserving_share": (cm["harm_signpres"] / harm
                                     if harm else None),
            "n_jumps": int(c.size),
            "contraction_median": float(np.median(c)) if c.size else None,
            "contraction_p90": float(np.percentile(c, 90)) if c.size else None,
            "good_jump_share": float((c < 0.5).mean()) if c.size else None,
            "harmful_jump_share": float((c > 1.0).mean()) if c.size else None,
        })
        diag = jp.finish()
        # INR 側で観測できる代理量だけを残す (真ラベルは INR では使えない)
        for k in ("neg_secant_rate", "undet_fail_rate", "det_fail_rate",
                  "jump_rate", "bench_rate", "jump_ratio_mean"):
            v = [x for x in diag[k] if x is not None]
            out[f"obs_{k}_mean"] = float(np.mean(v)) if v else None
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--conds", nargs="+", type=float, default=[10, 100, 1000])
    p.add_argument("--rhos", nargs="+", type=float, default=None,
                   help="既定: 0..1 を 21 点 (高分解能)")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--lrs", nargs="+", type=float, default=LRS)
    p.add_argument("--out", default=str(ROOT / "results" / "coupling"))
    p.add_argument("--quick", action="store_true")
    a = p.parse_args()

    rhos = a.rhos if a.rhos is not None else [round(i / 20, 3)
                                              for i in range(21)]
    if a.quick:
        rhos, a.conds, a.seeds = rhos[::5], a.conds[:1], a.seeds[:1]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = run_metadata()
    total = (len(rhos) * len(a.conds) * len(a.seeds) * len(OPTS)
             * len(a.lrs))
    print(f"# diag_coupling n={a.n} steps={a.steps} rho={len(rhos)} "
          f"cond={a.conds} seeds={a.seeds} lrs={a.lrs} -> {total} runs")
    print(f"# device={device} {meta}", flush=True)

    records = []
    t0 = time.time()
    done = 0
    for cond in a.conds:
        for seed in a.seeds:
            h, theta_star, theta0, _ = make_problem(a.n, cond, seed, device)
            for rho in rhos:
                A = coupled_matrix(h, rho, a.n, device, seed=seed)
                cp = coupling_proxy(A)
                for opt_name in OPTS:
                    for lr in a.lrs:
                        r = run(opt_name, lr, A, theta_star, theta0,
                                a.steps, device)
                        r.pop("losses")
                        r.update({"optimizer": opt_name, "lr": lr,
                                  "rho": rho, "cond": cond, "seed": seed,
                                  "coupling_proxy": cp, "n": a.n})
                        records.append(r)
                        done += 1
                if done % 50 < len(OPTS) * len(a.lrs):
                    el = time.time() - t0
                    print(f"  {done}/{total} (ETA "
                          f"{el / done * (total - done) / 60:.1f} min)",
                          flush=True)

    path = out_dir / "coupling_sweep.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"metadata": meta, "args": vars(a),
                               "records": records}))
    tmp.rename(path)
    print(f"\n-> {path}  ({len(records)} records, "
          f"{(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
