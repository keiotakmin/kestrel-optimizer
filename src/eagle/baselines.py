"""ICTAI 2026 用の追加ベースライン (ICMLA 査読 R5–R8 対応)。

torch / 外部パッケージにない 2 手法をここに実装する。外部パッケージ由来の
ベースライン (AdaBelief / AdaHessian) の登録は experiments/run_comparison.py。

- BB: Barzilai–Borwein (1988) の大域スカラーステップ勾配法。
  EAGLE の割線更新の直系の祖先 (R6-W1: Eq.(1) は BB ステップそのもの) で、
  「座標単位の割線 + 事後ベンチ保険」と「大域スカラー割線」の寄与を
  分離するための対照。フルバッチ決定論レジームは BB のホームグラウンド。
- AdamCosine: Adam + cosine アニーリング (η_t = η₀·½(1+cos(π·t/T)) → 0)。
  「終盤の到達 speedup は減衰スケジュール付き Adam でも得られるのでは」
  (R5-W7 / R6-W6) への対照。lr エンベロープが定数 lr しか覆っていない
  穴を塞ぐ。
"""

import math

import torch
from torch.optim.optimizer import Optimizer

from .optim import EAGLE


class BB(Optimizer):
    """Barzilai–Borwein 勾配法 (既定は BB1 = long step)。

    x_{k+1} = x_k − α_k g_k,
    BB1: α_k = (sᵀs)/(sᵀy),  BB2: α_k = (sᵀy)/(yᵀy)
    (s = x_k − x_{k−1}, y = g_k − g_{k−1}。モデル全体で 1 つのスカラー)

    セーフガード (標準的な非凸対応): 初回ステップ、sᵀy ≤ 0 (負曲率)、
    または α が非有限のときは α = lr にフォールバックする。
    lr は「初期/フォールバックステップ」で、lr エンベロープの走査対象。

    stab_delta: 非凸での非単調発散への標準的な対策 (stabilized BB,
    Burdakov–Dai–Huang 2019, arXiv:1907.06409): ステップノルムを Δ に
    制限する α_k ← min(α_k, Δ/‖g_k‖)。None で無効 (素の BB)。
    素の BB は tanh MLP 回帰で発散することを確認済み — これ自体が
    「大域割線も保険を要する」証拠だが、藁人形比較を避けるため
    stabilized 版もベースラインに含める。

    実装はホスト同期なし (α は 0 次元テンソルのままブロードキャスト)。
    """

    def __init__(self, params, lr=1e-3, variant="bb1", stab_delta=None):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if variant not in ("bb1", "bb2"):
            raise ValueError(f"variant は 'bb1' か 'bb2': {variant}")
        if stab_delta is not None and stab_delta <= 0:
            raise ValueError(f"Invalid stab_delta: {stab_delta}")
        super().__init__(params, dict(lr=lr, variant=variant,
                                      stab_delta=stab_delta))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            first = "prev_p" not in self.state[params[0]]

            alpha = None
            if not first:
                dev = params[0].device
                sts = torch.zeros((), device=dev)
                sty = torch.zeros((), device=dev)
                yty = torch.zeros((), device=dev)
                gtg = torch.zeros((), device=dev)
                for p in params:
                    st = self.state[p]
                    s = p.data - st["prev_p"]
                    yv = p.grad - st["prev_g"]
                    sts += (s * s).sum()
                    sty += (s * yv).sum()
                    yty += (yv * yv).sum()
                    gtg += (p.grad * p.grad).sum()
                if group["variant"] == "bb1":
                    alpha = sts / sty
                else:
                    alpha = sty / yty
                ok = torch.isfinite(alpha) & (sty > 0)
                alpha = torch.where(
                    ok, alpha, torch.full_like(alpha, group["lr"]))
                if group["stab_delta"] is not None:
                    # stabilized BB: ‖αg‖ ≤ Δ になるよう α を制限
                    alpha = torch.minimum(
                        alpha,
                        group["stab_delta"] / gtg.sqrt().clamp_min(1e-12))

            for p in params:
                st = self.state[p]
                if first:
                    st["prev_p"] = p.data.clone()
                    st["prev_g"] = p.grad.clone()
                else:
                    st["prev_p"].copy_(p.data)
                    st["prev_g"].copy_(p.grad)
                if alpha is None:
                    p.data.add_(p.grad, alpha=-group["lr"])
                else:
                    p.data.sub_(alpha * p.grad)

        return loss


class KestrelCosine(EAGLE):
    """KESTREL (eagle-dqn-cd) の Adam フォールバック lr に cosine アニーリング
    を掛けた変種。割線ジャンプは lr 非依存 (use_lr_in_eagle_update=False)
    なので、スケジュールは純粋にフォールバック/ベンチ枝の磨き込みにだけ効く。
    INR で観測した「終盤の磨きは cosine Adam が上」(adam-cos が最終 PSNR 最高)
    への直接回答: 到達速度 (ジャンプ) と最終床 (アニーリング) の両取りを狙う。
    """

    def __init__(self, params, lr=1e-3, T=1000, **kwargs):
        super().__init__(params, lr=lr, **kwargs)
        self._lr0 = lr
        self._T = max(int(T), 1)
        self._t = 0

    def step(self, closure=None):
        factor = 0.5 * (1.0 + math.cos(math.pi * min(self._t / self._T, 1.0)))
        for group in self.param_groups:
            group["lr"] = self._lr0 * factor
        self._t += 1
        return super().step(closure)


class AdamCosine(torch.optim.Adam):
    """Adam + cosine アニーリング (η_T → 0) の自己完結版。

    ハーネスの学習ループを変えずにスケジュールを効かせるため、step() 内で
    param_groups の lr を更新する。T (総ステップ数) は OPTIMIZER_BUILDERS
    経由で渡される。torch の CosineAnnealingLR と同じ η_t を刻む。
    """

    def __init__(self, params, lr=1e-3, T=1000, **kwargs):
        super().__init__(params, lr=lr, **kwargs)
        self._lr0 = lr
        self._T = max(int(T), 1)
        self._t = 0

    def step(self, closure=None):
        factor = 0.5 * (1.0 + math.cos(math.pi * min(self._t / self._T, 1.0)))
        for group in self.param_groups:
            group["lr"] = self._lr0 * factor
        self._t += 1
        return super().step(closure)
