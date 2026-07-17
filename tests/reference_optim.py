"""EAGLE オプティマイザ本体。

archive/ 以下のノートブック・スクリプトに散在していた実装
(AdaEAGLE, EAGLE, EAGLE-A, EAGLE-S, LR あり/なし) を 1 クラスに統合したもの。
"""

import math

import numpy as np
import torch
from torch.optim.optimizer import Optimizer


class EAGLE(Optimizer):
    """勾配の振動状態に応じて割線法更新とベース更新を要素ごとに切り替えるオプティマイザ。

    パラメータの各要素について、直近 2 ステップの勾配から状態を判定する。

    - ベース更新 (Adam または SGD+momentum):
        勾配の符号が安定している要素、または勾配変化 |Δg| が閾値未満の要素。
    - EAGLE 更新 (割線法):
        勾配が振動している要素。
            p <- p - c * g_t * Δp / Δg      (c = lr または 1.0)
        Δp = p_t - p_{t-1}, Δg = g_t - g_{t-1} で曲率 (ヘッセ対角) を近似する。

    ベース更新を選ぶ条件:
        (g_{t-1} * g_t >= 0 かつ g_t * Δg >= 0)  または  |Δg| < threshold

    Args:
        params: 最適化対象のパラメータ。
        lr: 学習率。
        betas: base="adam" のときの (β1, β2)。
        eps: Adam の分母安定化項。
        weight_decay: AdamW スタイルの decoupled weight decay。
        base: 安定時のベース更新則。"adam" (EAGLE-A) か "sgd" (EAGLE-S)。
        momentum: base="sgd" のときのモーメンタム係数。
        amsgrad: base="adam" のとき AMSGrad を使うか。
        adaptive_threshold: 勾配ノルム履歴の変動係数から閾値を自動調整するか。
        threshold: 切替閾値。adaptive_threshold=True のときは初期値。
        use_lr_in_eagle_update: EAGLE 更新に lr を掛けるか。
            卒論オリジナル実装は False (割線法そのまま)。False は 1 ステップの
            移動量が大きくなり得るため、大きいモデルでは True を推奨。
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, base="adam", momentum=0.9, amsgrad=False,
                 adaptive_threshold=False, threshold=5e-4,
                 use_lr_in_eagle_update=True):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if base not in ("adam", "sgd"):
            raise ValueError(f"base must be 'adam' or 'sgd', got: {base}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        base=base, momentum=momentum, amsgrad=amsgrad,
                        adaptive_threshold=adaptive_threshold, threshold=threshold,
                        use_lr_in_eagle_update=use_lr_in_eagle_update)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)
            group.setdefault("use_lr_in_eagle_update", True)

    @staticmethod
    def _adaptive_threshold(state, grad_norm, fallback):
        """勾配ノルム履歴の変動係数 (CV) に基づく切替閾値。変動が大きいほど閾値を上げる。"""
        history = state.setdefault("grad_norm_history", [])
        history.append(float(grad_norm))
        del history[:-10]
        if len(history) >= 5:
            mean = float(np.mean(history))
            std = float(np.std(history))
            cv = std / (mean + 1e-8)
            return max(1e-5, min(1e-2, cv * 5e-3))
        return fallback

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            base = group["base"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("EAGLE does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["prev_param"] = p.data.clone()
                    state["prev_grad"] = torch.zeros_like(p.data)
                    state["threshold"] = group["threshold"]
                    state["eagle_count"] = 0
                    state["base_count"] = 0
                    if base == "adam":
                        state["exp_avg"] = torch.zeros_like(p.data)
                        state["exp_avg_sq"] = torch.zeros_like(p.data)
                        if group["amsgrad"]:
                            state["max_exp_avg_sq"] = torch.zeros_like(p.data)
                    else:
                        state["momentum_buffer"] = torch.zeros_like(p.data)

                if group["weight_decay"] != 0:
                    p.data.mul_(1 - lr * group["weight_decay"])

                curr_param = p.data.clone()
                curr_grad = p.grad.clone()
                delta_param = curr_param - state["prev_param"]
                delta_grad = curr_grad - state["prev_grad"]

                if group["adaptive_threshold"] and state["step"] > 0:
                    state["threshold"] = self._adaptive_threshold(
                        state, torch.norm(curr_grad), state["threshold"])

                state["step"] += 1

                # ベース更新量の計算
                if base == "adam":
                    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                    exp_avg.mul_(beta1).add_(curr_grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(curr_grad, curr_grad, value=1 - beta2)
                    if group["amsgrad"]:
                        torch.maximum(state["max_exp_avg_sq"], exp_avg_sq,
                                      out=state["max_exp_avg_sq"])
                        second_moment = state["max_exp_avg_sq"]
                    else:
                        second_moment = exp_avg_sq
                    denom = (second_moment.sqrt()
                             / math.sqrt(1 - beta2 ** state["step"])).add_(eps)
                    base_step = exp_avg / denom * (lr / (1 - beta1 ** state["step"]))
                else:
                    buf = state["momentum_buffer"]
                    buf.mul_(group["momentum"]).add_(curr_grad)
                    base_step = lr * buf

                # 更新則の切り替えマスク
                stable = (state["prev_grad"] * curr_grad >= 0) & (curr_grad * delta_grad >= 0)
                small_change = torch.abs(delta_grad) < state["threshold"]
                base_mask = stable | small_change
                eagle_mask = ~base_mask

                state["base_count"] += int(base_mask.sum())
                state["eagle_count"] += int(eagle_mask.sum())

                p.data[base_mask] -= base_step[base_mask]

                # EAGLE (割線法) 更新。0 除算はクランプで回避
                safe_denom = delta_grad.clone()
                safe_denom[torch.abs(safe_denom) < 1e-8] = 1e-8
                coeff = lr if group["use_lr_in_eagle_update"] else 1.0
                p.data[eagle_mask] -= (coeff * curr_grad[eagle_mask]
                                       * delta_param[eagle_mask] / safe_denom[eagle_mask])

                state["prev_param"] = curr_param
                state["prev_grad"] = curr_grad

        return loss


def eagle_update_ratio(optimizer):
    """全パラメータ更新のうち EAGLE 更新則が使われた割合を返す。EAGLE 以外は None。"""
    if not isinstance(optimizer, EAGLE):
        return None
    eagle_count = 0
    base_count = 0
    for group in optimizer.param_groups:
        for p in group["params"]:
            state = optimizer.state[p]
            if "eagle_count" in state:
                eagle_count += state["eagle_count"]
                base_count += state["base_count"]
    total = eagle_count + base_count
    return eagle_count / total if total > 0 else 0.0
