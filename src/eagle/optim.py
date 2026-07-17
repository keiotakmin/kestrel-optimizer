"""EAGLE オプティマイザ本体。

実装は 2 経路:
- fused: PyTorch 本体の fused Adam/SGD と同様、1 パラメータテンソルにつき
  1 CUDA カーネルで全処理を行う (CUDA + float32 + contiguous のとき)。
- vectorized: 上記が使えない場合のフォールバック (CPU でも動く)。

どちらの経路もホスト-デバイス同期を発生させない。使用回数カウンタは
デバイス上のテンソルに蓄積し、eagle_update_ratio() を呼んだときだけ同期する。
"""

import math

import torch
from torch.optim.optimizer import Optimizer

from ._cuda import get_extension


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
        amsgrad: base="adam" のとき AMSGrad を使うか (vectorized 経路のみ)。
        adaptive_threshold: 勾配ノルム履歴の変動係数から閾値を自動調整するか。
        threshold: 切替閾値。adaptive_threshold=True のときは初期値。
        use_lr_in_eagle_update: EAGLE 更新に lr を掛けるか。
            卒論オリジナル実装は False (割線法そのまま)。
        snr_gate: B2 改良。None 以外なら固定/適応閾値の代わりに
            |Δg| > snr_gate·√v̂ を EAGLE 発動条件にする (√v̂ は Adam の
            バイアス補正済み 2 次モーメント = 勾配の典型スケール)。
            終盤に勾配が縮みノイズ比が上がると自動的に Adam へ軟着陸する。
            base="adam" のみ対応。
        trust_kappa: B1 改良。None 以外なら EAGLE ステップの絶対値を
            trust_kappa·|ベース更新量| に制限する (trust region)。
            正常な割線ジャンプは殺さず、ノイズ由来の異常ジャンプだけ抑える。
        cooldown_steps: B3 改良。None 以外なら、EAGLE ジャンプ直後に勾配符号が
            反転しかつ |g| が増大した要素を「ジャンプ失敗」とみなし、その要素を
            cooldown_steps ステップの間ベース更新に固定する (1〜255)。
            base="adam" のみ対応。
        coeff_anneal_steps: 対照実験用。None 以外なら EAGLE 係数を
            1.0 → lr に cosine スケジュールで減衰させる (T = このステップ数)。
            use_lr_in_eagle_update の設定より優先される。
        paired_secant: ペアバッチ割線。学習ループが各ミニバッチを 2 ステップ
            連続で与える前提で、偶数ステップ (ペアの 2 発目 = Δg が同一バッチの
            勾配差でノイズが厳密に相殺される) のみ EAGLE 更新を許可する。
            奇数ステップは全要素ベース更新に固定し、クールダウン状態も凍結。
            train_model(..., repeat_batch=True) と併用すること。base="adam" のみ。
        curvature_ema: 間欠ペア + 曲率 EMA (β_h をこの値に指定)。学習ループが
            時々同一バッチを 2 連使用し、2 発目の前に optimizer.clean_step=True
            を立てる前提。clean ステップのみ Δg/Δp (正値のみ) を要素ごとの
            曲率 EMA に取り込み、ジャンプは毎ステップ g/h_ema で行う
            (現在の Δg では割らない)。発火条件は勾配符号の不安定 + EMA 有効。
            train_model(..., pair_every=k) と併用。base="adam" のみ、
            paired_secant とは併用不可。
            なお curvature_ema モードでは snr_gate は「発火ゲート」として働く
            (|Δg| >= snr_gate·√v̂ の要素のみ EAGLE 発火。G1)。
        signal_gate: モーメンタム SNR 発火ゲート (G2)。curvature_ema モード
            専用。|m̂| >= signal_gate·√v̂ (= Adam ステップが lr·signal_gate 以上)
            の要素のみ EAGLE 発火。ノイズ床では m̂→0 となり自動停止する。
        conf_gate: 曲率信頼度ゲート。curvature_ema モード専用。クリーン測定の
            分散 EMA (h_var) も保持し、変動係数 √h_var/h_ema < conf_gate
            (= 曲率推定が安定している) の要素のみ EAGLE 発火。
            「割線が信頼できる所で撃つ」の直接実装で、G2 の代理変数性
            (ノイズ床と座標最小近傍の混同) を解消する。
        momentum_jump: True ならジャンプの分子を生の勾配 g ではなく
            ノイズ平均された m̂ にする (p -= m̂/h_ema)。curvature_ema モード専用。
        always_jump: True なら勾配符号の安定/振動による切替を行わず、曲率
            EMA が有効な要素はすべて EAGLE 更新にする (= 保険つき対角
            準ニュートン)。振動判定がフルバッチ・決定論レジームでも必要かを
            検証する対照用。curvature_ema モード専用。
        fused: fused CUDA カーネルを使うか。None なら使える環境で自動的に有効。
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, base="adam", momentum=0.9, amsgrad=False,
                 adaptive_threshold=False, threshold=5e-4,
                 use_lr_in_eagle_update=True, snr_gate=None, trust_kappa=None,
                 cooldown_steps=None, coeff_anneal_steps=None,
                 paired_secant=False, curvature_ema=None, signal_gate=None,
                 conf_gate=None, momentum_jump=False, always_jump=False,
                 fused=None):
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
        if fused and amsgrad:
            raise ValueError("fused=True は amsgrad に未対応")
        if snr_gate is not None:
            if base != "adam":
                raise ValueError("snr_gate は base='adam' のみ対応")
            if snr_gate <= 0:
                raise ValueError(f"Invalid snr_gate: {snr_gate}")
            if adaptive_threshold:
                raise ValueError("snr_gate と adaptive_threshold は併用不可")
        if signal_gate is not None:
            if curvature_ema is None:
                raise ValueError("signal_gate は curvature_ema モード専用")
            if signal_gate <= 0:
                raise ValueError(f"Invalid signal_gate: {signal_gate}")
        if conf_gate is not None:
            if curvature_ema is None:
                raise ValueError("conf_gate は curvature_ema モード専用")
            if conf_gate <= 0:
                raise ValueError(f"Invalid conf_gate: {conf_gate}")
        if momentum_jump and curvature_ema is None:
            raise ValueError("momentum_jump は curvature_ema モード専用")
        if always_jump and curvature_ema is None:
            raise ValueError("always_jump は curvature_ema モード専用")
        if trust_kappa is not None and trust_kappa <= 0:
            raise ValueError(f"Invalid trust_kappa: {trust_kappa}")
        if cooldown_steps is not None:
            if base != "adam":
                raise ValueError("cooldown_steps は base='adam' のみ対応")
            if not 1 <= cooldown_steps <= 255:
                raise ValueError(f"cooldown_steps は 1〜255: {cooldown_steps}")
        if coeff_anneal_steps is not None and coeff_anneal_steps <= 0:
            raise ValueError(f"Invalid coeff_anneal_steps: {coeff_anneal_steps}")
        if paired_secant and base != "adam":
            raise ValueError("paired_secant は base='adam' のみ対応")
        if curvature_ema is not None:
            if base != "adam":
                raise ValueError("curvature_ema は base='adam' のみ対応")
            # 0.0 は「毎回のクリーン測定をそのまま使う」(EMA なし、正値
            # フィルタと前回値ホールドは維持) — 決定論レジームの鮮度検証用
            if not 0.0 <= curvature_ema < 1.0:
                raise ValueError(f"Invalid curvature_ema: {curvature_ema}")
            if paired_secant:
                raise ValueError("curvature_ema と paired_secant は併用不可")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        base=base, momentum=momentum, amsgrad=amsgrad,
                        adaptive_threshold=adaptive_threshold, threshold=threshold,
                        use_lr_in_eagle_update=use_lr_in_eagle_update,
                        snr_gate=snr_gate, trust_kappa=trust_kappa,
                        cooldown_steps=cooldown_steps,
                        coeff_anneal_steps=coeff_anneal_steps,
                        paired_secant=paired_secant, curvature_ema=curvature_ema,
                        signal_gate=signal_gate, conf_gate=conf_gate,
                        momentum_jump=momentum_jump, always_jump=always_jump,
                        fused=fused)
        super().__init__(params, defaults)
        # 学習ループが「このステップは同一バッチ 2 発目 (クリーン測定可)」を
        # 通知するフラグ。step() が読んでリセットする
        self.clean_step = False

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)
            group.setdefault("use_lr_in_eagle_update", True)
            group.setdefault("snr_gate", None)
            group.setdefault("trust_kappa", None)
            group.setdefault("cooldown_steps", None)
            group.setdefault("coeff_anneal_steps", None)
            group.setdefault("paired_secant", False)
            group.setdefault("curvature_ema", None)
            group.setdefault("signal_gate", None)
            group.setdefault("conf_gate", None)
            group.setdefault("momentum_jump", False)
            group.setdefault("always_jump", False)
            group.setdefault("fused", None)

    def _init_state(self, p, group):
        state = self.state[p]
        state["step"] = 0
        state["prev_param"] = p.data.clone()
        state["prev_grad"] = torch.zeros_like(p.data)
        # 切替閾値と使用回数カウンタはデバイス上に置き、同期なしで更新する
        state["threshold_t"] = torch.tensor(
            float(group["threshold"]), dtype=torch.float32, device=p.device)
        state["eagle_count"] = torch.zeros((), dtype=torch.int64, device=p.device)
        state["base_count"] = torch.zeros((), dtype=torch.int64, device=p.device)
        if group["adaptive_threshold"]:
            state["gn_hist"] = torch.zeros(10, dtype=torch.float32, device=p.device)
            state["gn_len"] = 0
        if group["base"] == "adam":
            state["exp_avg"] = torch.zeros_like(p.data)
            state["exp_avg_sq"] = torch.zeros_like(p.data)
            if group["amsgrad"]:
                state["max_exp_avg_sq"] = torch.zeros_like(p.data)
            if group["cooldown_steps"] is not None:
                state["jumped"] = torch.zeros_like(p.data, dtype=torch.uint8)
                state["cooldown"] = torch.zeros_like(p.data, dtype=torch.uint8)
            if group["curvature_ema"] is not None:
                # 0 以下 = 未測定 (無効) のマーカー
                state["h_ema"] = torch.zeros_like(p.data)
                if group["conf_gate"] is not None:
                    state["h_var"] = torch.zeros_like(p.data)
        else:
            state["momentum_buffer"] = torch.zeros_like(p.data)

    @staticmethod
    def _update_adaptive_threshold(state, grad):
        """勾配ノルム履歴 (直近 10 個のリングバッファ) の変動係数から閾値を更新。

        すべてデバイス上の演算で行い、ホスト同期を発生させない。
        """
        hist = state["gn_hist"]
        hist[state["gn_len"] % 10] = torch.linalg.vector_norm(grad)
        state["gn_len"] += 1
        n = min(state["gn_len"], 10)
        if n >= 5:
            valid = hist[:n]
            mean = valid.mean()
            std = valid.std(correction=0)
            cv = std / (mean + 1e-8)
            state["threshold_t"].copy_((cv * 5e-3).clamp_(1e-5, 1e-2))

    def _dummy_uint8(self, device):
        """cooldown 無効時に fused カーネルへ渡すダミーバッファ (未参照)。"""
        dummies = getattr(self, "_dummies", None)
        if dummies is None:
            dummies = self._dummies = {}
        if device not in dummies:
            dummies[device] = torch.zeros(1, dtype=torch.uint8, device=device)
        return dummies[device]

    def _dummy_f32(self, device):
        """curvature_ema 無効時に fused カーネルへ渡すダミーバッファ (未参照)。"""
        dummies = getattr(self, "_dummies_f32", None)
        if dummies is None:
            dummies = self._dummies_f32 = {}
        if device not in dummies:
            dummies[device] = torch.zeros(1, dtype=torch.float32, device=device)
        return dummies[device]

    def _use_fused(self, p, group):
        if group["fused"] is False:
            return False
        if group["amsgrad"] or not p.is_cuda or p.dtype != torch.float32:
            if group["fused"]:
                raise RuntimeError(
                    "fused=True には CUDA 上の float32 パラメータ (amsgrad なし) が必要")
            return False
        return get_extension() is not None

    @staticmethod
    def _vectorized_step(p, grad, state, group, coeff, clean=False):
        beta1, beta2 = group["betas"]
        lr = group["lr"]
        step = state["step"]

        prev_p, prev_g = state["prev_param"], state["prev_grad"]
        delta_param = p.data - prev_p
        delta_grad = grad - prev_g

        vhat_sqrt = None
        if group["base"] == "adam":
            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            if group["amsgrad"]:
                torch.maximum(state["max_exp_avg_sq"], exp_avg_sq,
                              out=state["max_exp_avg_sq"])
                second_moment = state["max_exp_avg_sq"]
            else:
                second_moment = exp_avg_sq
            vhat_sqrt = second_moment.sqrt().div_(math.sqrt(1 - beta2 ** step))
            denom = vhat_sqrt + group["eps"]
            base_step = exp_avg / denom * (lr / (1 - beta1 ** step))
        else:
            buf = state["momentum_buffer"]
            buf.mul_(group["momentum"]).add_(grad)
            base_step = lr * buf

        # B2: SNR ゲート (勾配の典型スケール比) または固定/適応閾値
        if group["snr_gate"] is not None:
            thr = group["snr_gate"] * vhat_sqrt
        else:
            thr = state["threshold_t"]
        # ペアバッチ割線: 奇数ステップ (異バッチの Δg) では EAGLE を禁止
        allow_eagle = (not group["paired_secant"]) or (state["step"] % 2 == 0)

        stable = (prev_g * grad >= 0) & (grad * delta_grad >= 0)

        # 間欠ペア + 曲率 EMA: clean ステップのみ測定を EMA に取り込む
        use_ema = group["curvature_ema"] is not None
        if use_ema:
            he = state["h_ema"]
            if clean:
                valid = delta_param.abs() > 1e-12
                hn = delta_grad / torch.where(
                    valid, delta_param, torch.ones_like(delta_param))
                accept = valid & (hn > 0)
                beta_h = group["curvature_ema"]
                merged = torch.where(he > 0, beta_h * he + (1 - beta_h) * hn, hn)
                if group["conf_gate"] is not None:
                    # 曲率の分散 EMA (信頼度ゲート用)。初回は CV=1 で初期化
                    hv = state["h_var"]
                    delta = hn - he
                    merged_hv = torch.where(
                        he > 0, beta_h * hv + (1 - beta_h) * delta * delta,
                        hn * hn)
                    state["h_var"] = torch.where(accept, merged_hv, hv)
                state["h_ema"] = torch.where(accept, merged, he)
                he = state["h_ema"]
            # 発火条件: 勾配符号が不安定 かつ EMA が有効。
            # always_jump では振動判定を外し、EMA 有効な全要素で発火する
            if group["always_jump"]:
                base_mask = he <= 0
            else:
                base_mask = stable | (he <= 0)
            # G1: Δg ベースの SNR 発火ゲート
            if group["snr_gate"] is not None:
                base_mask = base_mask | (delta_grad.abs()
                                         < group["snr_gate"] * vhat_sqrt)
            # G2: モーメンタム SNR 発火ゲート (|m̂| >= c·√v̂)
            if group["signal_gate"] is not None:
                m_hat_abs = exp_avg.abs() / (1 - beta1 ** step)
                base_mask = base_mask | (m_hat_abs
                                         < group["signal_gate"] * vhat_sqrt)
            # 曲率信頼度ゲート: 変動係数 < conf_gate の要素のみ発火
            if group["conf_gate"] is not None:
                c2 = group["conf_gate"] ** 2
                base_mask = base_mask | (state["h_var"] >= c2 * he * he)
        else:
            base_mask = stable | (delta_grad.abs() < thr)
        if not allow_eagle:
            base_mask = torch.ones_like(base_mask)

        # B3: 失敗判定つきクールダウン (EAGLE 許可ステップのみ動作・更新)
        K = group["cooldown_steps"]
        if K is not None and allow_eagle:
            cd = state["cooldown"]
            failed = (state["jumped"].bool() & (prev_g * grad < 0)
                      & (grad.abs() > prev_g.abs()))
            cd = torch.where(failed, torch.full_like(cd, K), cd)
            force_base = cd > 0
            state["cooldown"] = torch.where(force_base, cd - 1, cd)
            base_mask = base_mask | force_base

        if use_ema:
            if group["momentum_jump"]:
                numerator = exp_avg / (1 - beta1 ** step)  # m̂ (ノイズ平均)
            else:
                numerator = grad
            eagle_step = coeff * numerator / state["h_ema"].clamp(min=1e-12)
        else:
            safe_denom = torch.where(delta_grad.abs() < 1e-8,
                                     torch.full_like(delta_grad, 1e-8),
                                     delta_grad)
            eagle_step = coeff * grad * delta_param / safe_denom
        # B1: trust region (EAGLE ステップを κ·|ベース更新量| に制限)
        if group["trust_kappa"] is not None:
            lim = group["trust_kappa"] * base_step.abs()
            eagle_step = torch.maximum(torch.minimum(eagle_step, lim), -lim)
        update = torch.where(base_mask, base_step, eagle_step)

        if K is not None and allow_eagle:
            state["jumped"] = (~base_mask).to(torch.uint8)

        n_base = base_mask.sum()
        state["base_count"] += n_base
        state["eagle_count"] += base_mask.numel() - n_base

        prev_p.copy_(p.data)
        prev_g.copy_(grad)
        p.data.sub_(update)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        clean = getattr(self, "clean_step", False)
        self.clean_step = False

        ext = None
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            coeff = lr if group["use_lr_in_eagle_update"] else 1.0

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("EAGLE does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    self._init_state(p, group)

                grad = p.grad
                if (group["adaptive_threshold"] and group["snr_gate"] is None
                        and state["step"] > 0):
                    self._update_adaptive_threshold(state, grad)

                state["step"] += 1

                # EAGLE 係数: 通常は定数、coeff_anneal_steps 指定時は
                # 1.0 → lr の cosine スケジュール (対照実験用)
                if group["coeff_anneal_steps"] is not None:
                    prog = min(state["step"] / group["coeff_anneal_steps"], 1.0)
                    coeff_p = lr + (1.0 - lr) * 0.5 * (1 + math.cos(math.pi * prog))
                else:
                    coeff_p = coeff

                fused_ok = (self._use_fused(p, group)
                            and p.data.is_contiguous() and grad.is_contiguous())
                if fused_ok:
                    if ext is None:
                        ext = get_extension()
                    snr_c = group["snr_gate"] or 0.0
                    kappa = group["trust_kappa"] or 0.0
                    if group["base"] == "adam":
                        if group["cooldown_steps"] is not None:
                            jumped, cooldown = state["jumped"], state["cooldown"]
                            cds = group["cooldown_steps"]
                        else:
                            jumped = cooldown = self._dummy_uint8(p.device)
                            cds = 0
                        allow = int((not group["paired_secant"])
                                    or (state["step"] % 2 == 0))
                        if group["curvature_ema"] is not None:
                            h_ema = state["h_ema"]
                            beta_h = group["curvature_ema"]
                            use_ema, clean_i = 1, int(clean)
                        else:
                            h_ema = self._dummy_f32(p.device)
                            beta_h, use_ema, clean_i = 0.0, 0, 0
                        msnr = group["signal_gate"] or 0.0
                        if group["conf_gate"] is not None:
                            h_var, conf_c = state["h_var"], group["conf_gate"]
                        else:
                            h_var, conf_c = self._dummy_f32(p.device), 0.0
                        ext.eagle_step_adam(
                            p.data, grad, state["exp_avg"], state["exp_avg_sq"],
                            state["prev_param"], state["prev_grad"],
                            state["threshold_t"],
                            state["eagle_count"], state["base_count"],
                            jumped, cooldown, h_ema, h_var,
                            lr, beta1, beta2, group["eps"], group["weight_decay"],
                            1 - beta1 ** state["step"],
                            math.sqrt(1 - beta2 ** state["step"]), coeff_p,
                            snr_c, kappa, cds, allow, beta_h, use_ema, clean_i,
                            msnr, conf_c, int(group["momentum_jump"]),
                            int(group["always_jump"]))
                    else:
                        ext.eagle_step_sgd(
                            p.data, grad, state["momentum_buffer"],
                            state["prev_param"], state["prev_grad"],
                            state["threshold_t"],
                            state["eagle_count"], state["base_count"],
                            lr, group["momentum"], group["weight_decay"], coeff_p,
                            kappa)
                else:
                    if group["weight_decay"] != 0:
                        p.data.mul_(1 - lr * group["weight_decay"])
                    self._vectorized_step(p, grad, state, group, coeff_p,
                                          clean=clean)

        return loss


def eagle_update_ratio(optimizer):
    """全パラメータ更新のうち EAGLE 更新則が使われた割合を返す。EAGLE 以外は None。

    ここで初めてデバイス→ホストの同期が起きる (学習ループ中は同期なし)。
    """
    if not isinstance(optimizer, EAGLE):
        return None
    eagle_count = 0
    base_count = 0
    for group in optimizer.param_groups:
        for p in group["params"]:
            state = optimizer.state[p]
            if "eagle_count" in state:
                eagle_count += int(state["eagle_count"])
                base_count += int(state["base_count"])
    total = eagle_count + base_count
    return eagle_count / total if total > 0 else 0.0
