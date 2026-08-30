"""KESTREL の jump/bench 機構を観測するための診断プローブ (継続研究 TO DO 7)。

`EAGLE._vectorized_step` の末尾フック `group["_probe"]` に差し込み、更新に
一切関与せずにテンソルを読むだけで、以下を毎ステップ記録する。
アルゴリズムの新規実装は行わない (既存 4 フラグの挙動を観測するだけ)。

記録する量と、それが答える問い
------------------------------
[A] 発火と変位            : bench が jump の量をどう変えるか
    jump_rate             : EAGLE 更新になった座標の割合
    jump_ratio_mean/max   : |eagle_step| / |base_step| (jump した座標のみ)
    jump_abs_mean         : |eagle_step| の平均 (jump した座標のみ)

[B] bench の実際の負荷    : どの程度 fallback に依存しているか
    bench_rate            : cooldown で強制的にベース更新にされた座標の割合
    fail_rate             : このステップで失敗判定された座標の割合
    fail_rate_of_jumped   : 直前に jump した座標のうち失敗判定された割合
    refail_rate           : 直近 2K ステップ以内に再失敗した座標の割合
    recover_steps_mean    : 失敗から次の jump 再開までの平均ステップ数

[C] 曲率状態 (stale 診断) : 検出器の盲点を定量化する
    h_valid_rate          : 曲率 EMA が有効 (h>0) な座標の割合
    neg_secant_rate       : 新しい割線が h<=0 で棄却された座標の割合。
                            EMA には古い正値が残るため、これが stale の入口
    h_rel_change_mean     : 生の割線測定 h_new と更新後 h_ema の相対乖離
                            |h_new - h_ema| / |h_ema| (受理された座標のみ)。
                            曲率推定のばらつきの代理

[D] 検出の混同行列        : 条件 (4) が何を検出できるか (査読 4.2 / RQ2)
    直前ステップで jump した座標を、今ステップの勾配で 4 分類する。
    worse = |g_t| > |g_{t-1}|、flip = g_{t-1}·g_t < 0 とすると
      det_fail   : flip かつ worse  -> bench が検出する失敗
      undet_fail : flip でない かつ worse -> **sign-preserving な未検出失敗**
      false_alarm: flip かつ worse でない -> 検出したが実は悪化していない
      good_jump  : flip でない かつ worse でない -> 成功した jump
    このうち undet_fail が RQ2 の主対象 (stale curvature の盲点) である。

集計はテンソルごとの件数を足し合わせ、ステップ末に一度だけ割り算する。
各ステップの値はデバイス上に貯め、`finish()` で 1 回だけホストへ転送するので
学習ループ中のホスト同期は発生しない。

使い方:
    probe = JumpProbe(device)
    for g in optimizer.param_groups:
        g["_probe"] = probe          # fused=False で走らせること
    ...
    optimizer.step()
    probe.flush(step, loss=float(loss))   # ステップ末に 1 回
    ...
    diag = probe.finish()                 # {"metric": [値, ...], ...}
"""

import torch

EPS = 1e-12

# 件数として累積するキー。すべて「分子」で、対応する分母を DENOM で指定する。
COUNTERS = [
    ("n_jump", "n"), ("n_bench", "n"), ("n_fail", "n"),
    ("n_prev_jump", "n"), ("n_fail_of_jumped", "n_prev_jump"),
    ("n_refail", "n_fail"),
    ("n_h_valid", "n"), ("n_neg_secant", "n_clean_valid"),
    ("n_det_fail", "n_prev_jump"), ("n_undet_fail", "n_prev_jump"),
    ("n_false_alarm", "n_prev_jump"), ("n_good_jump", "n_prev_jump"),
]
# 比率として出力する名前 -> (分子, 分母)
RATIOS = {
    "jump_rate": ("n_jump", "n"),
    "bench_rate": ("n_bench", "n"),
    "fail_rate": ("n_fail", "n"),
    "fail_rate_of_jumped": ("n_fail_of_jumped", "n_prev_jump"),
    "refail_rate": ("n_refail", "n_fail"),
    "h_valid_rate": ("n_h_valid", "n"),
    "neg_secant_rate": ("n_neg_secant", "n_clean_valid"),
    "det_fail_rate": ("n_det_fail", "n_prev_jump"),
    "undet_fail_rate": ("n_undet_fail", "n_prev_jump"),
    "false_alarm_rate": ("n_false_alarm", "n_prev_jump"),
    "good_jump_rate": ("n_good_jump", "n_prev_jump"),
}
# 和/件数で平均を出す量
MEANS = {
    "jump_ratio_mean": ("s_jump_ratio", "n_jump"),
    "jump_abs_mean": ("s_jump_abs", "n_jump"),
    "h_rel_change_mean": ("s_h_rel", "n_h_accept"),
    "recover_steps_mean": ("s_recover", "n_recover"),
}
MAXES = ["jump_ratio_max"]

_ZERO_KEYS = (["n", "n_clean_valid", "n_h_accept", "n_recover"]
              + [k for k, _ in COUNTERS]
              + ["s_jump_ratio", "s_jump_abs", "s_h_rel", "s_recover"])


class JumpProbe:
    """EAGLE の更新内部を読み取る診断プローブ (更新には関与しない)。"""

    def __init__(self, device, cooldown_steps=20):
        self.device = device
        self.K = cooldown_steps or 0
        self.per_param = {}          # id(p) -> 座標ごとの履歴テンソル
        self.rows = []               # ステップごとの生カウンタ (デバイス上)
        self.scalars = []            # ステップごとの (step, loss)
        self._acc = None
        self._reset_acc()

    # ---------------------------------------------------------- 内部ユーティリティ
    def _reset_acc(self):
        z = torch.zeros((), device=self.device)
        self._acc = {k: z.clone() for k in _ZERO_KEYS}
        self._acc["jump_ratio_max"] = z.clone()

    def _slot(self, p, key, dtype):
        d = self.per_param.setdefault(id(p), {})
        if key not in d:
            d[key] = torch.zeros_like(p.data, dtype=dtype)
        return d[key]

    # ------------------------------------------------------------------ フック
    def __call__(self, p, state, group, ctx):
        a = self._acc
        step = state["step"]
        jumped = ~ctx["base_mask"]
        prev_jump = self._slot(p, "prev_jump", torch.bool)
        last_fail = self._slot(p, "last_fail", torch.int32)
        last_fail_neg = self._slot(p, "last_fail_neg", torch.int32)

        a["n"] += jumped.numel()
        a["n_jump"] += jumped.sum()

        # [A] 変位: jump した座標の |eagle_step| と |base_step| 比。
        # ホスト同期を避けるため boolean indexing ではなく where で潰す。
        js = ctx["eagle_step"].abs()
        ratio = js / (ctx["base_step"].abs() + EPS)
        zero = torch.zeros((), device=js.device, dtype=js.dtype)
        a["s_jump_ratio"] += torch.where(jumped, ratio, zero).sum()
        a["s_jump_abs"] += torch.where(jumped, js, zero).sum()
        a["jump_ratio_max"] = torch.maximum(
            a["jump_ratio_max"], torch.where(jumped, ratio, zero).max())

        # [B] bench の負荷
        if ctx["force_base"] is not None:
            a["n_bench"] += ctx["force_base"].sum()
        if ctx["failed"] is not None:
            failed = ctx["failed"]
            a["n_fail"] += failed.sum()
            a["n_fail_of_jumped"] += (failed & prev_jump).sum()
            if self.K:
                recent = (step - last_fail) <= 2 * self.K
                a["n_refail"] += (failed & recent & (last_fail > 0)).sum()
            # 失敗 -> 次に jump を再開するまでのステップ数
            resumed = jumped & (last_fail_neg > 0)
            a["s_recover"] += torch.where(
                resumed, step - last_fail_neg,
                torch.zeros_like(last_fail_neg)).sum()
            a["n_recover"] += resumed.sum()
            last_fail_neg = torch.where(
                resumed, torch.zeros_like(last_fail_neg), last_fail_neg)
            last_fail = torch.where(failed, torch.full_like(last_fail, step),
                                    last_fail)
            last_fail_neg = torch.where(
                failed, torch.full_like(last_fail_neg, step), last_fail_neg)
            self.per_param[id(p)]["last_fail"] = last_fail
            self.per_param[id(p)]["last_fail_neg"] = last_fail_neg

        # [C] 曲率状態と stale の入口
        he = ctx["h_ema"]
        if he is not None:
            a["n_h_valid"] += (he > 0).sum()
        if ctx["clean"] and ctx["h_new"] is not None:
            hn, acc = ctx["h_new"], ctx["h_accept"]
            valid = ctx["delta_param"].abs() > 1e-12
            a["n_clean_valid"] += valid.sum()
            a["n_neg_secant"] += (valid & ~acc).sum()
            rel = (hn - he).abs() / (he.abs() + EPS)
            a["s_h_rel"] += torch.where(acc, rel,
                                        torch.zeros_like(rel)).sum()
            a["n_h_accept"] += acc.sum()

        # [D] 直前に jump した座標の帰結 (混同行列)
        pg, g = ctx["prev_grad"], ctx["grad"]
        worse = g.abs() > pg.abs()
        flip = pg * g < 0
        pj = prev_jump
        a["n_prev_jump"] += pj.sum()
        a["n_det_fail"] += (pj & flip & worse).sum()
        a["n_undet_fail"] += (pj & ~flip & worse).sum()
        a["n_false_alarm"] += (pj & flip & ~worse).sum()
        a["n_good_jump"] += (pj & ~flip & ~worse).sum()

        prev_jump.copy_(jumped)

    # ------------------------------------------------------------ ステップ末
    def flush(self, step, loss=None, psnr=None):
        """1 ステップ分の累積を確定する。ホスト同期は起こさない。"""
        keys = sorted(self._acc)
        self.rows.append(torch.stack([self._acc[k].float() for k in keys]))
        self._keys = keys
        self.scalars.append((step, loss, psnr))
        self._reset_acc()

    def finish(self):
        """デバイス→ホストへ 1 回だけ転送し、比率・平均に直して返す。"""
        if not self.rows:
            return {}
        raw = torch.stack(self.rows).cpu().numpy()
        cols = {k: raw[:, i] for i, k in enumerate(self._keys)}
        out = {"step": [s for s, _, _ in self.scalars],
               "loss": [l for _, l, _ in self.scalars],
               "psnr": [q for _, _, q in self.scalars]}
        for name, (num, den) in RATIOS.items():
            d = cols[den]
            out[name] = [float(n / x) if x > 0 else None
                         for n, x in zip(cols[num], d)]
        for name, (num, den) in MEANS.items():
            d = cols[den]
            out[name] = [float(n / x) if x > 0 else None
                         for n, x in zip(cols[num], d)]
        for name in MAXES:
            out[name] = [float(v) for v in cols[name]]
        out["_counts"] = {k: [float(v) for v in cols[k]] for k in cols}
        return out
