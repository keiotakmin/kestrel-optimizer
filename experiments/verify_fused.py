"""fused カーネルと vectorized 経路の同等性検証 (継続研究 TO DO 3)。

主張する比較単位ごとに実装条件が対称であることを保証するための前提確認。

(1) このホストで fused カーネルが実際にロードされるか (黙って
    vectorized にフォールバックしていないか)
(2) KESTREL の設定 (EMA + 常時ジャンプ + K=20 ベンチ) で、fused と
    vectorized が同じ軌道を出すか。TO DO 7 Stage B の診断は vectorized
    経路でしか採れないため、これが成り立たないと診断値を fused の挙動の
    説明に使えない。
(3) SIREN 規模での 1 ステップ実測時間を、実装クラスを明示して並べる
    (fused / foreach / python)。

実行: python experiments/verify_fused.py
"""

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eagle._cuda import get_extension  # noqa: E402
from eagle.optim import EAGLE  # noqa: E402
from expand_grid_kodak import run_metadata  # noqa: E402
from pilot_inr import Siren  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "analysis"
KESTREL_KW = dict(base="adam", adaptive_threshold=False, threshold=5e-4,
                  use_lr_in_eagle_update=False, cooldown_steps=20,
                  curvature_ema=0.8, always_jump=True)


def make(device, seed=0):
    torch.manual_seed(seed)
    model = Siren(out_dim=1).to(device)
    torch.manual_seed(seed + 1)
    X = (torch.rand(65536, 2, device=device) * 2 - 1)
    y = torch.rand(65536, 1, device=device)
    return model, X, y


def trajectory(device, fused, steps=120, seed=0):
    model, X, y = make(device, seed)
    opt = EAGLE(model.parameters(), lr=1e-3, fused=fused, **KESTREL_KW)
    crit = nn.MSELoss()
    losses = []
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = crit(model(X), y)
        loss.backward()
        losses.append(float(loss))
        opt.clean_step = True
        opt.step()
    return losses, [p.detach().clone() for p in model.parameters()]


def single_step_equivalence(device, checkpoints=(1, 2, 5, 10, 20, 50),
                            seed=0):
    """同一状態から 1 ステップだけ進めて fused と vectorized を比較する。

    KESTREL は初期条件に極端に敏感 (1 ULP の差が数ステップで O(10) に育つ)
    ため、軌道同士の比較では実装差とカオス増幅を区別できない。軌道上の
    複数点でスナップショットを取り、そこから 1 ステップだけ両経路を回す。
    """
    import copy
    model, X, y = make(device, seed)
    opt = EAGLE(model.parameters(), lr=1e-3, fused=False, **KESTREL_KW)
    crit = nn.MSELoss()
    worst_abs = worst_rel = worst_loss = 0.0

    def one_step(m, o):
        o.zero_grad(set_to_none=True)
        loss = crit(m(X), y)
        loss.backward()
        o.clean_step = True
        o.step()
        return float(loss)

    for t in range(1, max(checkpoints) + 1):
        if t in checkpoints:
            # 現在の状態を 2 つに複製し、片方を fused、片方を vectorized で
            # 1 ステップ進める
            outs = []
            for fused in (True, False):
                m2 = copy.deepcopy(model)
                o2 = EAGLE(m2.parameters(), lr=1e-3, fused=fused,
                           **KESTREL_KW)
                for p_old, p_new in zip(model.parameters(), m2.parameters()):
                    st = opt.state[p_old]
                    o2.state[p_new] = {
                        k: (v.clone() if torch.is_tensor(v) else v)
                        for k, v in st.items()}
                outs.append((m2, one_step(m2, o2)))
            (mf, lf), (mv, lv) = outs
            for a, b in zip(mf.parameters(), mv.parameters()):
                d = float((a - b).abs().max())
                worst_abs = max(worst_abs, d)
                worst_rel = max(worst_rel,
                                d / (float(b.abs().max()) + 1e-12))
            worst_loss = max(worst_loss, abs(lf - lv) / max(abs(lv), 1e-12))
        one_step(model, opt)
    return worst_abs, worst_rel, worst_loss


def steptime(device, opt_fn, clean=False, warmup=30, iters=200):
    model, X, y = make(device)
    opt = opt_fn(model.parameters())
    crit = nn.MSELoss()

    def one():
        if clean:
            opt.clean_step = True
        opt.zero_grad(set_to_none=True)
        crit(model(X), y).backward()
        opt.step()

    for _ in range(warmup):
        one()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        one()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-timing", action="store_true",
                    help="(3) の実測を行わない。GPU が他ジョブと共有されて"
                         "いるホストでは時間は無意味なので数値一致だけを見る")
    args = ap.parse_args()
    device = torch.device("cuda")
    meta = run_metadata()
    print(f"# verify_fused  {meta}\n")

    ext = get_extension()
    print(f"(1) fused 拡張のロード: {'OK' if ext is not None else '失敗'}")
    if ext is None:
        print("    -> このホストでは fused 経路が使えない。"
              "time 基準の主張はここでは作らないこと。")
        return

    print("\n(2) KESTREL 設定での fused vs vectorized (単一ステップ照合)")
    print("    軌道の突き合わせは使えない。この最適化器は 1 ULP の初期摂動でも")
    print("    数ステップで O(10) 発散する (verify_fused の注記参照)。したがって")
    print("    同一状態から 1 ステップだけ進めて両経路を比較する。")
    dpar, rel, dloss = single_step_equivalence(device)
    print(f"    1 ステップ後のパラメータ最大絶対差: {dpar:.3e} "
          f"(相対 {rel:.3e})")
    ok = rel < 1e-5
    print(f"    判定: {'実装は一致 (fp32 の丸め差のみ)' if ok else '不一致 — 実装差の疑い'}")

    if args.no_timing:
        print("\n(3) --no-timing: 実測は行わない "
              "(contention のあるホストで time を測らないため)")
        OUT.mkdir(parents=True, exist_ok=True)
        path = OUT / "fused_equivalence.json"
        path.write_text(json.dumps({
            "metadata": meta, "extension_loaded": True,
            "equivalence": {"max_rel_loss_diff": dloss,
                            "max_abs_param_diff": dpar,
                            "max_rel_param_diff": rel, "pass": ok}},
            ensure_ascii=False, indent=2))
        print(f"-> {path}")
        return

    print("\n(3) SIREN 規模の 1 ステップ実測 (実装クラスを明示)")
    rows = [
        ("adam (foreach 既定)", "foreach",
         lambda p: torch.optim.Adam(p, lr=1e-3), False),
        ("adam (fused=True)", "fused",
         lambda p: torch.optim.Adam(p, lr=1e-3, fused=True), False),
        ("adamw (fused=True)", "fused",
         lambda p: torch.optim.AdamW(p, lr=1e-3, fused=True), False),
        ("KESTREL (fused)", "fused",
         lambda p: EAGLE(p, lr=1e-3, fused=True, **KESTREL_KW), True),
        ("KESTREL (vectorized)", "python",
         lambda p: EAGLE(p, lr=1e-3, fused=False, **KESTREL_KW), True),
        ("rprop", "python",
         lambda p: torch.optim.Rprop(p, lr=1e-3), False),
    ]
    times = {}
    print(f"    {'optimizer':<24}{'impl':<10}{'ms/step':>9}")
    for name, impl, fn, clean in rows:
        t = steptime(device, fn, clean=clean)
        times[name] = {"impl": impl, "ms_per_step": t}
        print(f"    {name:<24}{impl:<10}{t:>9.3f}")

    kf = times["KESTREL (fused)"]["ms_per_step"]
    af = times["adam (fused=True)"]["ms_per_step"]
    ao = times["adam (foreach 既定)"]["ms_per_step"]
    print(f"\n    fused 同士の per-step 比 (KESTREL/adam): {kf / af:.3f}")
    print(f"    非対称な比較 (KESTREL fused / adam foreach): {kf / ao:.3f}"
          "  <- この数字は headline に使わない")

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "fused_verification.json"
    path.write_text(json.dumps({
        "metadata": meta, "extension_loaded": True,
        "equivalence": {"max_rel_loss_diff": dloss,
                        "max_abs_param_diff": dpar,
                        "max_rel_param_diff": rel, "pass": ok},
        "steptime_ms": times,
        "ratio_fused_vs_fused": kf / af,
        "ratio_asymmetric": kf / ao}, ensure_ascii=False, indent=2))
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
