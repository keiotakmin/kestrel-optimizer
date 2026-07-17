"""新実装 (vectorized / fused) が旧ループ実装と同じ更新を行うことの検証。

実行: python tests/test_equivalence.py
"""

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eagle.optim import EAGLE  # noqa: E402

# 旧実装 (書き換え前の optim.py のコピー) をロード
_ref_path = Path(__file__).parent / "reference_optim.py"
_spec = importlib.util.spec_from_file_location("reference_optim", _ref_path)
_ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ref)
ReferenceEAGLE = _ref.EAGLE


def make_model(seed, device):
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(20, 64), nn.ReLU(),
        nn.Linear(64, 64), nn.ReLU(),
        nn.Linear(64, 5),
    ).to(device)


def run_steps(model, optimizer, batches, device):
    criterion = nn.CrossEntropyLoss()
    snapshots = []
    for X, y in batches:
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
        snapshots.append([p.detach().clone() for p in model.parameters()])
    return snapshots


def max_diff(snaps_a, snaps_b):
    worst = 0.0
    for params_a, params_b in zip(snaps_a, snaps_b):
        for a, b in zip(params_a, params_b):
            worst = max(worst, (a - b).abs().max().item())
    return worst


def compare_trajectory(config, device, n_steps=30, tol=2e-5):
    """reference と vectorized は数式・演算順序まで同一なので軌道全体を厳密比較する。"""
    torch.manual_seed(123)
    batches = [(torch.randn(32, 20, device=device),
                torch.randint(0, 5, (32,), device=device))
               for _ in range(n_steps)]

    results = {}
    for name, build in {
        "reference": lambda ps: ReferenceEAGLE(ps, **config),
        "vectorized": lambda ps: EAGLE(ps, fused=False, **config),
    }.items():
        model = make_model(0, device)
        results[name] = run_steps(model, build(model.parameters()), batches, device)

    d = max_diff(results["reference"], results["vectorized"])
    ok = d <= tol
    print(f"  reference vs vectorized (軌道): max diff = {d:.2e}  "
          f"[{'OK' if ok else 'NG'}]")
    return ok


def _sync_state(opt_src, opt_dst, params_src, params_dst):
    for ps, pd in zip(params_src, params_dst):
        src = opt_src.state[ps]
        dst = opt_dst.state[pd]
        for k, v in src.items():
            if torch.is_tensor(v):
                if k in dst:
                    dst[k].copy_(v)
                else:
                    dst[k] = v.clone()
            else:
                dst[k] = v


def compare_single_step(config, device, n_steps=30, tol=1e-5):
    """vectorized vs fused の 1 ステップ等価性。

    浮動小数点の丸め順序 (FMA 縮約) の微小差は学習軌道上で増幅されるため
    軌道全体のビット一致は期待できない (PyTorch 純正の foreach/fused Adam も
    同様)。そこで毎ステップ、fused 側の状態を vectorized 側に同期してから
    同一の勾配で 1 ステップ進め、更新結果を比較する。

    誤差はパラメータスケールで正規化した相対値で測る (保険なし dqn などの
    発散型バリアントは軌道が数値爆発し、絶対誤差は丸め 1ulp でも巨大に
    なるため)。nan に達したら以降は両経路とも nan なので比較を打ち切る。"""
    criterion = nn.CrossEntropyLoss()
    torch.manual_seed(123)

    model_v = make_model(0, device)
    model_f = make_model(0, device)
    opt_v = EAGLE(model_v.parameters(), fused=False, **config)
    opt_f = EAGLE(model_f.parameters(), fused=True, **config)
    params_v = list(model_v.parameters())
    params_f = list(model_f.parameters())

    worst = 0.0
    count_mismatch = 0
    for i in range(n_steps):
        X = torch.randn(32, 20, device=device)
        y = torch.randint(0, 5, (32,), device=device)

        # vectorized 側で勾配を計算し、パラメータ・勾配・状態を fused 側へ同期
        opt_v.zero_grad()
        criterion(model_v(X), y).backward()
        for pv, pf in zip(params_v, params_f):
            pf.data.copy_(pv.data)
            pf.grad = pv.grad.clone()
        if i > 0:
            _sync_state(opt_v, opt_f, params_v, params_f)

        if config.get("curvature_ema"):
            clean = (i % 3 == 2)  # 3 ステップに 1 回クリーン測定を模擬
            opt_v.clean_step = clean
            opt_f.clean_step = clean

        opt_v.step()
        opt_f.step()

        if not all(torch.isfinite(pv.data).all() for pv in params_v):
            break
        for pv, pf in zip(params_v, params_f):
            scale = 1.0 + pv.data.abs().max().item()
            worst = max(worst, (pv.data - pf.data).abs().max().item() / scale)
        for pv, pf in zip(params_v, params_f):
            sv, sf = opt_v.state[pv], opt_f.state[pf]
            count_mismatch += abs(int(sv["eagle_count"]) - int(sf["eagle_count"]))

    ok = worst <= tol and count_mismatch == 0
    print(f"  vectorized vs fused (1step同期): max rel diff = {worst:.2e}, "
          f"マスク不一致 = {count_mismatch}  [{'OK' if ok else 'NG'}]")
    return ok


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    all_ok = True
    cases = [
        dict(lr=0.01, base="adam"),
        dict(lr=0.01, base="adam", weight_decay=1e-4),
        dict(lr=0.01, base="adam", use_lr_in_eagle_update=False),  # 卒論オリジナル
        dict(lr=0.01, base="adam", adaptive_threshold=True),
        dict(lr=0.01, base="sgd", momentum=0.9),
        dict(lr=0.01, base="sgd", momentum=0.9, weight_decay=1e-4),
        # B1/B2 改良 (旧実装には無い機能のため fused vs vectorized のみ)
        dict(lr=0.01, base="adam", use_lr_in_eagle_update=False, snr_gate=0.5),
        dict(lr=0.01, base="adam", use_lr_in_eagle_update=False, trust_kappa=50),
        dict(lr=0.01, base="adam", use_lr_in_eagle_update=False,
             snr_gate=0.5, trust_kappa=50),
        dict(lr=0.01, base="sgd", momentum=0.9, trust_kappa=50),
        # B3 (クールダウン) / アニーリング対照
        dict(lr=0.01, base="adam", use_lr_in_eagle_update=False,
             trust_kappa=50, cooldown_steps=5),
        dict(lr=0.01, base="adam", use_lr_in_eagle_update=False,
             trust_kappa=50, coeff_anneal_steps=100),
        # ペアバッチ割線
        dict(lr=0.01, base="adam", use_lr_in_eagle_update=False,
             paired_secant=True),
        dict(lr=0.01, base="adam", use_lr_in_eagle_update=False,
             trust_kappa=50, cooldown_steps=20, paired_secant=True),
        # 間欠ペア + 曲率 EMA
        dict(lr=0.01, base="adam", use_lr_in_eagle_update=False,
             trust_kappa=50, cooldown_steps=20, curvature_ema=0.8),
        # SNR 発火ゲート (G1: Δg ベース / G2: モーメンタム SNR)
        dict(lr=0.01, base="adam", use_lr_in_eagle_update=False,
             trust_kappa=50, cooldown_steps=20, curvature_ema=0.8,
             snr_gate=0.5),
        dict(lr=0.01, base="adam", use_lr_in_eagle_update=False,
             trust_kappa=50, cooldown_steps=20, curvature_ema=0.8,
             signal_gate=0.5),
        # 曲率信頼度ゲート / ジャンプ分子 m̂ 化
        dict(lr=0.01, base="adam", use_lr_in_eagle_update=False,
             trust_kappa=50, cooldown_steps=20, curvature_ema=0.8,
             conf_gate=1.0),
        dict(lr=0.01, base="adam", use_lr_in_eagle_update=False,
             trust_kappa=50, cooldown_steps=20, curvature_ema=0.8,
             conf_gate=1.0, momentum_jump=True),
        # 常時ジャンプ (振動切替なし): 裸 dqn / dqn-cd (卒論最終形) / eagle4-aj
        dict(lr=0.01, base="adam", use_lr_in_eagle_update=False,
             curvature_ema=0.8, always_jump=True),
        dict(lr=0.01, base="adam", use_lr_in_eagle_update=False,
             cooldown_steps=20, curvature_ema=0.8, always_jump=True),
        dict(lr=0.01, base="adam", use_lr_in_eagle_update=False,
             trust_kappa=50, cooldown_steps=20, curvature_ema=0.8,
             signal_gate=0.25, always_jump=True),
    ]
    for config in cases:
        print(f"config: {config}")
        # 旧実装に無い機能は reference との軌道比較をスキップ
        new_features = ("adaptive_threshold", "snr_gate", "trust_kappa",
                        "cooldown_steps", "coeff_anneal_steps", "paired_secant",
                        "curvature_ema", "signal_gate", "conf_gate",
                        "momentum_jump", "always_jump")
        if not any(config.get(k) for k in new_features):
            all_ok &= compare_trajectory(config, device)
        if device.type == "cuda":
            all_ok &= compare_single_step(config, device)

    print("\n" + ("ALL TESTS PASSED" if all_ok else "SOME TESTS FAILED"))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
