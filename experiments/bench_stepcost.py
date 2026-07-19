"""per-step コストのマイクロベンチ (論文 III-D の開示数値の生成元)。

adam (foreach 既定) / adam (fused=True) / KESTREL (fused) を
回帰 MLP スケールと SIREN スケールで計測し、
results/analysis/stepcost_bench.json に保存する
(ictai/gen_macros.py の Bench* マクロが読む)。

実行: python experiments/bench_stepcost.py
"""

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eagle.optim import EAGLE  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "analysis"
DEV = torch.device("cuda")

KESTREL_KW = dict(base="adam", adaptive_threshold=False, threshold=5e-4,
                  use_lr_in_eagle_update=False, cooldown_steps=20,
                  curvature_ema=0.8, always_jump=True)


class Sine(nn.Module):
    def forward(self, x):
        return torch.sin(30.0 * x)


def siren():
    torch.manual_seed(0)
    layers, dims = [], [2, 256, 256, 256, 1]
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(Sine())
    return nn.Sequential(*layers)


def mlp():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(8, 64), nn.Tanh(), nn.Linear(64, 1))


def bench(model_fn, X, y, opt_fn, warmup=30, iters=200, clean=False):
    m = model_fn().to(DEV)
    o = opt_fn(m.parameters())
    crit = nn.MSELoss()

    def step():
        if clean:
            o.clean_step = True
        o.zero_grad(set_to_none=True)
        crit(m(X), y).backward()
        o.step()

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        step()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def main():
    scales = {
        "regression": (mlp,
                       torch.randn(614, 8, device=DEV),
                       torch.randn(614, 1, device=DEV)),
        "siren": (siren,
                  torch.rand(262144, 2, device=DEV) * 2 - 1,
                  torch.rand(262144, 1, device=DEV)),
    }
    out = {}
    for scale, (mf, X, y) in scales.items():
        out[scale] = {
            "adam_foreach": bench(
                mf, X, y, lambda p: torch.optim.Adam(p, lr=1e-3)),
            "adam_fused": bench(
                mf, X, y, lambda p: torch.optim.Adam(p, lr=1e-3, fused=True)),
            "kestrel_fused": bench(
                mf, X, y, lambda p: EAGLE(p, lr=1e-3, **KESTREL_KW),
                clean=True),
        }
        print(scale, {k: f"{v:.3f}ms" for k, v in out[scale].items()})
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "stepcost_bench.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"-> {OUT / 'stepcost_bench.json'}")


if __name__ == "__main__":
    main()
