"""INR (SIREN 画像フィッティング) パイロット — ドメイン選定の de-risk 用。

目的: フルバッチ回帰ハーネスで生き残った EAGLE の効き
(中終盤の到達 speedup) が、INR フィッティング
(数十万パラメータ × フルバッチ MSE × 滑らかな周期活性化) に転移するかを
1 シードで素早く確認する。本実験に昇格させる場合はプロトコル v2
(lr グリッド × 3 シード) に載せ替えること。

タスク: 512x512 画像 (skimage.data) の座標→画素値回帰。
モデル: SIREN (sine 活性化、omega_0=30、標準初期化)。
指標: PSNR (画素 [0,1] 換算) vs steps / grad_evals / wall-clock。

実行例:
    python experiments/pilot_inr.py --image camera --steps 2000
    python experiments/pilot_inr.py --image astronaut --steps 2000
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_comparison import (GRAD_EVALS_PER_STEP, NEEDS_CREATE_GRAPH,  # noqa: E402
                            OPTIMIZER_BUILDERS, PAIR_EVERY)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


class Sine(nn.Module):
    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class Siren(nn.Module):
    """標準 SIREN (Sitzmann+ 2020)。第 1 層 w0=30、以降 w0=1 で
    重みを 30 倍スケールの一様分布で初期化する。"""

    def __init__(self, in_dim=2, hidden=256, depth=3, out_dim=1, w0=30.0):
        super().__init__()
        layers = []
        dim = in_dim
        for i in range(depth):
            lin = nn.Linear(dim, hidden)
            with torch.no_grad():
                if i == 0:
                    lin.weight.uniform_(-1.0 / in_dim, 1.0 / in_dim)
                else:
                    bound = math.sqrt(6.0 / dim) / w0
                    lin.weight.uniform_(-bound, bound)
            layers += [lin, Sine(w0)]
            dim = hidden
        head = nn.Linear(dim, out_dim)
        with torch.no_grad():
            bound = math.sqrt(6.0 / dim) / w0
            head.weight.uniform_(-bound, bound)
        layers.append(head)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def load_image(name):
    """skimage 内蔵画像 (camera/astronaut)、Kodak スイート (kodim01〜kodim24、
    data/kodak/、768×512 RGB)、または DIV2K 16 枚 (div2k0801〜div2k0816、
    data/div2k16/、768×512 に縮小済み)。"""
    if name.startswith("kodim"):
        from skimage.io import imread
        img = imread(DATA_DIR / "kodak" / f"{name}.png")
    elif name.startswith("div2k"):
        # 継続研究 A: DIV2K validation 先頭 16 枚を Kodak と同じ 768x512 に
        # 揃えたもの (experiments/prepare_div2k.py で生成)
        from skimage.io import imread
        img = imread(DATA_DIR / "div2k16" / f"{name}.png")
    else:
        from skimage import data as skdata
        img = getattr(skdata, name)()
    img = np.asarray(img, dtype=np.float32) / 255.0
    if img.ndim == 2:
        img = img[..., None]
    return img


def make_batch(img, device):
    h, w, c = img.shape
    ys = torch.linspace(-1, 1, h)
    xs = torch.linspace(-1, 1, w)
    grid = torch.stack(torch.meshgrid(ys, xs, indexing="ij"), dim=-1)
    coords = grid.reshape(-1, 2).to(device)
    target = torch.from_numpy(img).reshape(-1, img.shape[-1]).to(device)
    target = target * 2.0 - 1.0  # SIREN 慣例の [-1,1]
    return coords, target


def psnr_from_mse(mse):
    # target は [-1,1] なので [0,1] 換算の MSE は /4
    return -10.0 * math.log10(max(mse / 4.0, 1e-12))


def train_one(opt_name, lr, coords, target, steps, seed, device, eval_every,
              warmup_steps=10):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = Siren(out_dim=target.shape[-1]).to(device)
    builder = OPTIMIZER_BUILDERS[opt_name]
    optimizer = builder(model.parameters(), lr, steps)
    criterion = nn.MSELoss()
    is_lbfgs = opt_name == "lbfgs"
    clean = opt_name in PAIR_EVERY  # フルバッチ → 毎ステップがクリーン測定
    needs_cg = opt_name in NEEDS_CREATE_GRAPH  # adahessian の Hutchinson HVP
    gpe = GRAD_EVALS_PER_STEP.get(opt_name, 1)

    # 計測前ウォームアップ (プロトコル v2): JIT/cuDNN 初期化の実行順交絡を
    # 除去する。使い捨てのモデル+オプティマイザで同種カーネルを起動する
    # だけで、本計測の軌道には影響しない (実モデルは初期化済み)
    if warmup_steps > 0:
        wm = Siren(out_dim=target.shape[-1]).to(device)
        wopt = builder(wm.parameters(), lr, steps)
        wc, wt = coords[:4096], target[:4096]
        for _ in range(warmup_steps):
            if is_lbfgs:
                def wclosure():
                    wopt.zero_grad(set_to_none=True)
                    wl = criterion(wm(wc), wt)
                    wl.backward()
                    return wl
                wopt.step(wclosure)
            else:
                wopt.zero_grad(set_to_none=True)
                wl = criterion(wm(wc), wt)
                wl.backward(create_graph=needs_cg)
                if clean:
                    wopt.clean_step = True
                wopt.step()
        torch.cuda.synchronize()

    grad_evals = 0
    hist = {"steps": [], "psnr": [], "time": [], "grad_evals": []}
    log_pts = sorted({0, 1, 2, 3, 4, 6, 8, 11, 16, 23, 32, 45, 64, 91, 128}
                     | {i for i in range(0, steps + 1, eval_every)} | {steps})

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for step in range(steps + 1):
        if step in log_pts:
            torch.cuda.synchronize()
            t = time.perf_counter() - t0
            with torch.no_grad():
                mse = criterion(model(coords), target).item()
            hist["steps"].append(step)
            hist["psnr"].append(psnr_from_mse(mse))
            hist["time"].append(t)
            hist["grad_evals"].append(grad_evals)
            torch.cuda.synchronize()
            t0 -= (time.perf_counter() - t0 - t)  # 評価時間を計測から除外
        if step == steps:
            break

        if is_lbfgs:
            def closure():
                nonlocal grad_evals
                grad_evals += 1
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(coords), target)
                loss.backward()
                return loss
            optimizer.step(closure)
        else:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(coords), target)
            loss.backward(create_graph=needs_cg)
            grad_evals += gpe
            if clean:
                optimizer.clean_step = True
            optimizer.step()
    return hist


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", default="camera")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--optimizers", nargs="+",
                   default=["adam", "eagle3", "eagle3-noins", "eagle4-aj",
                            "eagle-dqn", "lbfgs"])
    p.add_argument("--lrs", nargs="+", type=float,
                   default=[1e-4, 3e-4, 1e-3, 3e-3])
    p.add_argument("--lbfgs-lrs", nargs="+", type=float,
                   default=[0.1, 0.3, 1.0])
    p.add_argument("--adahessian-lrs", nargs="+", type=float,
                   default=[1e-2, 3e-2, 1e-1, 3e-1],
                   help="adahessian 専用 lr グリッド (Adam と桁が違うため。"
                        "論文既定は 0.1)")
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--prefix", default="pilot_inr",
                   help="結果ディレクトリの prefix (本実験は inrv2 等にして"
                        "パイロットの上書きを防ぐ)")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    img = load_image(args.image)
    coords, target = make_batch(img, device)
    out_dir = RESULTS_DIR / f"{args.prefix}_{args.image}_s{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_hist = {}
    for opt_name in args.optimizers:
        if opt_name == "lbfgs":
            lrs = args.lbfgs_lrs
        elif opt_name == "adahessian":
            lrs = args.adahessian_lrs
        else:
            lrs = args.lrs
        for lr in lrs:
            key = f"{opt_name}@{lr:g}"
            print(f"=== {key} ===", flush=True)
            h = train_one(opt_name, lr, coords, target, args.steps,
                          args.seed, device, args.eval_every,
                          warmup_steps=args.warmup_steps)
            all_hist[key] = h
            print(f"  final PSNR {h['psnr'][-1]:.2f} dB, "
                  f"time {h['time'][-1]:.1f}s, "
                  f"grad_evals {h['grad_evals'][-1]}", flush=True)

    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"args": vars(args), "histories": all_hist}, f, indent=2)

    # 家族エンベロープ: 到達 PSNR しきい値までの steps / grad_evals / time
    fams = sorted({k.split("@")[0] for k in all_hist})
    thresholds = [25, 28, 30, 32, 35]
    print(f"\n{'family':<14}" + "".join(f"{t:>7}dB" for t in thresholds)
          + f"{'best':>8}")
    for basis in ("steps", "grad_evals", "time"):
        print(f"--- 到達 {basis} (家族エンベロープ = 家族内最速 lr) ---")
        for fam in fams:
            row = f"{fam:<14}"
            best_psnr = max(max(all_hist[k]["psnr"]) for k in all_hist
                            if k.startswith(fam + "@"))
            for t in thresholds:
                v = None
                for k in all_hist:
                    if not k.startswith(fam + "@"):
                        continue
                    h = all_hist[k]
                    for x, ps in zip(h[basis], h["psnr"]):
                        if ps >= t:
                            v = x if v is None else min(v, x)
                            break
                row += f"{v:>9.0f}" if isinstance(v, (int, float)) \
                    and basis != "time" else (f"{v:>9.1f}" if v else f"{'---':>9}")
            print(row + f"{best_psnr:>8.2f}")
    print(f"\n結果を保存: {out_dir}")


if __name__ == "__main__":
    main()
