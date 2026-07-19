"""KESTREL の INR 失敗モード診断 (ICTAI 査読 W3/W4 対応)。

camera (成功例) と astronaut (負け筋) で、family-best lr の KESTREL を
非 fused (状態introspection可能) で走らせ、以下を記録する:
- ジャンプ率 / ベンチ中率の時間推移 (W3: 失敗モードの機構比較)
- ジャンプ幅 |Δθ| の分布 (W4: 大ジャンプの有界性の実証)

出力: results/analysis/kestrel_inr_diag.json (+ 標準出力サマリー)。
図は ictai/paper_figs.py (fig_diag) が JSON から生成する。

実行: python experiments/diagnose_kestrel_inr.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pilot_inr import Siren, load_image, make_batch  # noqa: E402
from run_comparison import OPTIMIZER_BUILDERS  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT = RESULTS / "analysis" / "kestrel_inr_diag.json"
STEPS = 2000
EVERY = 20          # 率の記録間隔
MAG_EVERY = 10      # ジャンプ幅サンプリング間隔
BINS = np.logspace(-8, 2, 51)   # |Δθ| の対数ビン


def best_lr(image):
    """inrv3 の family エンベロープ (最終 PSNR 最大) の lr を返す。"""
    best, best_psnr = None, -np.inf
    for p in sorted(RESULTS.glob(f"inrv3_{image}_s*/metrics.json")):
        h = json.load(open(p))["histories"]
        for k, hh in h.items():
            fam, lr = k.split("@")
            if fam != "eagle-dqn-cd":
                continue
            m = max(hh["psnr"])
            if m > best_psnr:
                best_psnr, best = m, float(lr)
    return best


def run(image):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lr = best_lr(image)
    img = load_image(image)
    coords, target = make_batch(img, device)
    torch.manual_seed(42)
    np.random.seed(42)
    model = Siren(out_dim=target.shape[-1]).to(device)
    opt = OPTIMIZER_BUILDERS["eagle-dqn-cd"](model.parameters(), lr, STEPS)
    for g in opt.param_groups:
        g["fused"] = False   # 状態 introspection のため
    crit = nn.MSELoss()
    params = [p for p in model.parameters()]
    n_total = sum(p.numel() for p in params)

    rec = {"lr": lr, "steps": [], "jump_frac": [], "bench_frac": [],
           "hist": np.zeros(len(BINS) - 1), "n_mag_samples": 0}
    win_jump = 0
    prev_counts = 0
    for step in range(STEPS):
        opt.clean_step = True
        opt.zero_grad(set_to_none=True)
        crit(model(coords), target).backward()
        snap = ([p.detach().clone() for p in params]
                if step % MAG_EVERY == 0 else None)
        opt.step()

        # ジャンプ数 (このステップ) とベンチ中座標数
        jumped = benched = 0
        for p in params:
            st = opt.state[p]
            if "jumped" in st:
                jumped += int(st["jumped"].sum())
                benched += int((st["cooldown"] > 0).sum())
        win_jump += jumped
        if snap is not None and jumped > 0:
            mags = []
            for p, s0 in zip(params, snap):
                st = opt.state[p]
                m = st["jumped"].bool()
                if m.any():
                    mags.append((p.detach() - s0).abs()[m].flatten())
            mags = torch.cat(mags).float().cpu().numpy()
            h, _ = np.histogram(mags, bins=BINS)
            rec["hist"] += h
            rec["n_mag_samples"] += len(mags)

        if (step + 1) % EVERY == 0:
            rec["steps"].append(step + 1)
            rec["jump_frac"].append(win_jump / (EVERY * n_total))
            rec["bench_frac"].append(benched / n_total)
            win_jump = 0

    rec["hist"] = rec["hist"].tolist()
    q = np.array(rec["hist"]).cumsum()
    print(f"{image}: lr={lr:g}, mean jump {np.mean(rec['jump_frac'])*100:.2f}%,"
          f" mean bench {np.mean(rec['bench_frac'])*100:.2f}%,"
          f" mag samples {rec['n_mag_samples']}")
    return rec


def main():
    out = {"bins": BINS.tolist()}
    for image in ("camera", "astronaut"):
        out[image] = run(image)
    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
