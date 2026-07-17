"""INR 実験 (pilot_inr.py / run_inr_v2.sh) の集計。

(image, seed) ごとに家族 lr エンベロープの「到達 PSNR しきい値までの
steps / grad_evals / time」を求め、vs adam・vs lbfgs のエンベロープ比を
シード mean±SD で表にする。最終 best PSNR も家族エンベロープで併記。

実行: python experiments/analyze_inr.py --prefix inrv2
      python experiments/analyze_inr.py --prefix pilot_inr --seeds 42
"""

import argparse
import json
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parents[1] / "results"
THRESHOLDS = [28, 30, 32, 33]


def collect(prefixes, image):
    """{seed: {"opt@lr": history}}。複数 prefix は後のものが同一セルを上書き
    (例: inrv2 のベースラインに inrk の κ 変種をマージ)。"""
    if isinstance(prefixes, str):
        prefixes = [prefixes]
    data = {}
    for prefix in prefixes:
        for path in sorted(RESULTS.glob(f"{prefix}_{image}_s*/metrics.json")):
            seed = int(path.parent.name.rsplit("_s", 1)[1])
            with open(path) as f:
                d = json.load(f)
            data.setdefault(seed, {}).update(d["histories"])
    return data


def reach(h, thr, basis):
    xs = h[basis]
    for x, ps in zip(xs, h["psnr"]):
        if ps >= thr:
            return float(x) if x else None
    return None


def env_reach(hists, fam, thr, basis):
    vals = [reach(h, thr, basis) for k, h in hists.items()
            if k.split("@")[0] == fam]
    vals = [v for v in vals if v is not None]
    return min(vals) if vals else None


def fmt(vals, width=11):
    ok = [v for v in vals if v is not None]
    if len(ok) == len(vals) and ok:
        return f"{np.mean(ok):>{width - 5}.2f}±{np.std(ok):<4.2f}"
    if ok:
        return f"{np.mean(ok):>{width - 7}.2f}({len(ok)}/{len(vals)})"
    return f"{'---':>{width}}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prefix", nargs="+", default=["inrv2"])
    p.add_argument("--images", nargs="+", default=["camera", "astronaut"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = p.parse_args()

    for image in args.images:
        data = collect(args.prefix, image)
        seeds = [s for s in args.seeds if s in data]
        if not seeds:
            print(f"!! {image}: {args.prefix}_* の結果がない")
            continue
        fams = sorted({k.split("@")[0] for k in data[seeds[0]]})

        print(f"\n===== {image} (prefix={args.prefix}, seeds={seeds}) =====")
        # 最終 best PSNR (家族エンベロープ)
        print(f"{'family':<14}{'best PSNR (dB)':>16}")
        for fam in fams:
            best = [max(max(h["psnr"]) for k, h in data[s].items()
                        if k.split("@")[0] == fam) for s in seeds]
            print(f"{fam:<14}{np.mean(best):>11.2f}±{np.std(best):<4.2f}")

        # 到達比 (baseline 到達 / 家族到達、>1 = 家族が速い)
        for baseline in ("adam", "lbfgs"):
            if baseline not in fams:
                continue
            bases = (("steps", "steps"), ("grad_evals", "gevals"),
                     ("time", "time"))
            print(f"--- 到達 speedup vs {baseline} エンベロープ "
                  f"(PSNR しきい値: {THRESHOLDS} dB) ---")
            for fam in fams:
                if fam == baseline:
                    continue
                for basis, label in bases:
                    row = f"{fam + ' [' + label + ']':<22}"
                    for thr in THRESHOLDS:
                        sp = []
                        for s in seeds:
                            b = env_reach(data[s], baseline, thr, basis)
                            e = env_reach(data[s], fam, thr, basis)
                            sp.append(b / e if (b and e) else None)
                        row += fmt(sp)
                    print(row)


if __name__ == "__main__":
    main()
