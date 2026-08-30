"""Kodak-24 の学習率グリッド飽和監査 (継続研究 TO DO 2 の前段)。

生の results/*/metrics.json だけを読み、family ごとに

  - 主選択指標  : tuning 画像集合上の最終 PSNR の中央値
  - 副選択指標  : 所定 PSNR しきい値への censored reach (到達率つき)

を計算し、選択された学習率が探索区間の**内部**にあるかを判定する。
端に張り付いた family は、どちら向きにグリッドを広げるべきかを出力する。

使用例:
    python experiments/audit_grid_saturation.py
    python experiments/audit_grid_saturation.py --images all --json out.json
"""

import argparse
import json
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"

# family -> その family の正典プレフィックス。同じ family が複数 prefix に
# 存在する場合に、無自覚な混合を防ぐため明示的に固定する。
CANONICAL_PREFIX = {
    "adam": "kodakb",
    "adamw": "kodakb",
    "adam-cos": "kodakb",
    "adabelief": "kodakb",
    "bb-stab": "kodakb",
    "adahessian": "kodakb",
    "eagle-dqn-cd": "kodakb",   # = KESTREL 素
    "kestrel-cos": "kodakc",    # = KESTREL+cos
    "rprop": "kodakr",
    "eagle3": "kodak",
    "eagle-dqn": "kodak",
    "lbfgs": "kodak",
}


@lru_cache(maxsize=None)
def load_cells(prefix, seed=42):
    """{image: {(family, lr): history}} を返す (プロセス内でキャッシュする。
    metrics.json は数十 MB あり、セル参照のたびに再パースすると CPU 律速に
    なるため)。"""
    out = defaultdict(dict)
    for p in sorted(RESULTS.glob(f"{prefix}_kodim*_s{seed}/metrics.json")):
        img = re.search(r"(kodim\d+)", p.parent.name).group(1)
        d = json.loads(p.read_text())
        for key, h in d["histories"].items():
            fam, lr = key.rsplit("@", 1)
            out[img][(fam, float(lr))] = h
    return out


def reach(h, thresh):
    """PSNR が thresh に初めて到達した step。未到達なら None (censored)。"""
    for s, v in zip(h["steps"], h["psnr"]):
        if v >= thresh:
            return s
    return None


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return None
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def audit(families, images, thresh, seed):
    rows = []
    for fam in families:
        prefix = CANONICAL_PREFIX[fam]
        cells = load_cells(prefix, seed)
        lrs = sorted({lr for img in cells for (f, lr) in cells[img] if f == fam})
        if not lrs:
            continue
        per_lr = {}
        for lr in lrs:
            finals, reaches, n_img, n_reach = [], [], 0, 0
            for img in images:
                h = cells.get(img, {}).get((fam, lr))
                if h is None:
                    continue
                n_img += 1
                finals.append(h["psnr"][-1])
                r = reach(h, thresh)
                if r is not None:
                    reaches.append(r)
                    n_reach += 1
            if n_img == 0:
                continue
            per_lr[lr] = {
                "n_images": n_img,
                "median_final_psnr": median(finals),
                "reach_rate": n_reach / n_img,
                "median_reach_steps": median(reaches),
                "n_reached": n_reach,
            }
        if not per_lr:
            continue
        grid = sorted(per_lr)
        best = max(grid, key=lambda lr: per_lr[lr]["median_final_psnr"])
        # 副指標: 到達率を優先し、同率なら reach step が小さいほう
        best2 = max(grid, key=lambda lr: (per_lr[lr]["reach_rate"],
                                          -(per_lr[lr]["median_reach_steps"]
                                            or 10 ** 9)))
        at_lo, at_hi = best == grid[0], best == grid[-1]
        # 端との差 (どれだけ「まだ伸びている」か)
        if at_hi and len(grid) > 1:
            margin = (per_lr[grid[-1]]["median_final_psnr"]
                      - per_lr[grid[-2]]["median_final_psnr"])
        elif at_lo and len(grid) > 1:
            margin = (per_lr[grid[0]]["median_final_psnr"]
                      - per_lr[grid[1]]["median_final_psnr"])
        else:
            margin = None
        rows.append({
            "family": fam, "prefix": prefix, "grid": grid,
            "best_lr_final_psnr": best, "best_lr_reach": best2,
            "status": "SATURATED-HI" if at_hi else
                      "SATURATED-LO" if at_lo else "interior",
            "extend_toward": (grid[-1] * 3 if at_hi else
                              grid[0] / 3 if at_lo else None),
            "edge_margin_db": margin,
            "reach_best_at_edge": best2 in (grid[0], grid[-1]),
            "per_lr": per_lr,
        })
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", default="all",
                   help="all | tuning | eval | kodim01,kodim05,...")
    p.add_argument("--threshold", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--families", nargs="+", default=sorted(CANONICAL_PREFIX))
    p.add_argument("--json", default=None)
    a = p.parse_args()

    split = json.loads((Path(__file__).with_name("kodak_split.json")).read_text())
    if a.images == "all":
        images = split["tuning"] + split["evaluation"]
    elif a.images in split:
        images = split[a.images]
    else:
        images = a.images.split(",")
    images = sorted(images)

    rows = audit(a.families, images, a.threshold, a.seed)
    print(f"# Kodak lr グリッド監査  images={len(images)} ({a.images})  "
          f"threshold={a.threshold} dB  seed={a.seed}\n")
    hdr = (f"{'family':<14} {'grid':<26} {'best(PSNR)':>10} {'best(reach)':>11} "
           f"{'status':<13} {'margin dB':>9} {'extend to':>10}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        g = ",".join(f"{x:g}" for x in r["grid"])
        m = r["edge_margin_db"]
        mtxt = f"{m:+.2f}" if m is not None else "-"
        e = r["extend_toward"]
        etxt = f"{e:g}" if e else "-"
        print(f"{r['family']:<14} {g:<26} {r['best_lr_final_psnr']:>10g} "
              f"{r['best_lr_reach']:>11g} {r['status']:<13} "
              f"{mtxt:>9} {etxt:>10}")

    print("\n## family ごとの内訳 (median final PSNR / 到達率 / median reach)")
    for r in rows:
        print(f"\n[{r['family']}]  prefix={r['prefix']}  {r['status']}")
        for lr in r["grid"]:
            c = r["per_lr"][lr]
            mr = c["median_reach_steps"]
            print(f"   lr={lr:<8g} n={c['n_images']:<3} "
                  f"finalPSNR={c['median_final_psnr']:.3f}  "
                  f"reach={c['n_reached']}/{c['n_images']} "
                  f"({c['reach_rate']*100:.0f}%)  "
                  f"median_reach={mr if mr is not None else 'n/a'}")

    n_sat = sum(1 for r in rows if r["status"] != "interior")
    print(f"\n=> {n_sat}/{len(rows)} family がグリッド端で選択されている")
    if a.json:
        Path(a.json).write_text(json.dumps(
            {"images": images, "threshold": a.threshold, "rows": rows},
            indent=2))
        print(f"   JSON: {a.json}")


if __name__ == "__main__":
    main()
