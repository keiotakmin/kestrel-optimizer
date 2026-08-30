"""機構分析 16 config の学習率を lock する (TO DO 7 stage A -> stage B)。

Kodak 主比較と同じ事前登録規則を使う:
  主指標 = 全画像上の最終 PSNR の中央値
  副指標 = 30 dB への censored reach (到達率優先、同率なら median reach steps)
config ごとに規則を後出しで変えない。端に張り付いた config は
SATURATED として記録し、lock せずに手動確認へ回す。

実行: python experiments/lock_mech_lr.py --prefix mechA
"""

import argparse
import json
from pathlib import Path

from audit_grid_saturation import median, reach

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[0] / "results"


def load(prefix, seed, images):
    """{config: {lr: {image: history}}}"""
    out = {}
    for img in images:
        d = RESULTS / f"{prefix}_s{seed}" / img
        for p in sorted(d.glob("*.json")):
            name, lr = p.stem.rsplit("@", 1)
            rec = json.loads(p.read_text())
            out.setdefault(name, {}).setdefault(float(lr), {})[img] = \
                rec["history"]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prefix", default="mechA")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threshold", type=float, default=30.0)
    p.add_argument("--out", default=str(HERE / "mech_lr_lock.json"))
    a = p.parse_args()

    from run_bench_mechanism import IMAGES, KESTREL_NAME
    data = load(a.prefix, a.seed, IMAGES)
    locked, detail = {}, {}
    for name in sorted(data):
        per_lr = {}
        for lr, hs in sorted(data[name].items()):
            finals = [h["psnr"][-1] for h in hs.values()]
            rs = [reach(h, a.threshold) for h in hs.values()]
            got = [r for r in rs if r is not None]
            per_lr[lr] = {"n": len(hs), "median_final_psnr": median(finals),
                          "n_reached": len(got),
                          "reach_rate": len(got) / len(hs) if hs else None,
                          "median_reach_steps": median(got)}
        if not per_lr:
            continue
        grid = sorted(per_lr)
        best = max(grid, key=lambda lr: per_lr[lr]["median_final_psnr"])
        status = ("SATURATED-HI" if best == grid[-1] else
                  "SATURATED-LO" if best == grid[0] else "locked")
        sec = max(grid, key=lambda lr: (per_lr[lr]["reach_rate"],
                                        -(per_lr[lr]["median_reach_steps"]
                                          or 1e18)))
        locked[name] = best
        detail[name] = {"grid": grid, "locked_lr": best, "status": status,
                        "secondary_best_lr": sec,
                        "primary_secondary_agree": best == sec,
                        "per_lr": per_lr}

    out = {"_registered": "TO DO 7 stage A", "_prefix": a.prefix,
           "_images": IMAGES, "_threshold_db": a.threshold,
           "_kestrel_config": KESTREL_NAME,
           "_selection_metric": {
               "primary": "全画像上の最終 PSNR の中央値",
               "secondary": "30 dB への censored reach (到達率優先)"},
           "locked_lr": locked, "families": detail}
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=2))

    print(f"{'config':<20}{'lr':>8}{'status':>14}{'medPSNR':>10}"
          f"{'reach':>8}{'2nd':>8}")
    for name in sorted(detail):
        d = detail[name]
        c = d["per_lr"][d["locked_lr"]]
        r = "%d/%d" % (c["n_reached"], c["n"])
        mark = " <- KESTREL" if name == KESTREL_NAME else ""
        print(f"{name:<20}{d['locked_lr']:>8.4g}{d['status']:>14}"
              f"{c['median_final_psnr']:>10.2f}{r:>8}"
              f"{d['secondary_best_lr']:>8.4g}{mark}")
    n_sat = sum(1 for d in detail.values() if d["status"] != "locked")
    print(f"\n-> {a.out}  (端に張り付き {n_sat} config)")
    if n_sat:
        print("SATURATED の config はグリッドを広げるか、機構分析の対象外と"
              "して明記すること。")


if __name__ == "__main__":
    main()
