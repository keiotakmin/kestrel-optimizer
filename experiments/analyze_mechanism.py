"""bench 機構の因子分析の集計 (継続研究 TO DO 7)。

Stage B (mechB) の診断ログを読み、

  1. config ごとの性能 (最終 PSNR・30 dB 到達) と機構指標
  2. 4 因子 (jump / bench / pre-gate / trust) の主効果
  3. camera と astronaut の対比 (bench の効きが逆転する既知の対)

を出す。成功基準は「bench あり が一位」ではなく、bench の利得が失敗指標と
整合して条件付きで説明できること。

実行: python experiments/analyze_mechanism.py --prefix mechB
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[0] / "results"
KESTREL = "j1_b20_g0_k0"

DIAG = ["jump_rate", "bench_rate", "fail_rate", "det_fail_rate",
        "undet_fail_rate", "false_alarm_rate", "good_jump_rate",
        "neg_secant_rate", "jump_ratio_mean", "recover_steps_mean"]
FACTORS = {"jump": "always_jump", "bench": "cooldown_steps",
           "gate": "signal_gate", "kappa": "trust_kappa"}


def load(prefix, seed):
    """{config: {image: record}}"""
    out = defaultdict(dict)
    root = RESULTS / f"{prefix}_s{seed}"
    for img_dir in sorted(root.glob("*")):
        for p in sorted(img_dir.glob("*.json")):
            d = json.loads(p.read_text())
            out[d["config"]][d["image"]] = d
    return out


def reach(h, thr):
    for s, v in zip(h["steps"], h["psnr"]):
        if v >= thr:
            return s
    return None


def dmean(rec, key):
    v = [x for x in rec.get("diagnostics", {}).get(key, []) if x is not None]
    return float(np.mean(v)) if v else None


def agg(recs, thr):
    finals = [r["history"]["psnr"][-1] for r in recs.values()]
    rs = [reach(r["history"], thr) for r in recs.values()]
    got = [x for x in rs if x is not None]
    row = {"n": len(recs), "median_final_psnr": float(np.median(finals)),
           "min_final_psnr": float(np.min(finals)),
           "n_reached": len(got),
           "median_reach_steps": float(np.median(got)) if got else None}
    for k in DIAG:
        vals = [dmean(r, k) for r in recs.values()]
        vals = [v for v in vals if v is not None]
        row[k] = float(np.mean(vals)) if vals else None
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prefix", default="mechB")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threshold", type=float, default=30.0)
    p.add_argument("--json", default=None)
    a = p.parse_args()

    data = load(a.prefix, a.seed)
    if not data:
        raise SystemExit(f"{a.prefix}_s{a.seed} にデータがない")
    flags = {c: next(iter(v.values()))["flags"] for c, v in data.items()}
    rows = {c: agg(v, a.threshold) for c, v in data.items()}

    print(f"===== config 別 (prefix={a.prefix}, seed={a.seed}, "
          f"threshold={a.threshold:g} dB) =====")
    print(f"{'config':<20}{'PSNR':>8}{'min':>8}{'reach':>7}{'steps':>7}"
          f"{'jump':>7}{'bench':>7}{'fail':>7}{'det':>7}{'undet':>7}"
          f"{'good':>7}{'neg':>7}")
    order = sorted(rows, key=lambda c: -rows[c]["median_final_psnr"])
    for c in order:
        r = rows[c]
        f = lambda k: (f"{r[k]:.2f}" if r.get(k) is not None else "-")
        cnt = "%d/%d" % (r["n_reached"], r["n"])
        st = (f"{r['median_reach_steps']:.0f}"
              if r["median_reach_steps"] is not None else "-")
        mark = " <- KESTREL" if c == KESTREL else ""
        print(f"{c:<20}{r['median_final_psnr']:>8.2f}"
              f"{r['min_final_psnr']:>8.2f}{cnt:>7}{st:>7}"
              f"{f('jump_rate'):>7}{f('bench_rate'):>7}{f('fail_rate'):>7}"
              f"{f('det_fail_rate'):>7}{f('undet_fail_rate'):>7}"
              f"{f('good_jump_rate'):>7}{f('neg_secant_rate'):>7}{mark}")
    print("  det/undet/good = 直前にジャンプした座標の帰結の内訳、"
          "neg = 負の割線が棄却された割合")

    print("\n===== 4 因子の主効果 (他因子で平均。差 = ON - OFF) =====")
    print(f"{'factor':<10}{'ON':>26}{'OFF':>26}{'ΔPSNR':>9}{'Δreach率':>10}")
    eff = {}
    for name, key in FACTORS.items():
        on = [c for c in rows if flags[c][key] not in (None, False)]
        off = [c for c in rows if flags[c][key] in (None, False)]
        if not on or not off:
            continue
        po = float(np.mean([rows[c]["median_final_psnr"] for c in on]))
        pf = float(np.mean([rows[c]["median_final_psnr"] for c in off]))
        ro = float(np.mean([rows[c]["n_reached"] / rows[c]["n"] for c in on]))
        rf = float(np.mean([rows[c]["n_reached"] / rows[c]["n"] for c in off]))
        eff[name] = {"psnr_on": po, "psnr_off": pf, "d_psnr": po - pf,
                     "reach_on": ro, "reach_off": rf, "d_reach": ro - rf}
        print(f"{name:<10}{f'{po:.2f} dB / {100*ro:.0f}%':>26}"
              f"{f'{pf:.2f} dB / {100*rf:.0f}%':>26}"
              f"{po - pf:>+9.2f}{100 * (ro - rf):>+9.0f}%")

    print("\n===== camera / astronaut の対比 (bench の効きの逆転) =====")
    print(f"{'config':<20}{'camera':>9}{'astronaut':>11}{'差':>8}"
          f"{'cam undet':>11}{'ast undet':>11}")
    for c in order:
        recs = data[c]
        if "camera" not in recs or "astronaut" not in recs:
            continue
        cp = recs["camera"]["history"]["psnr"][-1]
        ap = recs["astronaut"]["history"]["psnr"][-1]
        cu = dmean(recs["camera"], "undet_fail_rate")
        au = dmean(recs["astronaut"], "undet_fail_rate")
        g = lambda v: (f"{v:.2f}" if v is not None else "-")
        mark = " <- KESTREL" if c == KESTREL else ""
        print(f"{c:<20}{cp:>9.2f}{ap:>11.2f}{cp - ap:>+8.2f}"
              f"{g(cu):>11}{g(au):>11}{mark}")

    if a.json:
        Path(a.json).write_text(json.dumps(
            {"prefix": a.prefix, "seed": a.seed, "threshold_db": a.threshold,
             "per_config": rows, "flags": flags, "main_effects": eff},
            ensure_ascii=False, indent=2))
        print(f"\n-> {a.json}")


if __name__ == "__main__":
    main()
