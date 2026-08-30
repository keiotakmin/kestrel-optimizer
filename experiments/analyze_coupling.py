"""diag_coupling.py の集計 (継続研究 TO DO 8)。

各 (optimizer, rho, cond, seed) について lr は最終損失が最小のものを採る
(best-to-best)。その上で seed 中央値を rho ごとに並べ、

  1. 適用条件マップ : 結合度 ‖D^{-1}E‖ と jump 収縮率・有害ジャンプ率
  2. 検出器の性能   : precision / recall と sign-preserving な見逃しの割合
  3. 代理量の妥当性 : INR でも観測できる量 (neg_secant_rate 等) が
                      真ラベル由来の有害率とどれだけ相関するか

を出す。3 が TO DO 7 の INR 側混同行列を解釈するための橋になる。

実行: python experiments/analyze_coupling.py --in results/coupling/coupling_sweep.json
"""

import argparse
import json
from collections import defaultdict

import numpy as np

OBS = ["obs_neg_secant_rate_mean", "obs_undet_fail_rate_mean",
       "obs_det_fail_rate_mean", "obs_jump_ratio_mean_mean"]


def med(xs):
    xs = [x for x in xs if x is not None]
    return float(np.median(xs)) if xs else None


def fmt(v, spec, width):
    return format(v, spec).rjust(width) if v is not None else "-".rjust(width)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="src",
                   default="results/coupling/coupling_sweep.json")
    p.add_argument("--cond", type=float, default=None,
                   help="この条件数だけ表示 (既定: 全部)")
    a = p.parse_args()
    recs = json.load(open(a.src))["records"]

    # best-to-best: (opt, rho, cond, seed) ごとに最終損失最小の lr を採る
    best = {}
    for r in recs:
        k = (r["optimizer"], r["rho"], r["cond"], r["seed"])
        if k not in best or r["final_loss"] < best[k]["final_loss"]:
            best[k] = r
    conds = sorted({k[2] for k in best})
    if a.cond is not None:
        conds = [a.cond]

    for cond in conds:
        print(f"\n===== 条件数 cond={cond:g} "
              f"(lr は最終損失で best-to-best、seed 中央値) =====")
        rows = defaultdict(dict)
        for (opt, rho, c, seed), r in best.items():
            if c != cond:
                continue
            rows[rho].setdefault(opt, []).append(r)

        print(f"{'rho':>5}{'|D^-1E|':>9} | "
              f"{'K final':>10}{'noB final':>10}{'adam final':>11} | "
              f"{'contr med':>10}{'contr p90':>10}{'harm%':>7} | "
              f"{'prec':>6}{'recall':>7}{'signpres%':>10}")
        for rho in sorted(rows):
            g = rows[rho]
            k = g.get("kestrel", [])
            nb = g.get("kestrel-nobench", [])
            ad = g.get("adam", [])
            cp = med([r["coupling_proxy"] for r in k + nb + ad])
            line = (f"{rho:>5.2f}{fmt(cp, '.3f', 9)} | "
                    f"{fmt(med([r['final_loss'] for r in k]), '.2e', 10)}"
                    f"{fmt(med([r['final_loss'] for r in nb]), '.2e', 10)}"
                    f"{fmt(med([r['final_loss'] for r in ad]), '.2e', 11)} | "
                    f"{fmt(med([r['contraction_median'] for r in k]), '.3f', 10)}"
                    f"{fmt(med([r['contraction_p90'] for r in k]), '.2f', 10)}"
                    f"{fmt(med([r['harmful_jump_share'] for r in k]), '.1%', 7)} | "
                    f"{fmt(med([r['precision'] for r in k]), '.2f', 6)}"
                    f"{fmt(med([r['recall'] for r in k]), '.2f', 7)}"
                    f"{fmt(med([r['signpreserving_share'] for r in k]), '.1%', 10)}")
            print(line)
        # 破綻境界: 有害ジャンプ率が 15% を初めて超える結合度
        thr, bnd = 0.15, None
        for rho in sorted(rows):
            h = med([r['harmful_jump_share'] for r in rows[rho].get('kestrel', [])])
            cp = med([r['coupling_proxy'] for r in rows[rho].get('kestrel', [])])
            if h is not None and h > thr and bnd is None:
                bnd = (rho, cp, h)
        if bnd:
            print(f"  -> 破綻境界 (有害率 >{thr:.0%}): rho={bnd[0]:.2f}, "
                  f"|D^-1E|={bnd[1]:.2f} (有害率 {bnd[2]:.1%})")
        else:
            print(f"  -> この条件数では有害率が {thr:.0%} を超えない")
        print("  contr med = ジャンプ後 |θ-θ*| / ジャンプ前 (<1 で改善)、"
              "harm% = 遠ざかったジャンプの割合")
        print("  prec/recall = 事後ベンチの発火条件が真の有害ジャンプを"
              "どれだけ当てるか")
        print("  signpres%  = 有害ジャンプのうち符号が反転しなかった割合 "
              "= 条件 (4) の構造的な盲点")

    # 代理量の妥当性: INR でも観測できる量と真ラベル有害率の相関
    print("\n===== 観測代理量 vs 真ラベル有害率 (Spearman、kestrel のみ) =====")
    ks = [r for r in best.values() if r["optimizer"] == "kestrel"]
    truth = np.array([r["harmful_jump_share"] for r in ks
                      if r["harmful_jump_share"] is not None])
    for o in OBS:
        xs, ys = [], []
        for r in ks:
            if r.get(o) is not None and r["harmful_jump_share"] is not None:
                xs.append(r[o])
                ys.append(r["harmful_jump_share"])
        if len(xs) < 5:
            print(f"  {o:<32} データ不足")
            continue
        rx = np.argsort(np.argsort(xs))
        ry = np.argsort(np.argsort(ys))
        rho_s = float(np.corrcoef(rx, ry)[0, 1])
        print(f"  {o:<32} n={len(xs):>4}  Spearman={rho_s:+.3f}")
    print(f"  (真ラベル有害率の分布: 中央値 {np.median(truth):.1%}, "
          f"範囲 {truth.min():.1%}-{truth.max():.1%})")


if __name__ == "__main__":
    main()
