"""algorithmic 層と systems 層を分けて報告する (継続研究 TO DO 3)。

TO DO 3 の規則:

  algorithmic : optimizer steps と gradient evaluations。同じ停止条件、
                未到達込み。ここでのみ「更新回数/勾配評価を削減した」と言える。
  systems     : wall-clock。**fused 対 fused または reference 対 reference**
                が成立し、かつ contention のないホストで測ったときだけ報告する。
                成立しない場合は headline から外す。

本スクリプトは各セルの metadata を読み、実装クラス (fused / foreach / python)
とホストが比較可能かを機械的に判定する。判定が通らない限り time 列は出さない。

実行:
    python experiments/report_layers.py --split div2k
    python experiments/report_layers.py --split evaluation --threshold 32
"""

import argparse
import json
import math
from pathlib import Path

from analyze_reach import (LOCK, SPLIT, history, km_median, median,
                           reach_steps)

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[0] / "results"

# 実装クラス。torch 既定の Adam/AdamW/AdamCosine は foreach、Rprop/L-BFGS 等は
# python ループ、KESTREL/EAGLE は fused カーネルが載る環境でのみ fused。
IMPL_DEFAULT = {
    "adam": "foreach", "adamw": "foreach", "adam-cos": "foreach",
    "adabelief": "python", "adahessian": "python", "bb-stab": "python",
    "rprop": "python", "lbfgs": "python",
    "eagle": "eagle-family", "eagle3": "eagle-family",
    "eagle-dqn-cd": "eagle-family", "kestrel-cos": "eagle-family",
}

def resolve_impl(fam, hosts, fused_flags, fused_hosts):
    """EAGLE 系が fused 経路で走ったかを判定する。

    新しいセルは metadata["fused_available"] を自分で持つ。持たない古いセル
    については --fused-hosts で明示的に宣言してもらう (既定は「不明」)。
    """
    base = IMPL_DEFAULT.get(fam, "?")
    if base != "eagle-family":
        return base
    if fused_flags == {True}:
        return "fused"
    if fused_flags == {False}:
        return "python"
    if hosts and fused_hosts and hosts <= set(fused_hosts):
        return "fused"
    return "unknown"


def cell_meta(image, seed, fam, lr, prefixes):
    """セルの metadata (host 等) を返す。新 prefix セルにしか入っていない。"""
    for prefix in prefixes:
        p = RESULTS / f"{prefix}_s{seed}" / image / f"{fam}@{lr:g}.json"
        if p.exists():
            d = json.loads(p.read_text())
            return d.get("metadata", {})
    return {}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="div2k",
                   choices=["div2k", "evaluation", "tuning"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--threshold", type=float, default=30.0)
    p.add_argument("--prefixes", nargs="*",
                   default=["kodake", "div2ka", "kodakg", "div2kg", "kodakx"])
    p.add_argument("--fused-hosts", nargs="*", default=[],
                   help="fused カーネルが載るホスト名。metadata に "
                        "fused_available が無い古いセル用の宣言")
    p.add_argument("--json", default=None)
    a = p.parse_args()

    if a.split == "div2k":
        images = json.loads(
            (HERE / "div2k_split.json").read_text())["images"]
    else:
        images = SPLIT["evaluation" if a.split == "evaluation" else "tuning"]

    lock = LOCK["families"]
    fams = [f for f, v in lock.items() if v["status"] == "locked"]

    rows, hosts, impls = [], set(), {}
    fam_hosts, no_meta, fam_fused = {}, {}, {}
    for fam in fams:
        lr = lock[fam]["locked_lr"]
        steps, gevals, finals, n, nr = [], [], [], 0, 0
        for seed in a.seeds:
            for img in images:
                h = history(img, seed, fam, lr, a.prefixes)
                if h is None:
                    continue
                n += 1
                finals.append(h["psnr"][-1])
                r = reach_steps(h, a.threshold)
                if r is not None:
                    nr += 1
                    steps.append(r)
                    # 到達時点の累積 gradient evaluations
                    i = h["steps"].index(r)
                    gevals.append(h["grad_evals"][i])
                m = cell_meta(img, seed, fam, lr, a.prefixes)
                if m.get("host"):
                    hosts.add(m["host"])
                    fam_hosts.setdefault(fam, set()).add(m["host"])
                    if "fused_available" in m:
                        fam_fused.setdefault(fam, set()).add(
                            m["fused_available"])
                else:
                    no_meta[fam] = no_meta.get(fam, 0) + 1
        if not n:
            continue
        impls[fam] = resolve_impl(fam, fam_hosts.get(fam, set()),
                                  fam_fused.get(fam, set()), a.fused_hosts)
        rows.append({
            "family": fam, "lr": lr, "n": n, "n_reached": nr,
            "reach_rate": nr / n,
            "median_reach_steps": median(steps),
            "km_median_reach_steps": km_median(steps, n),
            "median_reach_grad_evals": median(gevals),
            "median_final_psnr": median(finals),
            "impl": impls[fam],
        })
    rows.sort(key=lambda r: (-r["reach_rate"],
                             r["km_median_reach_steps"] or math.inf))

    print(f"===== algorithmic 層 (split={a.split}, seeds={a.seeds}, "
          f"threshold={a.threshold:g} dB) =====")
    print("報告できる主張: 「同じ停止条件の下で、更新回数または勾配評価回数を"
          "削減した」")
    print(f"\n{'family':<14}{'lr':>8}{'impl':>16}{'reach':>9}{'rate':>7}"
          f"{'med steps':>11}{'KM steps':>10}{'med gevals':>12}"
          f"{'med PSNR':>10}")
    for r in rows:
        cnt = "%d/%d" % (r["n_reached"], r["n"])
        km = r["km_median_reach_steps"]
        ms = r["median_reach_steps"]
        ge = r["median_reach_grad_evals"]
        ms_s = f"{ms:.0f}" if ms is not None else "---"
        km_s = f"{km:.0f}" if km is not None else ">budget"
        ge_s = f"{ge:.0f}" if ge is not None else "---"
        rate_s = f"{100 * r['reach_rate']:.0f}%"
        print(f"{r['family']:<14}{r['lr']:>8.4g}{r['impl']:>16}{cnt:>9}"
              f"{rate_s:>7}{ms_s:>11}{km_s:>10}{ge_s:>12}"
              f"{r['median_final_psnr']:>10.2f}")

    print(f"\n===== systems 層 =====")
    kinds = set(impls.values())
    print(f"実装クラス: {sorted(kinds)}")
    print(f"計測ホスト: {sorted(hosts) if hosts else '(metadata なし)'}")
    if no_meta:
        tot = sum(no_meta.values())
        print(f"環境 metadata のないセル: {tot} 個 "
              f"({', '.join(f'{k}:{v}' for k, v in sorted(no_meta.items()))})")
        print("  -> 旧 prefix からの再利用セル。実装クラス/ホストが確定しない"
              "ので time 基準には使えない。")
    ok = (len(hosts) == 1) and (len(kinds) == 1) and not no_meta
    if ok:
        print("同等実装かつ単一ホスト -> wall-clock を報告してよい")
    else:
        reasons = []
        if len(kinds) > 1:
            reasons.append(f"実装クラスが混在 ({sorted(kinds)})")
        if len(hosts) != 1:
            reasons.append(f"計測ホストが単一でない ({sorted(hosts)})")
        print("**wall-clock は報告しない**: " + "、".join(reasons))
        print("  -> TO DO 3 の規則により、時間軸の主張は fused 対 fused を")
        print("     contention のないホストで測り直してからにする。")

    if a.json:
        Path(a.json).write_text(json.dumps(
            {"split": a.split, "seeds": a.seeds, "threshold_db": a.threshold,
             "algorithmic": rows, "impl_classes": impls,
             "hosts": sorted(hosts), "systems_reportable": ok},
            ensure_ascii=False, indent=2))
        print(f"\n-> {a.json}")


if __name__ == "__main__":
    main()
