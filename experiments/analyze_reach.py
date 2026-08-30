"""INR 到達時間の censored 集計 (継続研究 TO DO 4)。

従来の集計 (analyze_inr.py) は「両手法が閾値に到達した画像だけ」の median を
speedup として報告していた。これは成功例だけを選んで速度を比べることになり、
到達率の低い手法ほど有利に見える。本スクリプトはそれを次のように置き換える。

  - 到達率      : 各 family の n_reached / n を常に表示する
  - censored    : 未到達を表から落とさず「> budget」として残す
                  (全画像で打ち切り時刻が同一なので、Kaplan-Meier 中央値は
                   到達率 > 0.5 のときだけ定義される)
  - paired      : 画像ごとの対応比較。両方到達した対だけでなく、
                  「片方だけ到達」も到達側の勝ちとして符号検定に含める
  - 同一画像集合: 到達率・到達 steps・最終 PSNR を同じ画像集合で表示する

学習率は experiments/kodak_lr_lock.json で事前登録された規則により lock 済み。
本スクリプトは lr を選び直さない (選択と評価を分離するため)。

実行例:
    python experiments/analyze_reach.py --split evaluation
    python experiments/analyze_reach.py --split tuning --threshold 32
    python experiments/analyze_reach.py --split evaluation --json out.json
"""

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_grid_saturation import CANONICAL_PREFIX, load_cells, median  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
HERE = Path(__file__).resolve().parent
SPLIT = json.loads((HERE / "kodak_split.json").read_text())
LOCK = json.loads((HERE / "kodak_lr_lock.json").read_text())


# ---------------------------------------------------------------- データ取得

def history(image, seed, opt, lr, extra_prefixes):
    """新規 prefix セル → 正典 prefix の順で history を探す。無ければ None。"""
    for prefix in extra_prefixes:
        p = RESULTS / f"{prefix}_s{seed}" / image / f"{opt}@{lr:g}.json"
        if p.exists():
            return json.loads(p.read_text())["history"]
    canon = CANONICAL_PREFIX.get(opt)
    if canon:
        h = load_cells(canon, seed).get(image, {}).get((opt, lr))
        if h is not None:
            return h
    return None


def reach_steps(h, thresh):
    """閾値に初めて到達した step。未到達 (censored) なら None。"""
    for s, v in zip(h["steps"], h["psnr"]):
        if v >= thresh:
            return s
    return None


# ------------------------------------------------------------------ 統計量

def km_median(reaches, n_total):
    """打ち切り時刻が全画像で共通の場合の中央値。到達率 <= 0.5 なら未定義。

    全画像が同じ budget で打ち切られるので、未到達画像の真の到達時刻は
    budget より大きい。したがって順序統計量としての中央値は、到達数が
    過半数のときだけ確定する。
    """
    if n_total == 0 or len(reaches) * 2 <= n_total:
        return None
    xs = sorted(reaches) + [math.inf] * (n_total - len(reaches))
    m = n_total // 2
    return xs[m] if n_total % 2 else 0.5 * (xs[m - 1] + xs[m])


def sign_test_p(wins, losses):
    """符号検定の両側 p 値 (引き分けは除外した exact binomial)。"""
    n = wins + losses
    if n == 0:
        return None
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired_compare(a, b, thresh):
    """family a と b の画像ごと対応比較。a, b は {image: history}。

    両方到達 -> reach steps を比較。片方だけ到達 -> 到達側の勝ち。
    両方未到達 -> undecided (符号検定から除外)。
    """
    both, a_only, b_only, neither = [], [], [], []
    wins = losses = ties = 0
    for img in sorted(set(a) & set(b)):
        ra, rb = reach_steps(a[img], thresh), reach_steps(b[img], thresh)
        if ra is not None and rb is not None:
            both.append((img, ra, rb))
            if ra < rb:
                wins += 1
            elif ra > rb:
                losses += 1
            else:
                ties += 1
        elif ra is not None:
            a_only.append(img)
            wins += 1
        elif rb is not None:
            b_only.append(img)
            losses += 1
        else:
            neither.append(img)
    ratios = [rb / ra for _, ra, rb in both if ra]
    return {
        "n_common": len(set(a) & set(b)),
        "n_both_reached": len(both),
        "n_a_only": len(a_only), "images_a_only": a_only,
        "n_b_only": len(b_only), "images_b_only": b_only,
        "n_neither": len(neither), "images_neither": neither,
        "median_speedup_both_reached": median(ratios),
        "wins": wins, "losses": losses, "ties": ties,
        "sign_test_p": sign_test_p(wins, losses),
    }


# -------------------------------------------------------------------- 集計

def summarize(fam, hists, thresh):
    imgs = sorted(hists)
    finals = [hists[i]["psnr"][-1] for i in imgs]
    reaches, unreached = [], []
    for i in imgs:
        r = reach_steps(hists[i], thresh)
        (reaches.append(r) if r is not None else unreached.append(i))
    return {
        "family": fam,
        "n": len(imgs),
        "n_reached": len(reaches),
        "reach_rate": len(reaches) / len(imgs) if imgs else None,
        "n_unreached": len(unreached),
        "images_unreached": unreached,
        "median_reach_steps_reached_only": median(reaches),
        "km_median_reach_steps": km_median(reaches, len(imgs)),
        "median_final_psnr": median(finals),
        "min_final_psnr": min(finals) if finals else None,
        "max_final_psnr": max(finals) if finals else None,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="evaluation",
                   choices=["evaluation", "tuning", "all", "div2k"],
                   help="div2k = 継続研究 A の confirmatory 16 枚")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44],
                   help="複数指定時は (画像, seed) を統計単位にする")
    p.add_argument("--threshold", type=float, default=30.0)
    p.add_argument("--prefixes", nargs="*",
                   default=["kodake", "div2ka", "kodakg", "div2kg", "kodakx"],
                   help="新規セルを探す prefix (前から順に優先)")
    p.add_argument("--families", nargs="*", default=None)
    p.add_argument("--reference", default="kestrel-cos",
                   help="paired 比較の基準 family")
    p.add_argument("--json", default=None)
    a = p.parse_args()

    if a.split == "all":
        images = sorted(SPLIT["tuning"] + SPLIT["evaluation"])
    elif a.split == "div2k":
        images = json.loads(
            (HERE / "div2k_split.json").read_text())["images"]
    else:
        images = SPLIT[a.split]

    lock = LOCK["families"]
    fams = a.families or [f for f, v in lock.items()
                          if v["status"] == "locked"]

    hists, lrs, missing = {}, {}, {}
    for fam in fams:
        if fam not in lock or lock[fam]["locked_lr"] is None:
            print(f"!! {fam}: lock されていない (skip)")
            continue
        lr = lock[fam]["locked_lr"]
        lrs[fam] = lr
        hists[fam] = {}
        miss = []
        for seed in a.seeds:
            for img in images:
                h = history(img, seed, fam, lr, a.prefixes)
                # 統計単位は (画像, seed)。複数 seed のときはこの対を 1 単位と
                # 数え、paired 比較も同じ対で突き合わせる。
                key = img if len(a.seeds) == 1 else f"{img}@s{seed}"
                if h is None:
                    miss.append(key)
                else:
                    hists[fam][key] = h
        if miss:
            missing[fam] = miss

    n_unit = len(images) * len(a.seeds)
    print(f"===== censored reach 集計 (split={a.split}, "
          f"n_images={len(images)}, seeds={a.seeds}, "
          f"統計単位={n_unit}, threshold={a.threshold:g} dB) =====")
    print(f"lr は {HERE.name}/kodak_lr_lock.json で lock 済み "
          f"(tuning 集合で事前登録規則により選択)。")
    if missing:
        print("\n[warn] 未測定セルがある family (集計は測定済み画像のみ。"
              "画像集合が揃うまで手法間比較は不当):")
        for fam, miss in missing.items():
            print(f"  {fam}@{lrs[fam]:g}: {len(miss)} 枚欠測 "
                  f"({', '.join(miss[:4])}{' ...' if len(miss) > 4 else ''})")

    rows = [summarize(f, h, a.threshold) for f, h in hists.items() if h]
    rows.sort(key=lambda r: (-r["reach_rate"],
                             r["km_median_reach_steps"] or math.inf))

    print(f"\n{'family':<14}{'lr':>8}{'reach':>9}{'rate':>7}"
          f"{'med steps':>11}{'KM med':>9}{'med PSNR':>10}{'min PSNR':>10}")
    for r in rows:
        fam = r["family"]
        rate = f"{100 * r['reach_rate']:.0f}%"
        cnt = "%d/%d" % (r["n_reached"], r["n"])
        ms = r["median_reach_steps_reached_only"]
        km = r["km_median_reach_steps"]
        ms_s = f"{ms:.0f}" if ms is not None else "---"
        km_s = f"{km:.0f}" if km is not None else ">budget"
        print(f"{fam:<14}{lrs[fam]:>8.4g}{cnt:>9}{rate:>7}"
              f"{ms_s:>11}{km_s:>9}"
              f"{r['median_final_psnr']:>10.2f}{r['min_final_psnr']:>10.2f}")
    print("\n  med steps = 到達画像だけの中央値 (楽観バイアスあり。到達率と"
          "併読すること)\n"
          "  KM med    = 未到達を > budget として残した中央値 "
          "(到達率 <= 50% では '>budget')")

    ref = a.reference
    pairs = {}
    if ref in hists:
        print(f"\n--- paired 比較 (基準 = {ref}, 画像ごとの対応、"
              f"未到達も符号検定に算入) ---")
        print(f"{'family':<14}{'both':>6}{'ref-only':>10}{'fam-only':>10}"
              f"{'neither':>9}{'spd(both)':>11}{'W/L/T':>10}{'sign p':>9}")
        for fam in hists:
            if fam == ref:
                continue
            c = paired_compare(hists[fam], hists[ref], a.threshold)
            pairs[fam] = c
            sp = c["median_speedup_both_reached"]
            sp_s = f"{sp:.2f}x" if sp else "---"
            wlt = "%d/%d/%d" % (c["wins"], c["losses"], c["ties"])
            pv = c["sign_test_p"]
            pv_s = f"{pv:.3f}" if pv is not None else "---"
            print(f"{fam:<14}{c['n_both_reached']:>6}{c['n_b_only']:>10}"
                  f"{c['n_a_only']:>10}{c['n_neither']:>9}"
                  f"{sp_s:>11}{wlt:>10}{pv_s:>9}")
        print(f"\n  spd(both) = 両方到達した画像だけの {ref}/family の "
              "reach steps 比 (>1 で family が速い)\n"
              "  W/L/T     = family から見た勝敗。片方だけ到達した画像は"
              "到達側の勝ちとして算入")

    if a.json:
        out = {
            "split": a.split, "images": images, "seeds": a.seeds,
            "threshold_db": a.threshold, "reference": ref,
            "locked_lr": lrs, "missing_cells": missing,
            "per_family": rows, "paired_vs_reference": pairs,
        }
        Path(a.json).write_text(json.dumps(out, ensure_ascii=False, indent=2))
        print(f"\n-> {a.json}")


if __name__ == "__main__":
    main()
