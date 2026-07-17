"""プロトコル v2 の集計: lr エンベロープ同士のマイルストーン比較。

run_protocol.sh が生成した proto_<dataset>_lr<lr>_s<seed>/metrics.json を
集め、(dataset, seed) ごとに全 (optimizer, lr) セルを共通の loss0/loss* で
マイルストーン化して、

  1. セルごとの到達速度 (vs adam エンベロープ = adam 家族の最速 lr)
  2. 家族エンベロープ同士の比較 (ステップ基準と wall-clock 基準)

を 3 シード平均 ± SD で表にする。指標は train_eval_loss (不偏な最適化進捗、
プライマリ) と test_loss (汎化) の両方。running train_loss は
ペア変種で楽観バイアスがあるためマイルストーンには使わない。

実行: python experiments/analyze_protocol.py --dataset iris covtype
"""

import argparse
import json
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parents[1] / "results"
FRACS = [0.5, 0.8, 0.9, 0.95]
SEEDS = (42, 43, 44)


def x_to(xs, losses, target):
    """losses が target に到達する最初の x (線形補間)。未到達は None。"""
    for i, loss in enumerate(losses):
        if loss <= target:
            if i == 0:
                return float(xs[0])
            prev_l, prev_x = losses[i - 1], xs[i - 1]
            if prev_l > target and loss < prev_l:
                frac = (prev_l - target) / (prev_l - loss)
                return prev_x + frac * (xs[i] - prev_x)
            return float(xs[i])
    return None


def collect(dataset, prefixes=("proto",)):
    """{seed: {"opt@lr": history}}。<prefix>_<dataset>_lr* を全部読む。

    複数 prefix を渡すと後の prefix の同一セルが前を上書きする
    (例: proto の eagle 系を protoc = clean_every_step 版で差し替える)。
    """
    data = {}
    for prefix in prefixes:
        for path in sorted(RESULTS.glob(f"{prefix}_{dataset}_lr*_s*/metrics.json")):
            body = path.parent.name[len(f"{prefix}_{dataset}_lr"):]
            lr, seed = body.rsplit("_s", 1)
            with open(path) as f:
                d = json.load(f)
            for opt, h in d["histories"].items():
                data.setdefault(int(seed), {})[f"{opt}@{lr}"] = h
    return data


def fmt(vals, width=9, prec=2):
    ok = [v for v in vals if v is not None]
    if len(ok) == len(vals) and ok:
        return f"{np.mean(ok):>{width - 5}.{prec}f}±{np.std(ok):<4.{prec}f}"
    if ok:
        return f"{np.mean(ok):>{width - 7}.{prec}f}({len(ok)}/{len(vals)})"
    return f"{'---':>{width}}"


def analyze(dataset, key, seeds, prefixes=("proto",), baseline="adam"):
    data = collect(dataset, prefixes)
    seeds = [s for s in seeds if s in data]
    if not seeds:
        print(f"!! {dataset}: proto_* の結果がない")
        return
    cells = sorted(data[seeds[0]].keys())
    families = sorted({c.split("@")[0] for c in cells})
    if baseline not in families:
        print(f"!! {dataset}: {baseline} 家族がないためエンベロープ比較不可")
        return

    # (seed, cell) ごとの到達ステップ / 到達時間 / 到達勾配評価数 / 最良 loss
    reach_s, reach_t, reach_g, best = {}, {}, {}, {}
    for s in seeds:
        hists = data[s]
        loss0 = max(h[key][0] for h in hists.values())
        lstar = min(min(h[key]) for h in hists.values())
        for c, h in hists.items():
            # grad_evals 未記録の旧ランは lbfgs 以外なら steps で代用
            # (lbfgs だけは line search で steps ≠ grad_evals のため代用不可)
            gevals = h.get("grad_evals")
            if gevals is None and not c.startswith("lbfgs"):
                gevals = h["steps"]
            for f in FRACS:
                tgt = loss0 - f * (loss0 - lstar)
                reach_s[s, c, f] = x_to(h["steps"], h[key], tgt)
                reach_t[s, c, f] = x_to(h["time"], h[key], tgt)
                reach_g[s, c, f] = (x_to(gevals, h[key], tgt)
                                    if gevals is not None else None)
            best[s, c] = min(h[key])

    def env(s, family, f, reach):
        vals = [reach.get((s, c, f)) for c in cells
                if c.split("@")[0] == family]
        vals = [v for v in vals if v is not None]
        return min(vals) if vals else None

    print(f"\n===== {dataset} / {key} 基準 (seeds: {seeds}, "
          f"baseline: {baseline}) =====")
    header = f"{'cell':<18}" + "".join(f"{'f=' + str(f):>12}" for f in FRACS)
    print(header + f"{'best_' + key:>18}")
    for c in cells:
        row = f"{c:<18}"
        for f in FRACS:
            sp = []
            for s in seeds:
                a, r = env(s, baseline, f, reach_s), reach_s.get((s, c, f))
                sp.append(a / r if (a and r) else None)
            row += f"{fmt(sp, 12)}"
        b = [best[s, c] for s in seeds if (s, c) in best]
        row += f"    {np.mean(b):>9.4f}±{np.std(b):.4f}"
        print(row)

    print(f"--- 家族エンベロープ vs {baseline} エンベロープ ---")
    for fam in families:
        if fam == baseline:
            continue
        for label, reach in (("steps", reach_s), ("time ", reach_t),
                             ("gevals", reach_g)):
            row = f"{fam + ' [' + label.strip() + ']':<18}"
            for f in FRACS:
                sp = []
                for s in seeds:
                    a, e = env(s, baseline, f, reach), env(s, fam, f, reach)
                    sp.append(a / e if (a and e) else None)
                row += f"{fmt(sp, 12)}"
            print(row)
        fam_best = [min(best[s, c] for c in cells
                        if c.split("@")[0] == fam and (s, c) in best)
                    for s in seeds]
        base_best = [min(best[s, c] for c in cells
                         if c.split("@")[0] == baseline and (s, c) in best)
                     for s in seeds]
        print(f"{'':>18}best({fam})={np.mean(fam_best):.4f}±"
              f"{np.std(fam_best):.4f}  best({baseline})="
              f"{np.mean(base_best):.4f}±{np.std(base_best):.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", nargs="+", default=["iris", "covtype"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--keys", nargs="+",
                        default=["train_eval_loss", "test_loss"])
    parser.add_argument("--prefix", nargs="+", default=["proto"],
                        help="結果ディレクトリの prefix (後のものが上書き)")
    parser.add_argument("--baseline", default="adam",
                        help="エンベロープ比の基準にする家族 (例: lbfgs)")
    args = parser.parse_args()
    for key in args.keys:
        for ds in args.dataset:
            analyze(ds, key, args.seeds, prefixes=tuple(args.prefix),
                    baseline=args.baseline)


if __name__ == "__main__":
    main()
