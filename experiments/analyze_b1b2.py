"""B1/B2 スイープの複数シード集計。

sweep_<dataset>_s<seed>/metrics.json を集めて、進捗率 f への到達ステップの
vs Adam speedup を シード平均 ± SD で表にする。

実行: python experiments/analyze_b1b2.py [--prefix sweep]
"""

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_results import FRACS, steps_to

RESULTS = Path(__file__).resolve().parents[1] / "results"


def collect(prefix):
    """{dataset: {seed: histories}} を返す。"""
    data = {}
    for path in sorted(RESULTS.glob(f"{prefix}_*_s*/metrics.json")):
        name = path.parent.name  # 例: sweep_iris_s42
        body = name[len(prefix) + 1:]
        ds, seed = body.rsplit("_s", 1)
        d = json.load(open(path))
        data.setdefault(ds, {})[int(seed)] = d["histories"]
    return data


def milestone_speedups(hists):
    """1 シード分: {variant: {f: speedup_vs_adam}} (未到達は None)。"""
    loss0 = max(h["train_loss"][0] for h in hists.values())
    loss_star = min(min(h["train_loss"]) for h in hists.values())
    out = {}
    for f in FRACS:
        target = loss0 - f * (loss0 - loss_star)
        reach = {name: steps_to(h["steps"], h["train_loss"], target)
                 for name, h in hists.items()}
        for name in hists:
            if name == "adam":
                continue
            sp = (reach["adam"] / reach[name]
                  if reach.get("adam") and reach.get(name) else None)
            out.setdefault(name, {})[f] = sp
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="sweep")
    args = parser.parse_args()

    data = collect(args.prefix)
    if not data:
        raise SystemExit(f"{args.prefix}_* の結果が見つかりません")

    for ds, seeds in data.items():
        variants = [n for n in next(iter(seeds.values())) if n != "adam"]
        per_seed = {s: milestone_speedups(h) for s, h in seeds.items()}

        print("\n" + "=" * 95)
        print(f"### {ds} (seeds: {sorted(seeds)}, vs Adam speedup, mean±sd)")
        print("=" * 95)
        header = f"{'variant':<14}" + "".join(f"{'f=' + str(f):>16}" for f in FRACS)
        header += f"{'final_loss':>13}{'usage(early/fin)':>18}"
        print(header)

        for v in variants:
            row = f"{v:<14}"
            for f in FRACS:
                vals = [per_seed[s][v][f] for s in seeds]
                ok = [x for x in vals if x is not None]
                if len(ok) == len(vals):
                    row += f"{np.mean(ok):>9.2f}±{np.std(ok):<5.2f} "
                elif ok:
                    row += f"{np.mean(ok):>7.2f}({len(ok)}/{len(vals)})  "
                else:
                    row += f"{'---':>15} "
            fl = [seeds[s][v]["train_loss"][-1] for s in seeds]
            row += f"{np.mean(fl):>7.4f}±{np.std(fl):<5.4f}"
            ratios = [[x for x in seeds[s][v]["eagle_ratio"] if x is not None]
                      for s in seeds]
            if all(r for r in ratios):
                k = max(1, len(ratios[0]) // 4)
                early = np.mean([np.mean(r[:k]) for r in ratios])
                fin = np.mean([r[-1] for r in ratios])
                row += f"{early:>9.2f}/{fin:<7.2f}"
            print(row)

        fl_adam = [seeds[s]["adam"]["train_loss"][-1] for s in seeds]
        print(f"{'(adam final)':<14}" + " " * (16 * len(FRACS))
              + f"{np.mean(fl_adam):>7.4f}±{np.std(fl_adam):<5.4f}")


if __name__ == "__main__":
    main()
