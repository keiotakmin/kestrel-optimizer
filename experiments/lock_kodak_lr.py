"""tuning subset の結果から Kodak の学習率をロックし、kodak_lr_lock.json に固定する。

事前登録済みの選択規則 (kodak_split.json の _selection_metric) を機械的に適用する:
  主 = tuning 8 枚の最終 PSNR 中央値
  副 = 30 dB への censored reach (到達率優先、同率なら median reach steps)
内部点でない family はロックせず SATURATED として残す (手動確認を強制する)。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from expand_grid_kodak import FAMILIES, evaluate_grid, run_metadata  # noqa: E402
from audit_grid_saturation import CANONICAL_PREFIX, load_cells  # noqa: E402

HERE = Path(__file__).resolve().parent
SPLIT = json.loads((HERE / "kodak_split.json").read_text())
PREFIX, SEED, THRESH = "kodakx", 42, 30.0
PROTOCOL = {"steps": 2000, "seed": SEED, "eval_every": 20, "warmup_steps": 10}


def full_grid(opt):
    lrs = set()
    canon = CANONICAL_PREFIX.get(opt)
    if canon:
        cells = load_cells(canon, SEED)
        lrs |= {lr for img in cells for (f, lr) in cells[img] if f == opt}
    d = Path(__file__).resolve().parents[1] / "results" / f"{PREFIX}_s{SEED}"
    for p in d.glob(f"*/{opt}@*.json"):
        lrs.add(float(p.stem.split("@")[1]))
    return sorted(lrs)


def main():
    imgs = SPLIT["tuning"]
    out = {"_registered": "2026-08-28", "_host": run_metadata()["host"],
           "_split": "kodak_split.json:tuning", "_images": imgs,
           "_threshold_db": THRESH, "_protocol": PROTOCOL,
           "_selection_metric": SPLIT["_selection_metric"],
           "_note": ("学習率の選択は tuning 8 枚のみで行い、evaluation 16 枚は "
                     "ロック後の 1 回の比較まで使用しない。選択指標は PSNR と "
                     "steps 基準の reach のみで、host 依存の wall-clock は使わない。"),
           "families": {}}
    for opt in FAMILIES:
        grid = full_grid(opt)
        if not grid:
            continue
        per = evaluate_grid(opt, grid, imgs, SEED, PREFIX, PROTOCOL, THRESH)
        known = sorted(per)
        best = max(known, key=lambda lr: per[lr]["median_final_psnr"])
        best2 = max(known, key=lambda lr: (per[lr]["reach_rate"],
                                           -(per[lr]["median_reach_steps"] or 1e9)))
        interior = known[0] < best < known[-1]
        out["families"][opt] = {
            "grid": known, "locked_lr": best if interior else None,
            "status": "locked" if interior else "SATURATED-DO-NOT-USE",
            "secondary_best_lr": best2,
            "primary_secondary_agree": best == best2,
            "per_lr": {f"{lr:g}": per[lr] for lr in known},
        }
    p = HERE / "kodak_lr_lock.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2))

    print(f"# Kodak lr ロック  tuning={len(imgs)} 枚  threshold={THRESH}dB\n")
    h = f"{'family':<14} {'locked lr':>10} {'grid':<30} {'PSNR(med)':>10} {'reach':>9} {'副指標一致':>10}"
    print(h); print("-" * len(h))
    for opt, r in out["families"].items():
        lr = r["locked_lr"]
        c = r["per_lr"][f"{lr:g}"]
        reach = '%d/%d' % (c['n_reached'], c['n'])
        agree = "yes" if r["primary_secondary_agree"] else "NO"
        print(f"{opt:<14} {lr:>10g} "
              f"{','.join(f'{x:g}' for x in r['grid']):<30} "
              f"{c['median_final_psnr']:>10.2f} {reach:>9} {agree:>10}")

    print(f"\n=> {p}")


if __name__ == "__main__":
    main()
