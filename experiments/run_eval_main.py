"""INR 主比較の 1 回きり実行 (継続研究 TO DO 5)。

experiments/kodak_lr_lock.json で lock 済みの学習率を、tuning に使っていない
evaluation 16 枚 x 3 シードで走らせる。lr の選び直しは行わない
(選択は tuning 集合、評価は evaluation 集合、という分離を守る)。

- 既存セル (新 prefix) と、seed 42 の正典 prefix セルは再利用してスキップする
- 書き込みは .json.tmp -> rename のアトミック書き。既存ディレクトリは上書きしない
- 途中で落ちても同じコマンドで再開できる (存在チェックのみで判断)

実行例:
    python experiments/run_eval_main.py --prefix kodake --seeds 42 43 44
    python experiments/run_eval_main.py --prefix kodake --dry-run
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from expand_grid_kodak import (cell_path, existing_history,  # noqa: E402
                               run_metadata)
from pilot_inr import load_image, make_batch, train_one  # noqa: E402

HERE = Path(__file__).resolve().parent
SPLIT = json.loads((HERE / "kodak_split.json").read_text())
LOCK = json.loads((HERE / "kodak_lr_lock.json").read_text())

# 1 セルあたりの概算秒数 (既存 kodakb/kodak の実測から)。進捗表示だけに使う。
SEC_PER_CELL = {"adahessian": 195, "lbfgs": 288}
SEC_DEFAULT = 70


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prefix", default="kodake",
                   help="新規セルの保存 prefix (既存を絶対に上書きしないこと)")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--families", nargs="*", default=None)
    p.add_argument("--image-set", default="kodak-evaluation",
                   choices=["kodak-evaluation", "kodak-tuning", "div2k"],
                   help="div2k = 継続研究 A の confirmatory 16 枚 "
                        "(experiments/div2k_split.json)")
    p.add_argument("--lr-grid", nargs="*", type=float, default=None,
                   help="指定すると lock を無視して families x この lr グリッド"
                        "を走らせる (新 family の lr 探索段用)")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    if a.image_set == "div2k":
        images = json.loads(
            (HERE / "div2k_split.json").read_text())["images"]
    elif a.image_set == "kodak-tuning":
        images = SPLIT["tuning"]
    else:
        images = SPLIT["evaluation"]
    lock = LOCK["families"]
    fams = a.families or [f for f, v in lock.items()
                          if v["status"] == "locked"]
    plan = []
    if a.lr_grid:
        # lr 探索段: lock を参照せず、指定 family x lr グリッドをそのまま走らせる
        if not a.families:
            raise SystemExit("--lr-grid は --families と併用すること")
        plan = [(fam, lr) for fam in a.families for lr in a.lr_grid]
        fams = a.families
    for fam in (fams if not a.lr_grid else []):
        # "-fused" アームは同一アルゴリズムの実装違いなので、lock は素の
        # family のものをそのまま使う (lr を選び直さない)。
        base = fam[:-6] if fam.endswith("-fused") else fam
        v = lock.get(base)
        if not v or v["locked_lr"] is None:
            print(f"[warn] {fam}: lock されていない (skip)")
            continue
        plan.append((fam, float(v["locked_lr"])))

    meta = run_metadata()
    print(f"# run_eval_main prefix={a.prefix} images={len(images)} "
          f"({a.image_set})"
          f" families={len(plan)} seeds={a.seeds}")
    print(f"# lr = kodak_lr_lock.json ({LOCK['_registered']} 登録) の lock 値")
    print(f"# {meta}\n", flush=True)

    # ---- 何を走らせるかを先に全部数える (再開時も同じ計算になる) ----
    todo = []
    reused = 0
    for seed in a.seeds:
        protocol = {"steps": a.steps, "seed": seed,
                    "eval_every": a.eval_every,
                    "warmup_steps": a.warmup_steps}
        for img in images:
            for fam, lr in plan:
                h, src = existing_history(img, seed, fam, lr, a.prefix,
                                          protocol)
                if h is None:
                    todo.append((seed, img, fam, lr))
                else:
                    reused += 1
    est = sum(SEC_PER_CELL.get(f, SEC_DEFAULT) for _, _, f, _ in todo)
    print(f"再利用 {reused} セル / 実行 {len(todo)} セル "
          f"(推定 {est / 3600:.1f} 時間)")
    per_seed = {}
    for seed, _, fam, _ in todo:
        per_seed[seed] = per_seed.get(seed, 0) + 1
    for seed in sorted(per_seed):
        print(f"  seed {seed}: {per_seed[seed]} セル")
    if a.dry_run or not todo:
        print("(--dry-run: 実行しない)" if a.dry_run else "実行するセルがない")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}\n", flush=True)

    # 画像ロードは重いので、(seed, image) 単位でまとめる
    done = 0
    t_start = time.time()
    for seed in a.seeds:
        protocol = {"steps": a.steps, "seed": seed,
                    "eval_every": a.eval_every,
                    "warmup_steps": a.warmup_steps}
        cfg_hash = hashlib.sha256(
            json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:12]
        for img in images:
            cells = [(f, lr) for (s, i, f, lr) in todo
                     if s == seed and i == img]
            if not cells:
                continue
            image_arr = load_image(img)
            coords, target = make_batch(image_arr, device)
            for fam, lr in cells:
                out = cell_path(a.prefix, img, seed, fam, lr)
                if out.exists():
                    continue
                out.parent.mkdir(parents=True, exist_ok=True)
                t0 = time.time()
                h = train_one(fam, lr, coords, target, a.steps, seed,
                              device, a.eval_every,
                              warmup_steps=a.warmup_steps)
                tmp = out.with_suffix(".json.tmp")
                tmp.write_text(json.dumps({
                    "optimizer": fam, "lr": lr, "image": img, "seed": seed,
                    "protocol": protocol, "config_hash": cfg_hash,
                    "split": a.image_set, "metadata": meta,
                    "history": h}, indent=2))
                tmp.rename(out)   # アトミック: 部分書きファイルを残さない
                done += 1
                el = time.time() - t_start
                eta = el / done * (len(todo) - done) / 3600
                print(f"  [done {done}/{len(todo)}] s{seed} {img} "
                      f"{fam}@{lr:g} final={h['psnr'][-1]:.2f}dB "
                      f"({time.time() - t0:.0f}s, ETA {eta:.1f}h)", flush=True)
            del coords, target
            torch.cuda.empty_cache()

    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
