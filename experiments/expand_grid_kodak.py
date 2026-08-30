"""Kodak の学習率グリッド飽和を解消する自動再探索ドライバ (TO DO 2)。

設計原則:
  * **上書き禁止**  1 セル = 1 ファイル (results/<prefix>/<img>/<opt>@<lr>.json)。
    既存ファイルがあればスキップするので、途中で落ちても再開可能。
  * **tuning subset のみ**  experiments/kodak_split.json の tuning 8 枚だけで
    グリッドを追う。evaluation 16 枚はロック後の 1 回の比較まで使わない。
  * **選択規則は事前固定**  主 = tuning 上の最終 PSNR 中央値、
    副 = しきい値への censored reach。family ごとに後出しで変えない。
  * **既存セルを再利用**  同一プロトコル (steps/seed/warmup/eval_every) の
    既存 results を読み込み、新しい lr だけを実際に走らせる。
  * **run metadata** commit/torch/cuda/GPU/dtype/TF32/config hash を各セルに記録。

使用例:
    python experiments/expand_grid_kodak.py --dry-run
    python experiments/expand_grid_kodak.py --max-rounds 3
"""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_grid_saturation import (CANONICAL_PREFIX, load_cells,  # noqa: E402
                                   median, reach)
from pilot_inr import load_image, make_batch, train_one  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SPLIT = json.loads((Path(__file__).with_name("kodak_split.json")).read_text())

# lr グリッドは 3 倍刻み。端に張り付いた向きへ 1 段ずつ広げる。
STEP_FACTOR = 3.0
HARD_BOUNDS = (1e-6, 10.0)

# 監査対象 family。素の KESTREL / KESTREL+cos / Adam+cos は既に内部点だが、
# lock 規則を全 family 共通にするため同じ経路で扱う。
FAMILIES = ["adam", "adamw", "adam-cos", "adabelief", "bb-stab", "adahessian",
            "rprop", "lbfgs", "eagle3", "eagle-dqn-cd", "kestrel-cos"]


def run_metadata():
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(
            ["git", "-C", str(ROOT), "status", "--porcelain"],
            text=True, stderr=subprocess.DEVNULL).strip())
    except Exception:
        commit, dirty = None, None
    try:
        sys.path.insert(0, str(ROOT / "src"))
        from eagle._cuda import get_extension
        fused = get_extension() is not None
    except Exception:
        fused = None
    return {
        "commit": commit, "git_dirty": dirty,
        # fused CUDA カーネルが実際にロードされたか。time 基準の比較可能性は
        # これで決まるので、各 run が自分で記録する。
        "fused_available": fused,
        "host": platform.node(), "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "dtype": "float32",
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }


def cell_path(prefix, image, seed, opt, lr):
    return RESULTS / f"{prefix}_s{seed}" / image / f"{opt}@{lr:g}.json"


def existing_history(image, seed, opt, lr, new_prefix, protocol):
    """新規セル or 既存の正典 prefix から同一プロトコルの history を探す。"""
    p = cell_path(new_prefix, image, seed, opt, lr)
    if p.exists():
        d = json.loads(p.read_text())
        return d["history"], "new"
    canon = CANONICAL_PREFIX.get(opt)
    if canon:
        cells = load_cells(canon, seed)
        h = cells.get(image, {}).get((opt, lr))
        if h is not None:
            return h, f"reuse:{canon}"
    return None, None


def evaluate_grid(opt, grid, images, seed, new_prefix, protocol, thresh):
    per_lr = {}
    for lr in grid:
        finals, reaches, n, nr = [], [], 0, 0
        for img in images:
            h, _ = existing_history(img, seed, opt, lr, new_prefix, protocol)
            if h is None:
                continue
            n += 1
            finals.append(h["psnr"][-1])
            r = reach(h, thresh)
            if r is not None:
                reaches.append(r)
                nr += 1
        if n:
            per_lr[lr] = {"n": n, "median_final_psnr": median(finals),
                          "reach_rate": nr / n, "n_reached": nr,
                          "median_reach_steps": median(reaches)}
    return per_lr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prefix", default="kodakx",
                   help="新規セルの保存 prefix (既存を絶対に上書きしないこと)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--threshold", type=float, default=30.0)
    p.add_argument("--families", nargs="+", default=FAMILIES)
    p.add_argument("--max-rounds", type=int, default=3)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    images = SPLIT["tuning"]
    protocol = {"steps": a.steps, "seed": a.seed, "eval_every": a.eval_every,
                "warmup_steps": a.warmup_steps}
    cfg_hash = hashlib.sha256(
        json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:12]
    meta = run_metadata()
    print(f"# expand_grid_kodak  prefix={a.prefix} images={len(images)} "
          f"(tuning) threshold={a.threshold}dB cfg={cfg_hash}")
    print(f"# {meta}\n", flush=True)

    # 各 family の初期グリッド = 既存の正典 prefix にあるもの
    grids = {}
    for opt in a.families:
        canon = CANONICAL_PREFIX.get(opt)
        cells = load_cells(canon, a.seed) if canon else {}
        lrs = sorted({lr for img in cells for (f, lr) in cells[img] if f == opt})
        if lrs:
            grids[opt] = lrs
        else:
            print(f"[warn] {opt}: 既存グリッドが見つからない (skip)")

    cache = {}
    for rnd in range(1, a.max_rounds + 1):
        todo = []          # (opt, lr) 実行待ち
        status = {}
        for opt, grid in grids.items():
            per_lr = evaluate_grid(opt, grid, images, a.seed, a.prefix,
                                   protocol, a.threshold)
            cache[opt] = per_lr
            known = sorted(per_lr)
            if not known:
                continue
            best = max(known, key=lambda lr: per_lr[lr]["median_final_psnr"])
            if best == known[-1]:
                nxt = known[-1] * STEP_FACTOR
                st = "SATURATED-HI"
            elif best == known[0]:
                nxt = known[0] / STEP_FACTOR
                st = "SATURATED-LO"
            else:
                nxt, st = None, "interior"
            if nxt is not None and not (HARD_BOUNDS[0] <= nxt <= HARD_BOUNDS[1]):
                st, nxt = "interior(hard-bound)", None
            status[opt] = (st, best, nxt)
            if nxt is not None:
                todo.append((opt, nxt))
                grids[opt] = sorted(set(grid) | {nxt})

        print(f"\n=== round {rnd} ===")
        for opt, (st, best, nxt) in sorted(status.items()):
            g = ",".join(f"{x:g}" for x in sorted(cache[opt]))
            print(f"  {opt:<14} best={best:<8g} {st:<20} grid=[{g}]"
                  + (f" -> add lr={nxt:g}" if nxt else ""))
        if not todo:
            print("\n全 family が内部点。グリッドをロックできる。")
            break
        n_cells = len(todo) * len(images)
        print(f"\n  実行セル数 = {len(todo)} lr x {len(images)} 画像 "
              f"= {n_cells} (推定 {n_cells * 1.6:.0f} 分)")
        if a.dry_run:
            print("  (--dry-run: 実行しない)")
            break

        device = "cuda" if torch.cuda.is_available() else "cpu"
        for img in images:
            image_arr = load_image(img)
            coords, target = make_batch(image_arr, device)
            for opt, lr in todo:
                out = cell_path(a.prefix, img, a.seed, opt, lr)
                if out.exists():
                    print(f"  [skip] {img} {opt}@{lr:g}", flush=True)
                    continue
                out.parent.mkdir(parents=True, exist_ok=True)
                t0 = time.time()
                h = train_one(opt, lr, coords, target, a.steps, a.seed,
                              device, a.eval_every,
                              warmup_steps=a.warmup_steps)
                tmp = out.with_suffix(".json.tmp")
                tmp.write_text(json.dumps({
                    "optimizer": opt, "lr": lr, "image": img,
                    "protocol": protocol, "config_hash": cfg_hash,
                    "round": rnd, "metadata": meta, "history": h}, indent=2))
                tmp.rename(out)   # アトミック: 部分書きファイルを残さない
                print(f"  [done] {img} {opt}@{lr:g} "
                      f"final={h['psnr'][-1]:.2f}dB ({time.time()-t0:.0f}s)",
                      flush=True)
            del coords, target
            torch.cuda.empty_cache()
    else:
        print(f"\n[warn] max-rounds={a.max_rounds} に達した。端が残る family は "
              f"手動で確認すること。")

    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
