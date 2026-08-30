"""bench 機構の因子分析 (継続研究 TO DO 7)。

既存 4 フラグの 2^4 完全因子で KESTREL を分解し、性能だけでなく
「なぜそうなるか」を診断量で説明できるようにする。新規アルゴリズムは
実装しない。

  jump      : always_jump      True / False   (振動判定を外すか)
  bench     : cooldown_steps   20   / None    (事後フェイルベンチ)
  pre-gate  : signal_gate      0.25 / None    (モーメンタム SNR 発火ゲート)
  step ctrl : trust_kappa      50   / None    (trust region)

KESTREL (= eagle-dqn-cd) は (jump=T, bench=20, gate=None, kappa=None)。

2 段構成
--------
Stage A (fused, 性能):     16 config x 3 lr x 6 画像。
    事前登録規則 (tuning 画像の最終 PSNR 中央値) で config ごとに lr を lock。
    config 間の比較を best-to-best にするためで、固定 lr 比較はしない。
Stage B (unfused, 診断):   16 config x 6 画像を lock した lr で再走。
    JumpProbe を挿して毎ステップの機構指標を記録する。
    probe は vectorized 経路にしか刺さらないので fused=False で走らせる
    (fused と vectorized の数値一致は tests/test_equivalence.py で確認する)。

画像は camera / astronaut (bench の効きが逆転する既知の対) と、Kodak の
**tuning 側 4 枚**のみ。evaluation 側 16 枚は主比較専用なので触らない。

実行例:
    python experiments/run_bench_mechanism.py --stage A --prefix mechA
    python experiments/run_bench_mechanism.py --stage B --prefix mechB
    python experiments/run_bench_mechanism.py --stage A --dry-run
"""

import argparse
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from bench_probe import JumpProbe  # noqa: E402
from expand_grid_kodak import run_metadata  # noqa: E402
from pilot_inr import Siren, load_image, make_batch, psnr_from_mse  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eagle.optim import EAGLE  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
HERE = Path(__file__).resolve().parent

# 機構分析に使う画像。Kodak は tuning 側だけ (evaluation は主比較専用)。
IMAGES = ["camera", "astronaut", "kodim01", "kodim07", "kodim13", "kodim22"]
LRS = [3e-4, 1e-3, 3e-3]

FACTORS = {
    "jump": (True, False),        # always_jump
    "bench": (20, None),          # cooldown_steps
    "gate": (0.25, None),         # signal_gate
    "kappa": (50, None),          # trust_kappa
}
BASE_KW = dict(base="adam", adaptive_threshold=False, threshold=5e-4,
               use_lr_in_eagle_update=False, curvature_ema=0.8)


def configs():
    """16 通りの (name, kwargs)。KESTREL 既定には印をつける。"""
    out = []
    for jump, bench, gate, kappa in itertools.product(*FACTORS.values()):
        name = ("j%d_b%s_g%s_k%s"
                % (int(jump), bench or "0", gate or "0", kappa or "0"))
        kw = dict(BASE_KW, always_jump=jump, cooldown_steps=bench,
                  signal_gate=gate, trust_kappa=kappa)
        out.append((name, kw))
    return out


KESTREL_NAME = "j1_b20_g0_k0"


def cell_path(prefix, image, seed, name, lr):
    return RESULTS / f"{prefix}_s{seed}" / image / f"{name}@{lr:g}.json"


def train_one(kw, lr, coords, target, steps, seed, device, eval_every,
              warmup_steps=10, probe=False, fused=None):
    """pilot_inr.train_one と同じプロトコル。probe=True で診断を採取する。"""
    torch.manual_seed(seed)
    model = Siren(out_dim=target.shape[-1]).to(device)
    optimizer = EAGLE(model.parameters(), lr=lr, fused=fused, **kw)
    criterion = nn.MSELoss()

    jp = None
    if probe:
        jp = JumpProbe(torch.device(device), cooldown_steps=kw["cooldown_steps"])
        for g in optimizer.param_groups:
            g["_probe"] = jp

    # 計測前ウォームアップ (プロトコル v2)
    wm = Siren(out_dim=target.shape[-1]).to(device)
    wopt = EAGLE(wm.parameters(), lr=lr, fused=fused, **kw)
    for _ in range(warmup_steps):
        wopt.zero_grad(set_to_none=True)
        wl = criterion(wm(coords), target)
        wl.backward()
        wopt.clean_step = True
        wopt.step()
    torch.cuda.synchronize()
    del wm, wopt

    hist = {"steps": [], "psnr": [], "time": [], "grad_evals": []}
    log_pts = sorted({0, 1, 2, 3, 4, 6, 8, 11, 16, 23, 32, 45, 64, 91, 128}
                     | {i for i in range(0, steps + 1, eval_every)} | {steps})
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for step in range(steps + 1):
        if step in log_pts:
            torch.cuda.synchronize()
            t = time.perf_counter() - t0
            with torch.no_grad():
                mse = criterion(model(coords), target).item()
            hist["steps"].append(step)
            hist["psnr"].append(psnr_from_mse(mse))
            hist["time"].append(t)
            hist["grad_evals"].append(step)
            torch.cuda.synchronize()
            t0 -= (time.perf_counter() - t0 - t)
        if step == steps:
            break
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(coords), target)
        loss.backward()
        optimizer.clean_step = True   # フルバッチなので毎ステップがクリーン測定
        optimizer.step()
        if jp is not None:
            jp.flush(step, loss=float(loss))
    return hist, (jp.finish() if jp is not None else None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["A", "B"], required=True)
    p.add_argument("--prefix", default=None,
                   help="既定: stage A -> mechA、stage B -> mechB")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--images", nargs="*", default=IMAGES)
    p.add_argument("--lock", default=str(HERE / "mech_lr_lock.json"),
                   help="stage A が書き、stage B が読む config ごとの lr")
    p.add_argument("--lrs", nargs="*", type=float, default=None,
                   help="stage A の lr グリッド (既定は LRS)。飽和した config"
                        "の外側へ広げるときに使う")
    p.add_argument("--configs", nargs="*", default=None,
                   help="config 名で絞る (既定: 16 通り全部)")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    prefix = a.prefix or ("mechA" if a.stage == "A" else "mechB")

    cfgs = configs()
    if a.configs:
        cfgs = [(n, kw) for n, kw in cfgs if n in a.configs]
    protocol = {"steps": a.steps, "seed": a.seed, "eval_every": a.eval_every,
                "stage": a.stage}
    cfg_hash = hashlib.sha256(
        json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:12]
    meta = run_metadata()

    if a.stage == "A":
        grid = a.lrs or LRS
        todo = [(img, name, kw, lr) for img in a.images
                for name, kw in cfgs for lr in grid
                if not cell_path(prefix, img, a.seed, name, lr).exists()]
        est_min = len(todo) * 1.2
    else:
        lock = json.loads(Path(a.lock).read_text())["locked_lr"]
        todo = [(img, name, kw, lock[name]) for img in a.images
                for name, kw in cfgs
                if not cell_path(prefix, img, a.seed, name,
                                 lock[name]).exists()]
        est_min = len(todo) * 3.0   # unfused + probe のぶん重い

    print(f"# run_bench_mechanism stage={a.stage} prefix={prefix} "
          f"configs={len(cfgs)} images={len(a.images)} cfg={cfg_hash}")
    print(f"# KESTREL 既定 = {KESTREL_NAME}")
    print(f"# {meta}")
    print(f"実行 {len(todo)} セル (推定 {est_min / 60:.1f} 時間)", flush=True)
    if a.dry_run or not todo:
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fused = None if a.stage == "A" else False
    done = 0
    t_start = time.time()
    for img in a.images:
        cells = [(n, kw, lr) for (i, n, kw, lr) in todo if i == img]
        if not cells:
            continue
        coords, target = make_batch(load_image(img), device)
        for name, kw, lr in cells:
            out = cell_path(prefix, img, a.seed, name, lr)
            if out.exists():
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            t1 = time.time()
            h, diag = train_one(kw, lr, coords, target, a.steps, a.seed,
                                device, a.eval_every,
                                probe=(a.stage == "B"), fused=fused)
            flags = {k: kw[k] for k in ("always_jump", "cooldown_steps",
                                        "signal_gate", "trust_kappa")}
            rec = {"config": name, "flags": flags, "lr": lr, "image": img, "seed": a.seed, "stage": a.stage,
                   "protocol": protocol, "config_hash": cfg_hash,
                   "metadata": meta, "history": h}
            if diag is not None:
                rec["diagnostics"] = diag
            tmp = out.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(rec))
            tmp.rename(out)          # アトミック: 部分書きファイルを残さない
            done += 1
            el = time.time() - t_start
            print(f"  [done {done}/{len(todo)}] {img} {name}@{lr:g} "
                  f"final={h['psnr'][-1]:.2f}dB ({time.time() - t1:.0f}s, "
                  f"ETA {el / done * (len(todo) - done) / 3600:.1f}h)",
                  flush=True)
        del coords, target
        torch.cuda.empty_cache()
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
