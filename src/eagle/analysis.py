"""分析ユーティリティ。

- convergence_speedup: 収束速度の比較 (EAGLE がベースラインの最終 loss に
  何ステップで到達したか)
- parameter_landscape: 学習済みパラメータを 1 要素ずつ動かして損失の変化を
  調べる損失地形分析 (archive/proj/lossfunction の統合版)
"""

import time

import numpy as np
import torch

from .train import evaluate


def convergence_speedup(histories, target, baselines):
    """target がベースラインの最終 train loss に到達したステップ数と速度向上率を返す。

    Args:
        histories: {optimizer_name: history dict} (train_model の戻り値)
        target: 分析対象のオプティマイザ名 (例: "eagle")
        baselines: 比較対象のオプティマイザ名のリスト (例: ["adam", "sgd"])

    Returns:
        {baseline_name: {baseline_final_loss, baseline_final_step,
                         target_reached_step, speedup_ratio}}
    """
    result = {}
    tgt = histories[target]
    for name in baselines:
        if name not in histories or name == target:
            continue
        base = histories[name]
        final_loss = base["train_loss"][-1]
        final_step = base["steps"][-1]

        reached_step = None
        for i, loss in enumerate(tgt["train_loss"]):
            if loss <= final_loss:
                reached_step = tgt["steps"][i]
                # 直前の点との線形補間で到達ステップを精緻化
                if i > 0 and tgt["train_loss"][i - 1] > final_loss:
                    prev_loss = tgt["train_loss"][i - 1]
                    prev_step = tgt["steps"][i - 1]
                    slope = (loss - prev_loss) / (reached_step - prev_step)
                    if slope != 0:
                        reached_step = int(round(
                            prev_step + (final_loss - prev_loss) / slope))
                break

        result[name] = {
            "baseline_final_loss": final_loss,
            "baseline_final_step": final_step,
            "target_reached_step": reached_step,
            "speedup_ratio": (final_step / reached_step
                              if reached_step else None),
        }
    return result


@torch.no_grad()
def parameter_landscape(model, loader, criterion, device,
                        samples_per_layer=20, param_range=5.0,
                        n_points=100, max_batches=10, seed=0):
    """weight 行列からランダムに選んだ要素を 1 つずつ動かし、損失の変化を記録する。

    他のパラメータは学習済みの値に固定したまま、選んだ 1 要素を
    [w - param_range, w + param_range] の範囲で n_points 点動かして損失を測る。

    Returns:
        {param_name: [{index, original_value, param_values, losses}, ...]}
    """
    rng = np.random.default_rng(seed)
    trained = {name: p.data.clone()
               for name, p in model.named_parameters() if "weight" in name}

    total = sum(min(samples_per_layer, p.numel()) for p in trained.values())
    done = 0
    start = time.time()

    results = {}
    params = dict(model.named_parameters())
    for name, trained_value in trained.items():
        p = params[name]
        n_samples = min(samples_per_layer, p.numel())
        indices = np.sort(rng.choice(p.numel(), n_samples, replace=False))

        layer_results = []
        for flat_idx in indices:
            idx = np.unravel_index(flat_idx, p.shape)
            original = trained_value[idx].item()
            values = np.linspace(original - param_range,
                                 original + param_range, n_points)
            losses = []
            for v in values:
                p.data[idx] = v
                loss, _ = evaluate(model, loader, criterion, device,
                                   max_batches=max_batches)
                losses.append(loss)
            p.data[idx] = original  # 元に戻す

            layer_results.append({
                "index": tuple(int(i) for i in idx),
                "original_value": original,
                "param_values": values.tolist(),
                "losses": losses,
            })
            done += 1
            elapsed = time.time() - start
            remaining = elapsed / done * (total - done)
            print(f"\r{name}: {done}/{total} "
                  f"(残り {remaining / 60:.1f} 分)", end="", flush=True)

        results[name] = layer_results

    print()
    # 念のため全 weight を学習済みの値に復元
    for name, value in trained.items():
        params[name].data.copy_(value)
    return results
