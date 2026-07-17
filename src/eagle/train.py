"""学習ループとメトリクス記録。

記録する指標 (すべてステップベース):
- train/test の loss と accuracy
- train_eval_loader 指定時: 固定 train サブセットの loss/acc (不偏な進捗指標)
- EAGLE 更新使用率 (EAGLE 系オプティマイザのみ)
- 累積学習時間 (評価にかかった時間は含めない。CUDA 同期込みで計測)
"""

import copy
import math
import time

import torch
import torch.nn as nn
from tqdm import tqdm

from .optim import eagle_update_ratio


def _is_regression(criterion):
    return isinstance(criterion, nn.MSELoss)


@torch.no_grad()
def evaluate(model, loader, criterion, device, max_batches=None):
    """平均 loss と精度指標を返す。

    精度指標は分類なら accuracy (%)、回帰 (MSELoss) なら RMSE。
    """
    regression = _is_regression(criterion)
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    n_batches = 0
    for i, (X, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        X, y = X.to(device), y.to(device)
        out = model(X)
        total_loss += criterion(out, y).item()
        if not regression:
            correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
        n_batches += 1
    avg_loss = total_loss / max(n_batches, 1)
    if regression:
        return avg_loss, math.sqrt(max(avg_loss, 0.0))
    return avg_loss, 100.0 * correct / max(total, 1)


def warmup_optimizer(model, optimizer_factory, batch, criterion, device,
                     steps=20, create_graph=False):
    """計測前ウォームアップ (プロトコル v2)。

    モデルの deepcopy に対して同一構成のオプティマイザを steps 回走らせ、
    CUDA カーネルの JIT コンパイル・cuDNN/cuBLAS の初期化・アロケータの
    ウォームアップを済ませる。これをしないと「プロセス内で最初に走った
    オプティマイザだけ遅い」実行順の交絡が wall-clock に乗る。

    RNG は消費しない (固定バッチ・dropout なしのモデルが前提) ので、
    学習の軌道には影響しない。
    """
    dummy = copy.deepcopy(model)
    optimizer = optimizer_factory(dummy.parameters())
    X, y = batch[0].to(device), batch[1].to(device)
    dummy.train()

    def closure():
        optimizer.zero_grad()
        loss = criterion(dummy(X), y)
        # create_graph: Hutchinson HVP 系 (adahessian) は勾配のグラフが必要
        loss.backward(create_graph=create_graph)
        return loss

    for _ in range(steps):
        # closure 渡しはどのオプティマイザでも有効で、L-BFGS には必須
        optimizer.step(closure)
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _log_record_steps(limit, factor=1.4):
    """対数間隔の記録ステップ集合 (1, 2, 3, 4, 6, 8, 11, 16, ...)。

    序盤のマイルストーン到達が最初の記録窓に埋もれる解像度問題への対策。
    """
    steps = set()
    s = 1.0
    while s <= limit:
        steps.add(int(round(s)))
        s *= factor
    return steps


def train_model(model, optimizer, train_loader, test_loader, criterion,
                epochs, device, eval_every=None, desc="", repeat_batch=False,
                pair_every=None, train_eval_loader=None, log_early=False,
                clean_every_step=False, create_graph=False,
                grad_evals_per_step=1):
    """モデルを学習し、履歴 (dict of lists) を返す。

    Args:
        eval_every: 何ステップごとに test 評価・記録するか。
            None なら 1 エポックごと。
        repeat_batch: 各ミニバッチを 2 ステップ連続で使う (ペアバッチ割線用。
            EAGLE(paired_secant=True) と併用する)。1 エポックのステップ数は
            2 倍になる。
        pair_every: k バッチに 1 回だけ同一バッチを 2 連使用し、2 発目の前に
            optimizer.clean_step=True を立てる (間欠ペア + 曲率 EMA 用。
            EAGLE(curvature_ema=β) と併用する)。
        train_eval_loader: 固定 train サブセットの評価用ローダ
            (make_train_eval_loader)。指定すると各記録点で eval モードの
            train_eval_loss/acc を記録し、学習前 (step 0) の記録点も追加する。
            running train loss のペア再訪バイアスを受けない進捗指標になる。
        log_early: eval_every の記録に加え、対数間隔 (1, 2, 3, 4, 6, ...) の
            早期記録点を足す (序盤マイルストーンの解像度向上)。
        clean_every_step: 毎ステップ optimizer.clean_step=True を立てる
            (バッチの繰り返しはしない)。フルバッチ学習専用 — 全ステップの
            勾配が同一関数の勾配なので、連続 2 ステップの Δg/Δp が常に
            クリーンな曲率測定になる。ペア測定 (pair_every) と違い
            ステップ数を増やさないため、他オプティマイザと計算量が揃う。
            pair_every / repeat_batch とは併用不可。
        create_graph: backward(create_graph=True) で勾配のグラフを保持する
            (adahessian の Hutchinson HVP 用)。
        grad_evals_per_step: 1 ステップに計上する勾配評価数。adahessian は
            HVP が backward 相当 1 回分なので 2 を渡す (予算会計の公正化)。

    回帰 (criterion が MSELoss) では acc 系の指標に RMSE を記録する。
    L-BFGS は closure 経由で学習する (1 記録ステップ = optimizer.step 1 回。
    line search が行う複数回の関数評価はステップ数に現れないため、
    L-BFGS との計算量公平な比較は time 基準で行うこと)。
    """
    if clean_every_step and (pair_every is not None or repeat_batch):
        raise ValueError("clean_every_step は pair_every / repeat_batch と併用不可")
    if eval_every is None:
        eval_every = len(train_loader) * (2 if repeat_batch else 1)
    regression = _is_regression(criterion)
    is_lbfgs = isinstance(optimizer, torch.optim.LBFGS)
    max_steps = epochs * len(train_loader) * 2  # ペア変種の上限
    log_steps = _log_record_steps(max_steps) if log_early else frozenset()

    history = {
        "steps": [], "train_loss": [], "train_acc": [],
        "test_loss": [], "test_acc": [], "eagle_ratio": [], "time": [],
        # 勾配評価回数 (予算会計)。通常は steps と一致するが、L-BFGS は
        # line search が 1 ステップに複数回評価するためこちらが真の計算量
        "grad_evals": [],
    }
    grad_evals = 0
    if train_eval_loader is not None:
        history["train_eval_loss"] = []
        history["train_eval_acc"] = []

        # step 0 (共通の初期値) の記録点。マイルストーン分析の loss0 が
        # 「最初の記録窓の running 平均」でなく初期値そのものになる
        te_loss, te_acc = evaluate(model, train_eval_loader, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        history["steps"].append(0)
        history["train_loss"].append(te_loss)   # step 0 は eval 値で代用
        history["train_acc"].append(te_acc)
        history["train_eval_loss"].append(te_loss)
        history["train_eval_acc"].append(te_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["eagle_ratio"].append(eagle_update_ratio(optimizer))
        history["time"].append(0.0)
        history["grad_evals"].append(0)

    use_cuda = torch.cuda.is_available() and str(device).startswith("cuda")
    train_time = 0.0
    global_step = 0
    steps_since_record = 0
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    for epoch in range(epochs):
        model.train()
        # disable=None: 非 TTY (ログ出力・nohup 実行) では進捗バーを出さない
        progress = tqdm(train_loader, desc=f"{desc} epoch {epoch + 1}/{epochs}",
                        leave=False, disable=None)
        for batch_idx, (X, y) in enumerate(progress):
          X, y = X.to(device), y.to(device)
          is_pair = repeat_batch or (pair_every is not None
                                     and batch_idx % pair_every == 0)
          for rep in range(2 if is_pair else 1):
            if pair_every is not None and rep == 1:
                optimizer.clean_step = True  # 同一バッチ 2 発目 = クリーン測定
            if clean_every_step:
                # フルバッチ: 全勾配が同一関数の勾配 = 毎ステップがクリーン測定
                optimizer.clean_step = True
            start = time.time()
            if is_lbfgs:
                out_box = []

                def closure():
                    nonlocal grad_evals
                    grad_evals += 1
                    optimizer.zero_grad()
                    out = model(X)
                    loss = criterion(out, y)
                    loss.backward()
                    del out_box[:]
                    out_box.append(out.detach())
                    return loss

                loss = optimizer.step(closure)
                out = out_box[0]
            else:
                optimizer.zero_grad()
                out = model(X)
                loss = criterion(out, y)
                loss.backward(create_graph=create_graph)
                optimizer.step()
                grad_evals += grad_evals_per_step
            if use_cuda:
                # GPU キューの完了までを計測窓に入れる (これがないと直後の
                # loss.item() の同期に GPU 時間が流出し、計測窓が過小になる)
                torch.cuda.synchronize()
            train_time += time.time() - start

            global_step += 1
            steps_since_record += 1
            running_loss += loss.item()
            if not regression:
                running_correct += (out.argmax(1) == y).sum().item()
            running_total += y.size(0)

            if global_step % eval_every == 0 or global_step in log_steps:
                n = steps_since_record
                test_loss, test_acc = evaluate(model, test_loader, criterion, device)
                history["steps"].append(global_step)
                history["train_loss"].append(running_loss / n)
                if regression:
                    history["train_acc"].append(
                        math.sqrt(max(running_loss / n, 0.0)))
                else:
                    history["train_acc"].append(
                        100.0 * running_correct / running_total)
                if train_eval_loader is not None:
                    te_loss, te_acc = evaluate(model, train_eval_loader,
                                               criterion, device)
                    history["train_eval_loss"].append(te_loss)
                    history["train_eval_acc"].append(te_acc)
                history["test_loss"].append(test_loss)
                history["test_acc"].append(test_acc)
                history["eagle_ratio"].append(eagle_update_ratio(optimizer))
                history["time"].append(train_time)
                history["grad_evals"].append(grad_evals)
                progress.set_postfix(loss=f"{running_loss / n:.4f}",
                                     test_metric=f"{test_acc:.3f}")
                running_loss = 0.0
                running_correct = 0
                running_total = 0
                steps_since_record = 0
                model.train()

    return history
