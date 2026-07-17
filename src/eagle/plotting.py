"""結果の可視化。サーバー上でのヘッドレス実行を前提に Agg バックエンドを使う。"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def plot_comparison(histories, save_path, title="", metric_label="Accuracy (%)"):
    """オプティマイザ比較の 6 面プロット (loss/acc/EAGLE 使用率/時間)。

    回帰では metric_label="RMSE" を渡す (acc スロットに RMSE が入っている)。
    """
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))

    panels = [
        ("train_loss", "Training Loss", "Loss"),
        ("train_acc", f"Training {metric_label}", metric_label),
        ("test_loss", "Test Loss", "Loss"),
        ("test_acc", f"Test {metric_label}", metric_label),
    ]
    for ax, (key, panel_title, ylabel) in zip(axes.flat[:4], panels):
        for name, h in histories.items():
            ax.plot(h["steps"], h[key], label=name)
        ax.set_title(panel_title)
        ax.set_xlabel("Steps")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, linestyle=":", alpha=0.5)

    ax = axes.flat[4]
    has_ratio = False
    for name, h in histories.items():
        if any(r is not None for r in h["eagle_ratio"]):
            ax.plot(h["steps"], h["eagle_ratio"], label=name)
            has_ratio = True
    ax.set_title("EAGLE Update Usage Ratio")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Ratio")
    if has_ratio:
        ax.legend()
    ax.grid(True, linestyle=":", alpha=0.5)

    ax = axes.flat[5]
    for name, h in histories.items():
        ax.plot(h["steps"], h["time"], label=name)
    ax.set_title("Cumulative Training Time")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Time (s)")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.5)

    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_landscape(landscape, baseline_loss, train_loss_history, save_path,
                   title="", n_cols=10):
    """損失地形分析のグリッドプロット。先頭行に学習曲線、以降に各要素の損失曲線。"""
    n_params = sum(len(v) for v in landscape.values())
    n_rows = (n_params + n_cols - 1) // n_cols + 1

    fig = plt.figure(figsize=(2 * n_cols, 3 * n_rows))
    gs = plt.GridSpec(n_rows, n_cols)

    ax_loss = fig.add_subplot(gs[0, :])
    ax_loss.plot(train_loss_history, "b-", linewidth=1)
    ax_loss.set_xlabel("Recorded step")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Training Loss")
    ax_loss.grid(True, linestyle=":", alpha=0.5)

    i = 0
    for param_name, entries in landscape.items():
        for entry in entries:
            ax = fig.add_subplot(gs[i // n_cols + 1, i % n_cols])
            ax.plot(entry["param_values"], entry["losses"], "b-", linewidth=1)
            ax.axvline(entry["original_value"], color="r", linestyle="--",
                       alpha=0.6)
            ax.axhline(baseline_loss, color="g", linestyle="--", alpha=0.6)
            ax.set_title(f"{param_name}\n{entry['index']}", fontsize=7)
            ax.tick_params(labelsize=6)
            ax.grid(True, linestyle=":", alpha=0.4)
            i += 1

    if title:
        fig.suptitle(title, fontsize=14, y=1.0)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
