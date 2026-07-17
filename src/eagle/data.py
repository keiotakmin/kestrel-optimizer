"""データセットのローダ (分類: Iris/Wine/Cancer/MNIST/表形式大規模、回帰: UCI 系)。

回帰データセットはターゲットも訓練統計で標準化する (loss はデータセット間で
比較可能な σ_y^2 単位の MSE になる。RMSE も σ_y 単位)。
"""

from pathlib import Path

import torch
import numpy as np
from sklearn.datasets import (fetch_covtype, fetch_openml, load_breast_cancer,
                              load_iris, load_wine)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, default_collate
from torchvision import datasets, transforms

# bachelor/data/ をデータ置き場にする (MNIST はここに自動ダウンロード)
DATA_DIR = Path(__file__).resolve().parents[2] / "data"

DATASET_INFO = {
    "iris": dict(input_size=4, num_classes=3, default_batch_size=16),
    "wine": dict(input_size=13, num_classes=3, default_batch_size=16),
    "cancer": dict(input_size=30, num_classes=2, default_batch_size=16),
    "mnist": dict(input_size=784, num_classes=10, default_batch_size=64),
    # 大規模表形式 (581k サンプル、初回に ~11MB ダウンロード)
    "covtype": dict(input_size=54, num_classes=7, default_batch_size=256),
    # 以下は OpenML から取得 (input_size はロード時に確定する)
    "adult": dict(input_size=None, num_classes=2, default_batch_size=256),
    "higgs": dict(input_size=None, num_classes=2, default_batch_size=256),
    # 回帰 (MSE)。num_classes=1 は出力次元。滑らかな損失地形の検証用
    "california": dict(input_size=8, num_classes=1, default_batch_size=256,
                       task="regression"),
    "concrete": dict(input_size=8, num_classes=1, default_batch_size=64,
                     task="regression"),
    "energy": dict(input_size=8, num_classes=1, default_batch_size=64,
                   task="regression"),
}


def dataset_task(name):
    """"classification" か "regression" を返す。"""
    return DATASET_INFO[name].get("task", "classification")


def _load_covtype():
    X, y = fetch_covtype(data_home=str(DATA_DIR), return_X_y=True)
    from types import SimpleNamespace
    return SimpleNamespace(data=X, target=y.astype(int) - 1)  # ラベルを 0 始まりに


def _load_adult():
    """Adult / Census Income (48.8k)。カテゴリ特徴は one-hot 化する。"""
    import pandas as pd
    from types import SimpleNamespace
    frame = fetch_openml("adult", version=2, as_frame=True,
                         data_home=str(DATA_DIR)).frame
    y = (frame["class"] == ">50K").astype(int).to_numpy()
    X = frame.drop(columns=["class"])
    cat_cols = [c for c in X.columns
                if not pd.api.types.is_numeric_dtype(X[c])]
    X[cat_cols] = X[cat_cols].astype(str)  # NaN も "nan" カテゴリとして扱う
    X = pd.get_dummies(X, columns=cat_cols)
    return SimpleNamespace(data=X.to_numpy(dtype=np.float32), target=y)


def _load_higgs():
    """HIGGS の OpenML 版サブサンプル (98k × 28、物理特徴量)。"""
    from types import SimpleNamespace
    X, y = fetch_openml(data_id=23512, as_frame=False, return_X_y=True,
                        data_home=str(DATA_DIR))
    X = np.nan_to_num(np.asarray(X, dtype=np.float32))
    y = np.asarray(y).astype(float).astype(int)
    return SimpleNamespace(data=X, target=y)


def _load_california():
    from sklearn.datasets import fetch_california_housing
    from types import SimpleNamespace
    X, y = fetch_california_housing(data_home=str(DATA_DIR), return_X_y=True)
    return SimpleNamespace(data=X.astype(np.float32),
                           target=y.astype(np.float32))


def _load_concrete():
    """UCI Concrete Compressive Strength (1030 × 8)。"""
    from types import SimpleNamespace
    X, y = fetch_openml(data_id=44959, as_frame=False, return_X_y=True,
                        parser="auto", data_home=str(DATA_DIR))
    return SimpleNamespace(data=np.asarray(X, dtype=np.float32),
                           target=np.asarray(y, dtype=np.float32))


def _load_energy():
    """UCI Energy Efficiency (768 × 8)。ターゲットは y1 (暖房負荷)、y2 は捨てる。"""
    from types import SimpleNamespace
    frame = fetch_openml(name="energy-efficiency", version=1, as_frame=True,
                         parser="auto", data_home=str(DATA_DIR)).frame
    y = frame["y1"].to_numpy(dtype=np.float32)
    X = frame.drop(columns=["y1", "y2"]).to_numpy(dtype=np.float32)
    return SimpleNamespace(data=X, target=y)


_SKLEARN_LOADERS = {
    "iris": load_iris,
    "wine": load_wine,
    "cancer": load_breast_cancer,
    "covtype": _load_covtype,
    "adult": _load_adult,
    "higgs": _load_higgs,
    "california": _load_california,
    "concrete": _load_concrete,
    "energy": _load_energy,
}


class _SingleBatchLoader:
    """事前に collate 済みの 1 バッチを返す軽量ローダ。

    DataLoader は巨大バッチでも毎エポック全サンプルを Python ループで
    collate し直す。フルバッチ学習 (1 エポック = 1 ステップ) ではこの
    再 collate が学習本体より桁違いに重くなるため、1 度だけ collate して
    使い回す。勾配はサンプル順に依存しない (平均) ので数値は等価。
    """

    def __init__(self, dataset, indices=None):
        self.dataset = dataset
        idx = range(len(dataset)) if indices is None else indices
        self._batch = default_collate([dataset[i] for i in idx])
        self.batch_size = len(idx)

    def __len__(self):
        return 1

    def __iter__(self):
        yield self._batch


def get_dataloaders(name, batch_size=None, test_size=0.2, seed=42,
                    full_batch=False):
    """データセット名から (train_loader, test_loader) を返す。

    表形式データ (iris/wine/cancer) は標準化してから分割する。
    標準化は訓練データのみで fit する (archive の旧実装は全データで fit していた)。
    回帰データセットはターゲットも訓練統計で標準化する (σ_y 単位)。
    full_batch=True で train/test とも全データ 1 バッチにする
    (勾配ノイズ σ→0 の理論レジーム検証用)。
    """
    if name not in DATASET_INFO:
        raise ValueError(f"Unknown dataset: {name} (choose from {list(DATASET_INFO)})")

    batch_size = batch_size or DATASET_INFO[name]["default_batch_size"]
    regression = dataset_task(name) == "regression"

    if name in _SKLEARN_LOADERS:
        bunch = _SKLEARN_LOADERS[name]()
        X_train, X_test, y_train, y_test = train_test_split(
            bunch.data, bunch.target, test_size=test_size,
            random_state=seed,
            stratify=None if regression else bunch.target)

        scaler = StandardScaler().fit(X_train)
        X_train = torch.tensor(scaler.transform(X_train), dtype=torch.float32)
        X_test = torch.tensor(scaler.transform(X_test), dtype=torch.float32)
        if regression:
            y_mean, y_std = float(np.mean(y_train)), float(np.std(y_train))
            y_train = torch.tensor((y_train - y_mean) / y_std,
                                   dtype=torch.float32).unsqueeze(1)
            y_test = torch.tensor((y_test - y_mean) / y_std,
                                  dtype=torch.float32).unsqueeze(1)
        else:
            y_train = torch.tensor(y_train, dtype=torch.long)
            y_test = torch.tensor(y_test, dtype=torch.long)

        # one-hot 化などで次元がロード時に確定するデータセットに対応
        DATASET_INFO[name]["input_size"] = X_train.shape[1]

        train_set = TensorDataset(X_train, y_train)
        test_set = TensorDataset(X_test, y_test)
    else:  # mnist
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train_set = datasets.MNIST(root=str(DATA_DIR), train=True,
                                   download=True, transform=transform)
        test_set = datasets.MNIST(root=str(DATA_DIR), train=False,
                                  download=True, transform=transform)

    if full_batch:
        train_loader = _SingleBatchLoader(train_set)
        test_loader = _SingleBatchLoader(test_set)
    else:
        train_loader = DataLoader(train_set, batch_size=batch_size,
                                  shuffle=True)
        test_loader = DataLoader(test_set, batch_size=batch_size,
                                 shuffle=False)
    return train_loader, test_loader


def make_train_eval_loader(train_loader, max_samples=10240):
    """訓練データの固定サブセットを評価用に返す (shuffle なし)。

    学習中の running train loss は、ペア変種 (同一バッチを 2 連使用) では
    「直前に学習したバッチの再訪」の低い loss が混入して楽観バイアスになる。
    最適化の進捗はこのローダを eval モードで測るのが公平 (プロトコル v2)。

    collate は 1 度だけ行い、以降の評価では使い回す (_SingleBatchLoader)。
    """
    dataset = train_loader.dataset
    n = min(len(dataset), max_samples)
    return _SingleBatchLoader(dataset, range(n))
