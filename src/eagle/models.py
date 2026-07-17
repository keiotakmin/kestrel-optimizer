"""実験用の小規模モデル。"""

import torch.nn as nn

from .data import DATASET_INFO


ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}


class MLP(nn.Module):
    """全結合ネットワーク。hidden_sizes で層構成を指定する。

    旧実装との対応: IrisNet = MLP(4, (25,), 3), WineNet = MLP(13, (15,), 3) など。
    activation: relu (既定) / tanh / gelu。tanh/gelu は損失地形から
    ReLU 由来のキンクを除く (割線法の理論レジーム検証用)。
    """

    def __init__(self, input_size, hidden_sizes, num_classes,
                 activation="relu"):
        super().__init__()
        layers = []
        prev = input_size
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), ACTIVATIONS[activation]()]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() > 2:  # 画像入力 (MNIST) はフラット化
            x = x.flatten(1)
        return self.net(x)


class SimpleCNN(nn.Module):
    """MNIST 用の小さな CNN。"""

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(dataset, arch="mlp", hidden_sizes=(25,), activation="relu"):
    """データセット名とアーキテクチャ指定からモデルを構築する。

    回帰データセット (num_classes=1) は出力 1 次元の MLP になる。
    """
    info = DATASET_INFO[dataset]
    if arch == "cnn":
        if dataset != "mnist":
            raise ValueError("arch='cnn' は mnist のみ対応")
        return SimpleCNN(num_classes=info["num_classes"])
    return MLP(info["input_size"], tuple(hidden_sizes), info["num_classes"],
               activation=activation)
