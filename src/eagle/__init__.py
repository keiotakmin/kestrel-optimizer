"""EAGLE オプティマイザ研究 (卒論) 用パッケージ。"""

from .analysis import convergence_speedup, parameter_landscape
from .data import DATASET_INFO, get_dataloaders, make_train_eval_loader
from .models import MLP, SimpleCNN, build_model
from .optim import EAGLE, eagle_update_ratio
from .plotting import plot_comparison, plot_landscape
from .train import evaluate, train_model, warmup_optimizer
from .utils import get_device, set_seed

__all__ = [
    "EAGLE", "eagle_update_ratio",
    "DATASET_INFO", "get_dataloaders", "make_train_eval_loader",
    "MLP", "SimpleCNN", "build_model",
    "train_model", "evaluate", "warmup_optimizer",
    "convergence_speedup", "parameter_landscape",
    "plot_comparison", "plot_landscape",
    "set_seed", "get_device",
]
