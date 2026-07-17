"""オプティマイザの速度ベンチマーク。

forward + backward + optimizer.step() の 1 ステップあたり時間を、
Adam / SGD / EAGLE (旧ループ実装・vectorized・fused) で比較する。

実行: python experiments/bench_speed.py
"""

import importlib.util
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eagle.models import SimpleCNN  # noqa: E402
from eagle.optim import EAGLE  # noqa: E402

_ref_path = Path(__file__).resolve().parents[1] / "tests" / "reference_optim.py"
_spec = importlib.util.spec_from_file_location("reference_optim", _ref_path)
_ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ref)
ReferenceEAGLE = _ref.EAGLE


def bench(model_fn, data_fn, optimizer_fn, device, warmup=30, iters=200):
    torch.manual_seed(0)
    model = model_fn().to(device)
    optimizer = optimizer_fn(model.parameters())
    criterion = nn.CrossEntropyLoss()
    X, y = data_fn(device)

    def one_step():
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()

    for _ in range(warmup):
        one_step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        one_step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000  # ms/step


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name() if device.type == 'cuda' else ''})")

    models = {
        "MLP 784-100-10 (80K params)": (
            lambda: nn.Sequential(nn.Flatten(), nn.Linear(784, 100), nn.ReLU(),
                                  nn.Linear(100, 10)),
            lambda d: (torch.randn(64, 784, device=d),
                       torch.randint(0, 10, (64,), device=d)),
        ),
        "MLP 784-1024-1024-10 (1.9M params)": (
            lambda: nn.Sequential(nn.Flatten(), nn.Linear(784, 1024), nn.ReLU(),
                                  nn.Linear(1024, 1024), nn.ReLU(),
                                  nn.Linear(1024, 10)),
            lambda d: (torch.randn(64, 784, device=d),
                       torch.randint(0, 10, (64,), device=d)),
        ),
        "SimpleCNN (430K params)": (
            SimpleCNN,
            lambda d: (torch.randn(64, 1, 28, 28, device=d),
                       torch.randint(0, 10, (64,), device=d)),
        ),
    }

    eagle_kw = dict(lr=1e-3, base="adam", adaptive_threshold=True)
    optimizers = {
        "Adam (torch)": lambda ps: optim.Adam(ps, lr=1e-3),
        "SGD (torch)": lambda ps: optim.SGD(ps, lr=1e-3, momentum=0.9),
        "EAGLE 旧実装": lambda ps: ReferenceEAGLE(
            ps, lr=1e-3, base="adam", adaptive_threshold=True),
        "EAGLE vectorized": lambda ps: EAGLE(ps, fused=False, **eagle_kw),
        "EAGLE fused": lambda ps: EAGLE(ps, fused=True, **eagle_kw),
    }

    for model_name, (model_fn, data_fn) in models.items():
        print(f"\n### {model_name}")
        adam_ms = None
        for opt_name, opt_fn in optimizers.items():
            ms = bench(model_fn, data_fn, opt_fn, device)
            if opt_name.startswith("Adam"):
                adam_ms = ms
            rel = f"(Adam 比 x{ms / adam_ms:.2f})" if adam_ms else ""
            print(f"{opt_name:<18} {ms:8.3f} ms/step  {rel}")


if __name__ == "__main__":
    main()
