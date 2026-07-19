# KESTREL

Code accompanying the paper *"KESTREL: Coordinate-Wise Secant Steps with
Post-Hoc Failure Benching for Deterministic Neural Optimization"* (under
review). This repository is private during double-blind review and will be
made public upon publication.

KESTREL is a curvature-aware optimizer for **deterministic full-batch**
objectives (INR/SIREN fitting, full-batch regression). It maintains a
per-coordinate secant estimate of the diagonal curvature from consecutive
gradients (no extra gradient cost), takes learning-rate-free Newton-like
steps on every coordinate with a valid estimate, and secures them with a
single safeguard: a **post-hoc bench** that detects failed jumps from the
gradient response and demotes only the failing coordinates to Adam for
K=20 steps.

## Install

```bash
pip install -e .                       # core (torch, sklearn, ...)
pip install adabelief-pytorch pytorch_optimizer   # paper baselines (optional)
```

Python >= 3.10, PyTorch >= 2.x. A CUDA device enables the fused kernel
(JIT-compiled on first use; automatic fallback to a vectorized
implementation on CPU).

## Quick start

```python
from eagle.optim import EAGLE

# KESTREL = curvature-EMA + always-jump + post-hoc bench (cooldown)
opt = EAGLE(model.parameters(), lr=1e-3, base="adam",
            use_lr_in_eagle_update=False, adaptive_threshold=False,
            threshold=5e-4, curvature_ema=0.8, always_jump=True,
            cooldown_steps=20)

for step in range(T):
    opt.clean_step = True   # full-batch: every step is a clean measurement
    opt.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    opt.step()
```

`lr` tunes only the Adam fallback; the secant jump is learning-rate-free.
For the recommended default with a cosine-annealed fallback
(KESTREL+cos), see `eagle.baselines.KestrelCosine`.

Historical note: the optimizer class is named `EAGLE` because this code
base evolved from the earlier EAGLE optimizer (arXiv:2502.01036); the
KESTREL configuration corresponds to the CLI name `eagle-dqn-cd`
(`kestrel-cos` for the +cos variant).

## Reproducing the paper

All experiments run through a tuning-fair protocol (learning-rate-grid ×
3-seed family envelopes, fixed eval subsets, harness-counted gradient
evaluations, warm-started CUDA-synchronized timing):

```bash
# full measurement suite: regression, SIREN camera/astronaut, Kodak-24
bash experiments/run_ictai_baselines.sh
# grid-edge extensions + published-EAGLE baseline + KESTREL+cos
bash experiments/run_ictai_resume.sh
# K / beta_h sensitivity sweep (paper Table III)
bash experiments/run_ictai_sens.sh
# Rprop baseline across all domains
bash experiments/run_ictai_rprop.sh
# diagnostics: per-step cost microbenchmark + INR instrumentation
python experiments/bench_stepcost.py
python experiments/diagnose_kestrel_inr.py
python experiments/diagnose_secant.py

# analysis
python experiments/analyze_protocol.py --dataset california concrete energy \
    --prefix protoe protoeh protoe2 protoe3 protoe4 protoe5
python experiments/analyze_inr.py --prefix inrv3 inrv3c inrv3r
python ictai/gen_macros.py     # paper numbers (single-source, incl. bolding)
python ictai/paper_figs.py     # paper figures
```

Datasets: California Housing / Concrete / Energy via scikit-learn / UCI
(auto-download to `data/`); Kodak-24 PNGs go in `data/kodak/`
(`kodim01.png` ... `kodim24.png`, https://r0k.us/graphics/kodak/).
Results are written to `results/` (git-ignored).

Numerical equivalence of the fused CUDA kernel and the reference path:

```bash
python tests/test_equivalence.py
```

## License

MIT (see `LICENSE`).
