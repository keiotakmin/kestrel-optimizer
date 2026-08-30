# Reproducibility statement

This document states precisely **what this artifact reproduces and what it
cannot**. Please read it before comparing numbers.

## 1. KESTREL trajectories are not bit-reproducible

KESTREL is extremely sensitive to initial conditions. Perturbing a *single
weight by one ULP* and running the *same* implementation with the *same* seed
produces the following divergence in the maximum absolute parameter
difference (SIREN, 65 536 collocation points, `lr=1e-3`):

| step | KESTREL | Adam | Rprop |
|-----:|--------:|-----:|------:|
| 1  | 4.7e-6 | 4.7e-6 | 3.0e-8 |
| 3  | **9.2e+0** | 4.7e-6 | 3.0e-8 |
| 5  | **3.6e+1** | 5.5e-6 | 3.0e-8 |
| 40 | 3.9e+1 | 6.7e-6 | 3.7e-2 |

One ULP grows to O(10) within five steps; Adam shows no amplification over
40 steps. The mechanism is plausibly catastrophic cancellation in the secant
difference `Δg = g_t - g_{t-1}` combined with the `g / h` amplification when
the curvature estimate `h` is small.

Consequences:

* Two runs that differ only in floating-point reduction order — a different
  GPU, a different BLAS/cuDNN version, the fused kernel versus the reference
  path — will follow **different trajectories** and end at different PSNR.
* A single run's final metric is therefore **not** a reproducibility target.
* What *is* reproducible is the **statistics over the pre-registered
  evaluation units** (image x seed). All claims in this project are made at
  that level, over 48 units per dataset.

## 2. How implementation equivalence is tested

Because of (1), comparing whole trajectories cannot distinguish an
implementation bug from chaotic amplification. The fused CUDA kernel and the
reference (vectorized) path are therefore compared **one step at a time from
an identical state**, at several points along a trajectory:

```bash
python experiments/verify_fused.py            # single-step equivalence + timing
python experiments/verify_fused.py --no-timing # equivalence only
```

Observed agreement: maximum relative parameter difference **1.2e-6** after one
step, i.e. float32 rounding. The two paths implement the same update rule.

## 3. Wall-clock is only reported for symmetric implementations

Optimizers here fall into three implementation classes: `fused` (single CUDA
kernel), `foreach` (PyTorch multi-tensor path) and `python` (per-tensor
operations). Comparing across classes measures kernel engineering, not the
algorithm.

`experiments/report_layers.py` enforces this mechanically. It reads each
result cell's metadata and **refuses to print a wall-clock column** unless all
compared families share one implementation class and one measurement host:

```
===== systems layer =====
implementation classes: ['foreach', 'python']
**wall-clock is not reported**: mixed implementation classes
```

Every result cell records `fused_available`, `host`, `commit`, `git_dirty`,
`torch`, `cuda`, `gpu`, `dtype`, TF32 flags and the protocol hash, so this
check is automatic for any newly produced data.

The algorithmic layer — optimizer **steps** and **gradient evaluations** — is
implementation-independent and is what the headline claims rest on.

## 4. Pre-registration

Selection and evaluation are separated by files that are written *before* the
corresponding runs and are not edited afterwards:

| file | fixes |
|---|---|
| `experiments/kodak_split.json` | tuning / evaluation image split, selection metric |
| `experiments/kodak_lr_lock.json` | the learning rate of every family, with the full per-lr statistics behind the choice |
| `experiments/div2k_split.json` | the never-used confirmatory image set, its preprocessing, and the PSNR thresholds |
| `experiments/mech_lr_lock.json` | the learning rate of every mechanism configuration |

The selection rule is identical for all families: **primary** = median final
PSNR on the tuning subset, **secondary** = censored reach to the threshold
(reach rate first, then median reach steps). A family whose optimum sits at a
grid endpoint is not bracketed; its grid is extended outward by a factor of
three until the optimum is interior, and only then is the rate locked.

Known limitation, stated in `kodak_split.json` itself: the Kodak split reduces
selection bias but is not fully independent of earlier exploratory analysis on
the same 24 images. This is why the confirmatory evaluation uses a separate,
never-used image set.

## 5. Aggregation rules

Reach-to-threshold is **censored**, not filtered:

* every family's reach rate `n_reached / n` is always shown;
* images that never reach the threshold stay in the table as `> budget`
  rather than being dropped;
* the Kaplan-Meier style median is reported as `>budget` whenever the reach
  rate is at or below 50 %, instead of quoting a median over survivors only;
* paired comparisons count an image where only one method reached as a win for
  that method, so a method cannot gain by failing more often.

Thresholds are cut out of one long run per cell; no separate run is made per
threshold or per budget.

## 6. Environment

`pip install -e .` plus, optionally, `adabelief-pytorch` and
`pytorch_optimizer` for two baselines. Python >= 3.10, PyTorch >= 2.x.

The fused CUDA kernel is JIT-compiled on first use and requires `nvcc` and
`ninja`; where either is missing the code falls back to the vectorized path
automatically and records `fused_available: false` in the run metadata. On
hosts whose default GCC is newer than the CUDA toolkit supports, the opt-in
environment variable `EAGLE_ALLOW_UNSUPPORTED_COMPILER=1` adds
`-allow-unsupported-compiler`; if you use it, run
`python experiments/verify_fused.py` before trusting any measurement.
