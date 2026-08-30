# KESTREL

Research code for **KESTREL**, a coordinate-wise secant optimizer with
post-hoc failure benching for **deterministic full-batch** objectives
(implicit neural representation fitting, full-batch regression).

KESTREL maintains a per-coordinate secant estimate of the diagonal curvature
from consecutive gradients at no extra gradient cost, takes learning-rate-free
Newton-like steps on the coordinates with a valid estimate, and secures them
with a single safeguard: a **post-hoc bench** that detects a failed jump from
the gradient response and demotes only the failing coordinates to Adam for
`K = 20` steps.

This repository holds the optimizer, the full measurement pipeline, the
pre-registration files that fix every selection decision, and the analysis
code that turns raw runs into the reported tables.

## Install

```bash
pip install -e .                                   # core
pip install adabelief-pytorch pytorch_optimizer    # two extra baselines
```

Python >= 3.10, PyTorch >= 2.x. On CUDA the fused kernel is JIT-compiled on
first use; without `nvcc`/`ninja` the code falls back to a vectorized path and
records that fact in the run metadata.

## Quick start

```python
from eagle.optim import EAGLE

opt = EAGLE(model.parameters(), lr=1e-3, base="adam",
            use_lr_in_eagle_update=False, adaptive_threshold=False,
            threshold=5e-4, curvature_ema=0.8, always_jump=True,
            cooldown_steps=20)

for step in range(T):
    opt.clean_step = True        # full batch: every step is a clean measurement
    opt.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    opt.step()
```

`lr` tunes only the Adam fallback; the secant jump itself carries no learning
rate. The recommended default pairs KESTREL with a cosine-annealed fallback:
`eagle.baselines.KestrelCosine`.

The optimizer class is called `EAGLE` because this code base grew out of the
EAGLE optimizer (arXiv:2502.01036). KESTREL is the configuration registered as
`eagle-dqn-cd`; `kestrel-cos` is the cosine-fallback variant. The published
EAGLE configuration is available under the name `eagle` and is included as a
baseline.

## Running the experiments

One entry point, resumable at every stage. A cell already on disk is skipped,
and no stage overwrites an existing results directory.

```bash
bash experiments/run_pipeline.sh all      # or: verify base lock main confirm ...
```

| stage | what it does |
|---|---|
| `verify` | numerical checks: reference vs fused, single-step equivalence, per-step cost |
| `base` | learning-rate grids for every optimizer family on the INR suite |
| `lock` | grid-saturation audit, outward grid expansion, learning-rate lock |
| `main` | main evaluation on the held-out image subset, 3 seeds |
| `confirm` | confirmatory evaluation on a never-used image set, 3 seeds |
| `prior` | published-EAGLE arm under the same selection rule |
| `mech` | 2^4 factorial mechanism study with per-step jump/bench diagnostics |
| `coupling` | coupled-quadratic diagnostics with ground-truth jump labels |
| `analyze` | censored-reach tables, two-layer report, mechanism and coupling summaries |

## How claims are protected

Four properties of this pipeline are deliberate and are what the reported
numbers rest on. They are described in full in **[REPRODUCIBILITY.md](REPRODUCIBILITY.md)**.

1. **Selection is separated from evaluation.** Learning rates are chosen on a
   tuning subset by a rule fixed in advance, written to
   `experiments/kodak_lr_lock.json` together with the per-rate statistics
   behind each choice, and never re-chosen on the evaluation data. A separate
   never-used image set (`experiments/div2k_split.json`) provides a
   confirmatory evaluation with the same locked settings.

2. **No family is compared at a grid edge.** `audit_grid_saturation.py` finds
   families whose optimum sits at an endpoint; `expand_grid_kodak.py` extends
   the grid outward by a factor of three until the optimum is interior.

3. **Reach is censored, never filtered.** Reach rates are always shown,
   non-reaching images stay in the table as `> budget`, medians are withheld
   when the reach rate is at or below 50 %, and paired comparisons count an
   image where only one method reached as a win for that method.

4. **Wall-clock is gated on implementation symmetry.**
   `report_layers.py` separates the algorithmic layer (steps, gradient
   evaluations) from the systems layer (time) and refuses to print a time
   column unless every compared family shares one implementation class
   (`fused` / `foreach` / `python`) and one measurement host.

**KESTREL trajectories are not bit-reproducible.** A single-ULP perturbation
of one weight grows to O(10) in parameter space within five steps, where Adam
shows no amplification at all. Claims are therefore made over the
pre-registered evaluation units (image x seed), never over a single run.
See REPRODUCIBILITY.md section 1.

## Repository layout

```
src/eagle/            optimizer, fused CUDA kernel, baselines, models, data
experiments/
  run_pipeline.sh     single entry point for all stages
  *_split.json        pre-registered image splits and thresholds
  *_lr_lock.json      pre-registered learning rates with their evidence
  pilot_inr.py        SIREN INR training harness
  run_eval_main.py    evaluation runner (locked rates, or an explicit grid)
  audit_grid_saturation.py / expand_grid_kodak.py / lock_kodak_lr.py
                      grid saturation audit, expansion and locking
  analyze_reach.py    censored reach, paired sign tests
  report_layers.py    algorithmic vs systems reporting with the wall-clock gate
  bench_probe.py      per-step jump/bench diagnostics (read-only optimizer hook)
  run_bench_mechanism.py / lock_mech_lr.py / analyze_mechanism.py
                      factorial mechanism study
  diag_coupling.py / analyze_coupling.py
                      coupled quadratics with ground-truth jump labels
  verify_fused.py     single-step fused/reference equivalence and timing
  prepare_div2k.py    fetch and preprocess the confirmatory image set
tests/                reference implementation and equivalence tests
results/analysis/     small analysis outputs (checked in)
results/coupling/     coupled-quadratic sweep records (checked in)
```

Large per-run measurement files are written under `results/` and are
git-ignored; the small analysis outputs that the tables are built from are
checked in.

## Diagnostics

`bench_probe.JumpProbe` attaches to the optimizer through a read-only hook and
records, per step, the jump rate, the jump displacement relative to the Adam
step, the benched-coordinate fraction, the failure and re-failure rates, the
recovery time, the fraction of rejected negative secants, and a four-way
breakdown of what happened to the coordinates that jumped (detected failure /
undetected sign-preserving failure / false alarm / good jump). The hook does
not participate in the update; a probe-on / probe-off run is bit-identical.

`diag_coupling.py` runs the same probe on quadratics where the optimum is
known, so the same four-way breakdown can be scored against ground truth and
the observable proxies can be validated against it.

## Data

Nothing is redistributed here; every acquisition step is scripted. Sources and
terms are listed in **[DATA.md](DATA.md)**.

## License

MIT (see `LICENSE`).
