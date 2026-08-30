# KESTREL

Research code for **KESTREL**, a coordinate-wise secant optimizer with
post-hoc failure benching for **deterministic full-batch** objectives such as
implicit neural representation (INR) fitting and full-batch regression.

In this regime curvature is available at no extra gradient cost, and KESTREL
turns it into a measurable reduction in optimizer steps - within a
characterised band of target qualities, and with a safeguard whose limits are
measured rather than assumed. *What this repository establishes* below states
the findings and points at the code and the stored results behind each one.

## The method

**Setting.** The objective is deterministic: the same parameters always
produce the same gradient, because the whole batch is used at every step.
Two consecutive gradients then differ only because the parameters moved, so
the finite difference along each coordinate is a curvature measurement rather
than a noise sample. KESTREL is built for exactly this regime and deliberately
degrades to Adam when it does not hold.

**Design principle.** Do not try to predict which secant steps will succeed.
Take them all, detect the failures from the gradient response one step later,
and demote only the failing coordinates.

**Notation.** All operations are element-wise over coordinates `i`. At step
`t`: `g` is the current gradient, `g⁻` the previous one, `θ⁻` the previous
parameters. State carried per coordinate: Adam moments `m, v`; the curvature
estimate `h`; the flag `jumped`; the counter `cooldown`. Constants:
`β₁, β₂, ε` (Adam), `β_h = 0.8` (curvature EMA), `K = 20` (bench length),
`lr` (fallback learning rate only).

```
for each step t:
    g ← ∇L(θ)                                  # one gradient, full batch

    # --- 1. Adam fallback branch (always maintained) ------------------
    m ← β₁·m + (1-β₁)·g
    v ← β₂·v + (1-β₂)·g²
    v̂ ← v / (1-β₂ᵗ)
    adam_step ← lr/(1-β₁ᵗ) · m / (√v̂ + ε)

    # --- 2. Curvature measurement (secant, no extra gradient) ---------
    Δθ ← θ - θ⁻
    Δg ← g - g⁻
    where |Δθ| > 1e-12:
        h_new ← Δg / Δθ                        # diagonal curvature estimate
        where h_new > 0:                       # positive measurements only
            h ← β_h·h + (1-β_h)·h_new  if h > 0  else  h_new
    # a non-positive or undefined measurement is discarded; h keeps its value

    # --- 3. Post-hoc bench: judge LAST step's jumps by this step's g ---
    failed ← jumped ∧ (g⁻·g < 0) ∧ (|g| > |g⁻|)    # overshoot signature
    cooldown ← K            where failed
    benched  ← cooldown > 0
    cooldown ← cooldown - 1 where benched

    # --- 4. Select the update per coordinate --------------------------
    jump ← (h > 0) ∧ ¬benched
    θ ← θ - g / max(h, 1e-12)      where jump         # Newton-like, no lr
    θ ← θ - adam_step              where ¬jump        # Adam fallback

    # --- 5. Bookkeeping ------------------------------------------------
    jumped ← jump
    θ⁻ ← θ (pre-update),  g⁻ ← g
```

Five properties follow directly from the pseudocode.

1. **The jump carries no learning rate.** `g / h` is a Newton step on the
   diagonal. `lr` tunes only the Adam fallback, which is why pairing KESTREL
   with a cosine-annealed fallback changes the finish without touching the
   jumps.
2. **Curvature costs nothing extra.** `Δg / Δθ` reuses gradients the optimizer
   already computed: one gradient evaluation per step, the same as Adam, and
   no Hessian-vector products or line searches.
3. **Only positive curvature is admitted.** A Newton step along a direction of
   negative measured curvature points away from the minimum, so those
   measurements are discarded and the stale positive value is kept. This is a
   deliberate blind spot and it is measured, not assumed: see
   `bench_probe.py`'s `neg_secant_rate`.
4. **Failure detection is post-hoc, and it is a proxy.** A coordinate is
   judged only after its jump, by whether the gradient flipped sign *and* grew.
   That signature catches overshoot; it cannot see a failure that preserves the
   gradient sign. `diag_coupling.py` scores this detector against ground truth
   on quadratics where the optimum is known.
5. **Benching is per coordinate.** A failing coordinate falls back to Adam for
   `K` steps while every other coordinate keeps jumping. In practice the bench
   is far from a rare correction: on the six images of the mechanism study it
   holds about 88 % of coordinates on the Adam branch at any time, and its
   measured effect falls on the worst case rather than on the average
   (`analyze_mechanism.py`).

**Variants.** The configuration above is KESTREL; internally it is registered
as `eagle-dqn-cd`. `kestrel-cos` adds a cosine schedule on the fallback
learning rate and is the recommended default. The pipeline also builds the
published EAGLE configuration (arXiv:2502.01036), registered as `eagle`, as a
baseline, along with the factorial variants that switch the four mechanisms
(always-jump, bench, pre-gate, trust region) on and off.

**Cost.** Per step: one gradient evaluation and one element-wise pass. Per
coordinate state: Adam's `m, v` plus `θ⁻`, `g⁻`, `h` in float32 and `jumped`,
`cooldown` in uint8. A fused CUDA kernel performs the whole update in a single
launch; a vectorized PyTorch path is used wherever the kernel is unavailable,
and the two are verified to agree step by step.

## What this repository establishes

Every number below is produced by the pipeline in this repository and is
stored in `results/analysis/`. The INR results use SIREN fitting; the
statistical unit is one (image, seed) pair, 48 units per dataset (16 images x
3 seeds), with learning rates selected on a separate tuning subset and never
re-chosen on the evaluation data.

### 1. Free curvature buys optimizer steps, and it survives a held-out dataset

On a never-used image set (DIV2K, first 16 validation images), with every
learning rate locked in advance on Kodak, KESTREL+cos reaches 30 dB with the
**highest reach rate (37/48) and the highest median final PSNR (30.79 dB)** of
eleven optimizers. Paired over images, it beats every baseline:

| baseline | paired W/L/T | sign-test p | steps ratio |
|---|---|---|---|
| Adam+cos | 28 / 12 / 1 in KESTREL's favour | 0.017 | 1.42x |
| Rprop | 25 / 12 / 1 | 0.047 | 1.00x |
| published EAGLE | 36 / 2 / 0 | < 0.001 | 1.85x |
| Adam, AdamW, AdaBelief, L-BFGS, AdaHessian, BB-stab | 30-37 / 0-6 | < 0.001 | 1.3-6x |

The same ordering holds on the held-out Kodak subset (KESTREL+cos beats
Adam+cos 29/5, p < 0.001).

Curvature is obtained from gradients the optimizer already computes, so the
step advantage is also a **gradient-evaluation** advantage. The two curvature
baselines pay for their information: at 30 dB on DIV2K, L-BFGS needs 1422
gradient evaluations against 570 optimizer steps, and AdaHessian 1720 against
860 (`report_layers.py`).

### 2. The advantage is banded, not general

Cutting the same runs at the pre-registered thresholds shows three regimes
rather than one winner (DIV2K, paired against KESTREL+cos):

| threshold | KESTREL+cos reach | vs Adam+cos | vs Rprop |
|---|---|---|---|
| 28 dB | 41/48 | wins, p < 0.001 | **loses**, Rprop 1.99x faster, p < 0.001 |
| 30 dB | 37/48 (best) | wins, p = 0.017 | wins, p = 0.047 |
| 32 dB | 10/48 | no difference, p = 1.000 | no difference, p = 0.227 |
| 34 dB | 2/48 | no difference; Adam+cos reaches more often | Rprop never reaches |

Rprop owns the low-fidelity band, KESTREL+cos the middle, Adam+cos the finish.
On the held-out Kodak subset Adam+cos also ends at a higher median final PSNR
(31.73 vs 31.36 dB) despite losing on reach. Reporting a single threshold
would hide all of this.

### 3. The safeguard protects the worst case, not the average

A 2^4 factorial over the four mechanisms (always-jump, post-hoc bench,
pre-gate, trust region) with the same locking rule
(`run_bench_mechanism.py`, `analyze_mechanism.py`):

* KESTREL is **not** a method that jumps everywhere. The bench holds about
  **88 % of coordinates on the Adam branch**; roughly 12 % take a secant step
  at any given time.
* Removing the bench moves the median final PSNR by +0.5 dB but the **minimum
  over images from 26.01 dB to 16.09 dB**. The safeguard buys the tail.
* The trust region is not merely redundant, it is **harmful**: -0.66 dB as a
  main effect. Pre-gating and always-jump contribute +0.40 and +0.30 dB.

### 4. The failure detector has a measurable blind spot

On coupled quadratics the optimum is known, so every jump can be labelled as
genuinely harmful or not and the bench's firing condition scored against that
ground truth (`diag_coupling.py`, 2835 runs):

* precision **0.44-0.64**, recall **0.17-0.30**: the detector misses roughly
  four out of five genuinely harmful jumps;
* **52-70 % of harmful jumps preserve the gradient sign**, and that share
  grows with coupling. The condition "sign flip and growing gradient" cannot
  see them by construction;
* secant jumps break down at a coupling strength of **||D^-1 E|| ~ 3-4**,
  consistently across condition numbers 10, 100 and 1000 - the condition
  number only sets how much off-diagonal mixing is needed to get there.

The quantity `neg_secant_rate`, which is observable without knowing the
optimum, tracks the true harmful-jump rate at **Spearman +0.93** (n = 315).
The diagnostics therefore transfer to real tasks, where no ground truth
exists.

### 5. The optimizer is chaotic, and that shows up as rare collapses

Perturbing a single weight by one ULP and rerunning the same implementation
with the same seed changes the parameters by O(10) within five steps; Adam
shows no amplification over 40 steps. Consistent with this, KESTREL-family
runs collapse below 20 dB on 1-2 of 48 units per dataset, where Adam, Adam+cos
and Rprop never do. Trajectories are therefore not bit-reproducible and all
claims here are made over the 48 pre-registered units, never over a single
run. `REPRODUCIBILITY.md` gives the measurement and its consequences.

## Install

```bash
pip install -e .                                   # core
pip install adabelief-pytorch pytorch_optimizer    # two extra baselines
```

Python >= 3.10, PyTorch >= 2.x. On CUDA the fused kernel is JIT-compiled on
first use; without `nvcc`/`ninja` the code falls back to a vectorized path and
records that fact in the run metadata.

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

## Evaluation protocol

The numbers above depend on four rules that the code enforces rather than
documents. `REPRODUCIBILITY.md` states them in full.

1. **Selection is separated from evaluation.** Learning rates are chosen on a
   tuning subset by a rule fixed in advance and written, with the per-rate
   statistics behind each choice, to `experiments/kodak_lr_lock.json`. They
   are never re-chosen on evaluation data, and the confirmatory image set in
   `experiments/div2k_split.json` had never been used when it was registered.
2. **No family is compared at a grid edge.** `audit_grid_saturation.py` finds
   families whose optimum sits at an endpoint; `expand_grid_kodak.py` widens
   the grid by a factor of three until it is interior.
3. **Reach is censored, never filtered.** Reach rates are always shown,
   non-reaching images stay as `> budget`, medians are withheld below a 50 %
   reach rate, and a paired comparison scores a one-sided reach as a win for
   the method that reached - so a method cannot gain by failing more often.
4. **Wall-clock is gated on implementation symmetry.** `report_layers.py`
   separates optimizer steps and gradient evaluations from wall-clock, and
   refuses to print a time column unless every compared family shares one
   implementation class (`fused` / `foreach` / `python`) and one measurement
   host.

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

**Naming.** The optimizer class is called `EAGLE` and the package `eagle`
because this code base grew out of the EAGLE optimizer (arXiv:2502.01036).
KESTREL is a configuration of that class, not a separate implementation; the
mapping between the names used in the code and the methods described above is
given under *Variants*.

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
