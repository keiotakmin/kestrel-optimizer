#!/usr/bin/env bash
# KESTREL: end-to-end measurement pipeline.
#
# Every stage is resumable: a cell that already exists on disk is skipped, so
# re-running the same command after an interruption continues where it stopped.
# No stage ever overwrites an existing results directory; additional runs go to
# a new prefix.
#
# Usage:
#   bash experiments/run_pipeline.sh <stage> [<stage> ...]
#   bash experiments/run_pipeline.sh all
#
# Stages:
#   verify     numerical checks (fused vs reference, single-step equivalence)
#   base       INR learning-rate grids for all optimizer families
#   lock       grid-saturation audit, outward expansion, learning-rate lock
#   main       main evaluation on the held-out Kodak subset (3 seeds)
#   confirm    confirmatory evaluation on a never-used image set (DIV2K, 3 seeds)
#   prior      published-EAGLE arm (grid on tuning subset, then evaluation)
#   mech       factorial mechanism study (stage A -> lock -> stage B diagnostics)
#   coupling   coupled-quadratic diagnostics (synthetic, ground-truth labels)
#   analyze    all analyses and reports
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
SEEDS=${SEEDS:-"42 43 44"}

stage_verify() {
  $PY tests/test_equivalence.py
  $PY experiments/verify_fused.py || \
    echo "[note] no fused kernel on this host; wall-clock claims are not available here"
}

stage_base() {
  # Learning-rate grids for the baseline families on the INR suite.
  bash experiments/run_ictai_baselines.sh
  bash experiments/run_ictai_resume.sh
  bash experiments/run_ictai_rprop.sh
}

stage_lock() {
  # 1. which families were selected at a grid edge?
  $PY experiments/audit_grid_saturation.py
  # 2. extend those grids outward on the tuning subset only
  $PY experiments/expand_grid_kodak.py --prefix kodakx --max-rounds 3
  # 3. apply the pre-registered rule and write the canonical lock file
  $PY experiments/lock_kodak_lr.py
}

stage_main() {
  $PY experiments/run_eval_main.py --image-set kodak-evaluation \
      --prefix kodake --seeds $SEEDS
}

stage_confirm() {
  $PY experiments/prepare_div2k.py
  $PY experiments/run_eval_main.py --image-set div2k \
      --prefix div2ka --seeds $SEEDS
}

stage_prior() {
  # published EAGLE (arXiv:2502.01036) under the same pre-registered rule
  $PY experiments/run_eval_main.py --image-set kodak-tuning --families eagle \
      --lr-grid 3e-5 1e-4 3e-4 1e-3 3e-3 --prefix eaglet --seeds 42
  $PY experiments/run_eval_main.py --image-set kodak-evaluation \
      --families eagle --prefix kodakg --seeds $SEEDS
  $PY experiments/run_eval_main.py --image-set div2k \
      --families eagle --prefix div2kg --seeds $SEEDS
}

stage_mech() {
  $PY experiments/run_bench_mechanism.py --stage A --prefix mechA
  $PY experiments/lock_mech_lr.py --prefix mechA
  $PY experiments/run_bench_mechanism.py --stage B --prefix mechB
}

stage_coupling() {
  $PY experiments/diag_coupling.py --conds 10 100 1000 --seeds 0 1 2 3 4 \
      --out results/coupling
}

stage_analyze() {
  mkdir -p results/analysis
  for split in evaluation div2k; do
    $PY experiments/analyze_reach.py --split $split --seeds $SEEDS \
        --threshold 30 --json results/analysis/reach_${split}.json
    $PY experiments/report_layers.py --split $split \
        --json results/analysis/layers_${split}.json
  done
  $PY experiments/analyze_mechanism.py --prefix mechB \
      --json results/analysis/mechanism.json
  $PY experiments/analyze_coupling.py --in results/coupling/coupling_sweep.json
}

ALL="verify base lock main confirm prior mech coupling analyze"
[ "$#" -gt 0 ] || { echo "usage: $0 <stage>... | all"; echo "stages: $ALL"; exit 1; }
[ "$1" = "all" ] && set -- $ALL
for s in "$@"; do
  echo "=============== stage: $s  ($(date '+%F %T')) ==============="
  "stage_$s"
done
echo "pipeline done: $(date '+%F %T')"
