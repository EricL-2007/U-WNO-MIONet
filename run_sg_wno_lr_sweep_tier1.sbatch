#!/bin/bash
#SBATCH --job-name=uwno-mionet-sg-wnolr-tier1
#SBATCH --account=bhdh-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --gpus=1
#SBATCH --array=0-6%4
#SBATCH --output=%x_%A_%a.out

# Tier 1 LR-grid extension for sg, per the sweep result showing WNO's R2 monotonic in
# wno_lr with the best value (5e-5) at the EDGE of the tested grid -- extending
# downward to bracket the optimum. 7 cells (not a clean grid): a 3rd-seed rerun of the
# existing best (5e-5) for direct comparability, plus 3 seeds each at 2e-5 and 1e-5.
#
# --iterations stays capped at 1200 (same cap as the original sweep, still ~2x every
# observed early-stopping point). --time=02:00:00, unchanged: LR does not affect
# per-step compute cost, so the timing-probe numbers already measured
# (Fourier 0.7359 s/step, WNO 3.1030 s/step) still apply directly; no new probe
# needed for this tier.

set -e

cd /scratch/bhdh/eliu7/U-WNO-MIONet

module purge
source venv/bin/activate

python3 -c "from wavelet_convolution import DWT, IDWT; print('wavelet imports OK')"

echo "Host: $(hostname)"
nvidia-smi

FOURIER_LR=5e-4

# Explicit (wno_lr, seed) pairs, NOT a full cross product -- index 0 is the 3rd-seed
# rerun of the existing best (5e-5) for direct comparability; indices 1-6 are the two
# new wno_lr values x 3 seeds each.
CELL_LRS=(5e-5  2e-5  2e-5  2e-5  1e-5  1e-5  1e-5)
CELL_SEEDS=(44   42    43    44    42    43    44)

WNO_LR="${CELL_LRS[$SLURM_ARRAY_TASK_ID]}"
SEED="${CELL_SEEDS[$SLURM_ARRAY_TASK_ID]}"
STATE_DIR="pre_train_sweep/sg_wnolr${WNO_LR}_seed${SEED}"

echo "=== sg Tier1 cell: wno_lr=${WNO_LR} seed=${SEED} state_dir=${STATE_DIR} ==="

python -u Fourier-UWNO-MIONet_sg.py \
    --iterations 1200 \
    --seed "$SEED" \
    --fourier-lr "$FOURIER_LR" \
    --wno-lr "$WNO_LR" \
    --state-dir "$STATE_DIR"
