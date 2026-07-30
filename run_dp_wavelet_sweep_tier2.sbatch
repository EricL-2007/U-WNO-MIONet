#!/bin/bash
#SBATCH --job-name=uwno-mionet-dp-wavelet-tier2
#SBATCH --account=bhdh-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --gpus=1
#SBATCH --array=0-8%4
#SBATCH --output=%x_%A_%a.out

# TEMPLATE -- DO NOT SUBMIT YET. WNO_LR_TIER2 below is a placeholder; fill it in with
# whichever wno_lr wins Tier 1 for dP (run_dp_wno_lr_sweep_tier1.sbatch), then remove
# this comment block before submitting.
#
# Tier 2: wavelet basis x decomposition level, at Tier 1's winning wno_lr, 1 seed (42)
# per cell -- 9 cells (3 wavelets x 3 levels), the first-ever sweep of this knob.
# state_dir encodes wavelet+level explicitly (not just wno_lr+seed) so this cannot
# collide with any Tier 0/Tier 1 directory, including the db6/level=4 default cell
# which reuses the same (wno_lr, seed) as an already-completed run.
#
# --time=02:00:00 REUSES Tier 1's per-step timing (measured at level=4 only) as an
# approximation -- level=3/5 change the number of recursive DWT/IDWT calls and were
# NOT separately measured. Run a quick supplemental timing check (level=3, level=5;
# level=4 is already known) before trusting this budget for a real submission.
#
# NOTE: every cell here also retrains Fourier-MIONet, which has no wavelet component
# and is unaffected by wavelet/level (same fourier_lr, same seed each time) -- 9x
# redundant Fourier compute baked into the current one-invocation-trains-both-models
# design. Worth trimming (e.g. reuse Tier 1's Fourier checkpoint instead) if the real
# timing check shows this tier is tight against budget.

set -e

cd /scratch/bhdh/eliu7/U-WNO-MIONet

module purge
source venv/bin/activate

python3 -c "from wavelet_convolution import DWT, IDWT; print('wavelet imports OK')"

echo "Host: $(hostname)"
nvidia-smi

WNO_LR_TIER2="REPLACE_WITH_TIER1_WINNING_WNO_LR"
FOURIER_LR_OVERRIDE=1e-3
SEED=42

WAVELETS=(db4 db6 sym4)
LEVELS=(3 4 5)

# 3x3 grid: wavelet_idx = id / 3, level_idx = id % 3
wavelet_idx=$(( SLURM_ARRAY_TASK_ID / 3 ))
level_idx=$(( SLURM_ARRAY_TASK_ID % 3 ))
WAVELET="${WAVELETS[$wavelet_idx]}"
LEVEL="${LEVELS[$level_idx]}"
STATE_DIR="pre_train_sweep/dp_wavelet${WAVELET}_level${LEVEL}_wnolr${WNO_LR_TIER2}_seed${SEED}"

echo "=== dP Tier2 cell: wavelet=${WAVELET} level=${LEVEL} wno_lr=${WNO_LR_TIER2} seed=${SEED} state_dir=${STATE_DIR} ==="

DP_WNO_LAYERS="${DP_WNO_LAYERS:-1}" \
DP_WNO_WIDTH="${DP_WNO_WIDTH:-36}" \
python -u Fourier-UWNO-MIONet_dP_intermediate.py \
    --ntrain 1000 \
    --ntest 80 \
    --batch-size 10 \
    --timestep-batch-size 8 \
    --iterations 2500 \
    --display-every 25 \
    --seed "$SEED" \
    --fourier-lr "$FOURIER_LR_OVERRIDE" \
    --wno-lr "$WNO_LR_TIER2" \
    --wavelet "$WAVELET" \
    --wavelet-level "$LEVEL" \
    --state-dir "$STATE_DIR"
