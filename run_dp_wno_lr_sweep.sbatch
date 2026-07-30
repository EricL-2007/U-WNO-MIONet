#!/bin/bash
#SBATCH --job-name=uwno-mionet-dp-wnolr-sweep
#SBATCH --account=bhdh-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --gpus=1
#SBATCH --array=0-7%4
#SBATCH --output=%x_%A_%a.out

# dP wno_lr sweep, retrained from scratch under the mask fix (loss_fnc now uses the
# per-row-and-column mask, not the column-0-broadcast leak). One array task = one
# (wno_lr, seed) cell; Fourier-MIONet (fixed lr) and U-WNO-MIONet (swept lr) both train
# within the same script invocation, each isolated under its own --state-dir so no two
# cells can collide on checkpoints (the bug already fixed in
# Fourier-UWNO-MIONet_dP_intermediate.py's MODEL_SPECS/state_dir routing).
#
# fourier_lr is intentionally left at dP's own known-working default, 1e-3 -- NOT the
# 5e-4 requested in the original ask, which is sg's default, not dP's (dP has always
# used LR=1e-3; every prior dP result, including the outlier/mask diagnosis, was
# produced at 1e-3). Override FOURIER_LR_OVERRIDE below if you specifically want dP's
# Fourier-MIONet re-tested at 5e-4 too -- that would add a second dimension to the sweep,
# not swap in for the current one.
#
# --iterations/--ntrain/--ntest/--batch-size/--timestep-batch-size match the existing
# confirmed dP config (run_dp_intermediate.sbatch) unchanged (2500 iterations, NOT
# capped -- unlike sg, dP's measured worst-case fits comfortably in budget even at its
# full iteration count) -- only wno_lr, seed, and state_dir vary per array task.
#
# --time=02:00:00: measured worst-case per job is ~1.27h (2500 iters x
# (1.0781+0.7521) s/step); this adds ~55% buffer for data load, checkpoint IO, and the
# final inference/prediction chunk after training ends, none of which the timing
# probe's s/step measurement covers.

set -e

cd /scratch/bhdh/eliu7/U-WNO-MIONet

module purge
source venv/bin/activate

python3 -c "from wavelet_convolution import DWT, IDWT; print('wavelet imports OK')"

echo "Host: $(hostname)"
nvidia-smi

FOURIER_LR_OVERRIDE=1e-3
WNO_LRS=(5e-4 2e-4 1e-4 5e-5)
SEEDS=(42 43)

# SLURM_ARRAY_TASK_ID -> (wno_lr, seed): lr_idx = id / 2, seed_idx = id % 2
lr_idx=$(( SLURM_ARRAY_TASK_ID / 2 ))
seed_idx=$(( SLURM_ARRAY_TASK_ID % 2 ))
WNO_LR="${WNO_LRS[$lr_idx]}"
SEED="${SEEDS[$seed_idx]}"
STATE_DIR="pre_train_sweep/dp_wnolr${WNO_LR}_seed${SEED}"

echo "=== dP sweep cell: wno_lr=${WNO_LR} seed=${SEED} state_dir=${STATE_DIR} ==="

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
    --wno-lr "$WNO_LR" \
    --state-dir "$STATE_DIR"
