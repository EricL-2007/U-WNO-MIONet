#!/bin/bash
#SBATCH --job-name=uwno-mionet-sg-wnolr-sweep
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

# sg wno_lr sweep, retrained from scratch under the mask fix (Rsquare_plume_tegother/
# MAE_plume now use the per-row-and-column mask -- sg has no masked training loss, so
# this is an eval-metric-only fix for sg, already effective without retraining; this
# sweep's purpose is the wno_lr instability investigation, not the mask). One array
# task = one (wno_lr, seed) cell; Fourier-MIONet (fixed lr) and U-WNO-MIONet (swept lr)
# both train within the same script invocation, isolated under their own --state-dir
# (sg's state_dir routing was already correct before this sweep).
#
# fourier_lr stays at sg's own known-working default, 5e-4 -- unchanged, matches the
# original ask directly (this IS sg's actual current default, unlike dP's).
#
# --iterations is CAPPED to 1200 (down from sg's full 7500) specifically for this
# sweep, per the timing-probe results: measured s/step (Fourier 0.7359, WNO 3.1030)
# put the uncapped worst-case (no early stopping) at ~63.9h for sg's 8 cells alone,
# which risked blowing the 48h budget ceiling outright. 1200 is ~2x every observed
# sg early-stopping point this session (all under 600 steps) while bounding the
# worst case if EarlyStopping doesn't fire for some wno_lr value -- exactly the
# scenario this sweep exists to find. DECAY_STEP/WNO_PATIENCE etc. all re-derive from
# --iterations automatically (see Fourier-UWNO-MIONet_sg.py's __main__), so this cap
# doesn't desync any other schedule.
#
# --time=02:00:00: measured worst-case per job is ~1.28h (1200 iters x
# (0.7359+3.1030) s/step); this adds ~50% buffer for data load, checkpoint IO, and
# the final inference/prediction chunk after training ends, none of which the
# timing probe's s/step measurement covers.

set -e

cd /scratch/bhdh/eliu7/U-WNO-MIONet

module purge
source venv/bin/activate

python3 -c "from wavelet_convolution import DWT, IDWT; print('wavelet imports OK')"

echo "Host: $(hostname)"
nvidia-smi

FOURIER_LR=5e-4
WNO_LRS=(5e-4 2e-4 1e-4 5e-5)
SEEDS=(42 43)

# SLURM_ARRAY_TASK_ID -> (wno_lr, seed): lr_idx = id / 2, seed_idx = id % 2
lr_idx=$(( SLURM_ARRAY_TASK_ID / 2 ))
seed_idx=$(( SLURM_ARRAY_TASK_ID % 2 ))
WNO_LR="${WNO_LRS[$lr_idx]}"
SEED="${SEEDS[$seed_idx]}"
STATE_DIR="pre_train_sweep/sg_wnolr${WNO_LR}_seed${SEED}"

echo "=== sg sweep cell: wno_lr=${WNO_LR} seed=${SEED} state_dir=${STATE_DIR} ==="

python -u Fourier-UWNO-MIONet_sg.py \
    --iterations 1200 \
    --seed "$SEED" \
    --fourier-lr "$FOURIER_LR" \
    --wno-lr "$WNO_LR" \
    --state-dir "$STATE_DIR"
