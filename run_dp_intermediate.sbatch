#!/bin/bash
#SBATCH --job-name=uwno-mionet-dp-intermediate
#SBATCH --account=bhdh-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --gpus=1
#SBATCH --output=%x_%j.out

set -e  # fail fast — see wavelet import sanity check below

cd /scratch/bhdh/eliu7/U-WNO-MIONet

module purge
source venv/bin/activate

# Catch a broken wavelet_convolution.py import immediately, not partway into training.
python3 -c "from wavelet_convolution import DWT, IDWT; print('wavelet imports OK')"

echo "Host: $(hostname)"
nvidia-smi

# If this run gets preempted/times out, re-submitting this exact sbatch script picks
# back up automatically -- Fourier-UWNO-MIONet_dP_intermediate.py checks
# pre_train/dP_intermediate_{fourier,wno}_train_state.json on startup and resumes from
# the last saved step (weights + optimizer + LR-scheduler state + EarlyStopping
# counters) rather than restarting either model from scratch. No special flag needed;
# just resubmit.
#
# DP_WNO_LAYERS=1 is the new best-known capacity config from the layers A/B test
# (ntrain=400 reduced scale: 4->2 layers fixed catastrophic overfitting, 2->1 showed no
# further change). DP_WNO_WIDTH defaults to 36 here (unconfirmed pending that A/B test's
# width leg) -- override to whatever width Task 1's test shows helps, e.g.:
#   DP_WNO_WIDTH=18 sbatch run_dp_intermediate.sbatch
DP_WNO_LAYERS="${DP_WNO_LAYERS:-1}" \
DP_WNO_WIDTH="${DP_WNO_WIDTH:-36}" \
python -u Fourier-UWNO-MIONet_dP_intermediate.py \
    --ntrain 1000 \
    --ntest 80 \
    --batch-size 10 \
    --timestep-batch-size 8 \
    --iterations 2500 \
    --display-every 25
