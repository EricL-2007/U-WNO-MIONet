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
# DP_WNO_LAYERS=1, DP_WNO_WIDTH=36 are the CONFIRMED-best capacity config from the
# completed layers/width A/B tests (ntrain=400 reduced scale):
#   layers=4, width=36 (original): 8.39M params, R2=-42.16, MAE=0.727
#   layers=2, width=36:            4.20M params, R2=-2.99,  MAE=0.638
#   layers=1, width=36:            2.10M params, R2=-3.00,  MAE=0.627  <- best MAE
#   layers=1, width=18:            0.53M params, R2=-3.18,  MAE=0.631
# layers=1/width=36 wins on MAE with R2 tied against width=18 -- width reduction below
# 36 showed no benefit, so capacity tuning is done. These are locked-in defaults now,
# not placeholders; no --export override needed to get the best-known config.
DP_WNO_LAYERS="${DP_WNO_LAYERS:-1}" \
DP_WNO_WIDTH="${DP_WNO_WIDTH:-36}" \
python -u Fourier-UWNO-MIONet_dP_intermediate.py \
    --ntrain 1000 \
    --ntest 80 \
    --batch-size 10 \
    --timestep-batch-size 8 \
    --iterations 2500 \
    --display-every 25
