#!/bin/bash
#SBATCH --job-name=uwno-mionet-sg-scaleup
#SBATCH --account=bhdh-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --gpus=1
#SBATCH --output=%x_%j.out

set -e  # fail fast — see wavelet import sanity check below

cd /scratch/bhdh/eliu7/U-WNO-MIONet

module purge
source venv/bin/activate

# Catch a broken wavelet_convolution.py import immediately, not hours into the
# Fourier-MIONet phase before U-WNO-MIONet even starts.
python3 -c "from wavelet_convolution import DWT, IDWT; print('wavelet imports OK')"

echo "Host: $(hostname)"
nvidia-smi

# 48h partition cap: if this run gets preempted/times out, re-submitting this exact
# sbatch script picks back up automatically -- Fourier-UWNO-MIONet_sg.py checks
# pre_train/sg_{fourier,wno}_train_state.json on startup and resumes from the last
# saved step (weights + optimizer + LR-scheduler state + EarlyStopping counters) rather
# than restarting either model from scratch. No special flag needed; just resubmit.
python -u Fourier-UWNO-MIONet_sg.py \
    --ntrain 2000 \
    --ntest 222 \
    --batch-size 20 \
    --timestep-batch-size 8 \
    --iterations 7500 \
    --display-every 100
