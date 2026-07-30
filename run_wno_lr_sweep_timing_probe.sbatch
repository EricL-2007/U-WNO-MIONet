#!/bin/bash
#SBATCH --job-name=uwno-mionet-timing-probe
#SBATCH --account=bhdh-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --gpus=1
#SBATCH --output=%x_%j.out

# Timing-only probe for the wno_lr sweep GPU-hour projection -- NOT part of the real
# sweep. Runs both dP and sg for just enough iterations for each script's built-in
# EarlyTimingProbe callback (probe_steps=20) to print a measured s/step for BOTH
# Fourier-MIONet and U-WNO-MIONet, at the mask-fixed code, so the sweep's GPU-hour
# projection is based on a real number instead of a guess.
#
# Uses --state-dir pre_train_probe/{dp,sg} so it cannot collide with or corrupt any
# real training state. Safe to delete pre_train_probe/ afterward.

set -e

cd /scratch/bhdh/eliu7/U-WNO-MIONet

module purge
source venv/bin/activate

python3 -c "from wavelet_convolution import DWT, IDWT; print('wavelet imports OK')"

echo "Host: $(hostname)"
nvidia-smi

echo "=== dP timing probe (Fourier-MIONet + U-WNO-MIONet, ~60 iterations) ==="
DP_WNO_LAYERS="${DP_WNO_LAYERS:-1}" \
DP_WNO_WIDTH="${DP_WNO_WIDTH:-36}" \
python -u Fourier-UWNO-MIONet_dP_intermediate.py \
    --ntrain 1000 \
    --ntest 80 \
    --batch-size 10 \
    --timestep-batch-size 8 \
    --iterations 60 \
    --display-every 10 \
    --state-dir pre_train_probe/dp \
    --disable-wno-divergence-guard

echo "=== sg timing probe (Fourier-MIONet + U-WNO-MIONet, ~60 iterations) ==="
python -u Fourier-UWNO-MIONet_sg.py \
    --iterations 60 \
    --display-every 10 \
    --state-dir pre_train_probe/sg \
    --disable-wno-divergence-guard

echo "=== Probe done. Grep the two '[timing] ... s/step' lines above per model. ==="
