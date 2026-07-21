#!/bin/bash
#SBATCH --job-name=uwno-mionet-dp-only
#SBATCH --account=bhdh-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --gpus=1
#SBATCH --output=%x_%j.out

set -e  # fail fast — see wavelet import sanity check below

cd /scratch/bhdh/eliu7/U-WNO-MIONet

module purge
source venv/bin/activate

# Catch a broken wavelet_convolution.py import immediately, not partway into the
# dP U-WNO-MIONet phase.
python3 -c "from wavelet_convolution import DWT, IDWT; print('wavelet imports OK')"

echo "Host: $(hostname)"
nvidia-smi

python -u run_dp_only.py
