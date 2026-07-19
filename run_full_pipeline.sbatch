#!/bin/bash
#SBATCH --job-name=uwno-mionet-full-pipeline
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

# Catch a broken wavelet_convolution.py import immediately, not an hour into the
# job after the Fourier-MIONet phase already completed (hit this failure mode
# multiple times this session — it only surfaces once the U-WNO-MIONet phase starts).
python3 -c "from wavelet_convolution import DWT, IDWT; print('wavelet imports OK')"

echo "Host: $(hostname)"
nvidia-smi

python -u run_full_pipeline.py
