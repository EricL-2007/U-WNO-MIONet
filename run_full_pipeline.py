"""Run the entire pipeline end-to-end: train both gas-saturation models (Fourier-MIONet
+ U-WNO-MIONet), train both pressure-buildup models, then generate every prediction
array and plot from the resulting checkpoints.

Each training script auto-detects GPU on its own (CUDA preferred on HPC clusters, MPS on
Apple Silicon) and only falls back to CPU if neither is available — see the preflight
check below for what it reports before running. Set FORCE_CPU=1 in the environment to
force CPU regardless of GPU availability (e.g. to work around the MPS backward-pass bug
documented in Fourier-UWNO-MIONet_sg.py/_dP.py's device-detection block).

Runs the two training scripts sequentially (not concurrently) to avoid contending for a
single GPU's memory; each internally trains its Fourier-MIONet baseline before its
U-WNO-MIONet model. If you're on a multi-GPU node and want them running concurrently
instead, ask for that — it's a small change to this file.

Usage:
    python run_full_pipeline.py
"""
import os
import subprocess
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

TRAINING_STAGES = [
    ("Gas saturation (sg): Fourier-MIONet + U-WNO-MIONet", "Fourier-UWNO-MIONet_sg.py"),
    ("Pressure buildup (dP): Fourier-MIONet + U-WNO-MIONet", "Fourier-UWNO-MIONet_dP.py"),
]

REQUIRED_FILES = [
    "Fourier-UWNO-MIONet_sg.py",
    "Fourier-UWNO-MIONet_dP.py",
    "generate_predictions_and_plots.py",
    "plot_fields.py",
    "plot_comparison.py",
    "wavelet_convolution.py",
    "sg_train_a.npz", "sg_train_u.npz", "sg_train_a_MIO.npy",
    "sg_test_a.npz", "sg_test_u.npz", "sg_test_a_MIO.npy",
    "dP_train_a.npz", "dP_train_u.npz", "dP_train_a_MIO.npy",
    "dP_test_a.npz", "dP_test_u.npz", "dP_test_a_MIO.npy",
]


def preflight_check():
    """Fail fast with a clear message rather than crashing hours into training."""
    print(f"{'=' * 70}\nPreflight check\n{'=' * 70}")

    missing = [f for f in REQUIRED_FILES if not os.path.exists(os.path.join(PROJECT_ROOT, f))]
    if not os.path.isdir(os.path.join(PROJECT_ROOT, "deepxde")):
        missing.append("deepxde/ (vendored local fork — the public PyPI deepxde package won't work)")
    if missing:
        print("Missing required files:")
        for m in missing:
            print(" -", m)
        sys.exit(1)

    try:
        import torch
    except ImportError:
        print("PyTorch is not installed in this environment.")
        sys.exit(1)

    for module_name, pip_name in [
        ("pytorch_wavelets", "pytorch_wavelets"),
        ("sklearn", "scikit-learn"),
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib"),
    ]:
        try:
            __import__(module_name)
        except ImportError:
            print(f"Missing dependency '{module_name}'. Install with: pip install {pip_name}")
            sys.exit(1)

    if os.environ.get("FORCE_CPU") == "1":
        print("GPU check: FORCE_CPU=1 is set — will run on CPU regardless of GPU availability.")
    elif torch.cuda.is_available():
        print(f"GPU check: CUDA available ({torch.cuda.get_device_name(0)}) — will use GPU.")
    elif torch.backends.mps.is_available():
        print("GPU check: Apple MPS available — will use GPU.")
    else:
        print("GPU check: no GPU detected — will run on CPU (slow for the WNO models).")

    print("All required files and dependencies found.\n")


def run_script(label, script_name):
    path = os.path.join(PROJECT_ROOT, script_name)
    print(f"\n{'=' * 70}\nSTAGE: {label}\nRunning: {script_name}\n{'=' * 70}\n", flush=True)
    start = time.time()
    result = subprocess.run([sys.executable, path], cwd=PROJECT_ROOT)
    elapsed = time.time() - start
    if result.returncode != 0:
        print(
            f"\n[WARNING] {script_name} exited with code {result.returncode} after "
            f"{elapsed / 60:.1f} min. Continuing to the next stage — "
            f"generate_predictions_and_plots.py degrades gracefully and will use whatever "
            f"checkpoints exist on disk."
        )
        return False
    print(f"\n{script_name} finished in {elapsed / 60:.1f} min.")
    return True


def main():
    print(f"{'=' * 70}\nFull pipeline: train sg + train dP + generate all predictions/plots\n{'=' * 70}\n")
    preflight_check()

    training_ok = {}
    for label, script_name in TRAINING_STAGES:
        training_ok[script_name] = run_script(label, script_name)

    print(f"\n{'=' * 70}\nSTAGE: Generating predictions + plots\nRunning: generate_predictions_and_plots.py\n{'=' * 70}\n")
    plot_result = subprocess.run(
        [sys.executable, os.path.join(PROJECT_ROOT, "generate_predictions_and_plots.py")],
        cwd=PROJECT_ROOT,
    )

    print(f"\n{'=' * 70}\nPipeline summary\n{'=' * 70}")
    for script_name, ok in training_ok.items():
        print(f"  {script_name}: {'OK' if ok else 'FAILED (see log above)'}")
    print(f"  generate_predictions_and_plots.py: {'OK' if plot_result.returncode == 0 else 'FAILED'}")
    print(
        "\nCheck predictions/ for saved y_true/y_pred/permeability arrays, and "
        "plots/field_plots/{sg,dP}/ + plots/comparison/{sg,dP}/ for every figure."
    )

    if plot_result.returncode != 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
