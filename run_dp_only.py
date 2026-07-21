"""Run ONLY the pressure-buildup (dP) half of the pipeline: train dP's Fourier-MIONet +
U-WNO-MIONet, then generate predictions/plots.

Time-sensitive re-run after the LR-decay/checkpoint fixes (commit 379fa53, confirmed
working in b79bb89) — the dP plots/predictions checked into the repo predate that fix
(training only reached step 150/1200, see predictions_from_delta/meta.json) and need to
be regenerated. sg is already correct and is intentionally left untouched.

Reuses preflight_check()/run_script() from run_full_pipeline.py rather than duplicating
them.

Note: generate_predictions_and_plots.py scans pre_train/ for whatever checkpoints exist
and has no sg/dP selection flag, so if sg_fourier/sg_wno checkpoints are still on disk
from a prior run, it will also re-run sg inference + replot sg alongside dP. That's
read-only inference against sg's already-correct BEST checkpoints (no retraining), so
it's a minor time cost, not a correctness risk — sg's checkpoints/outputs are not
modified by this script.

Usage:
    python run_dp_only.py
"""
import os
import subprocess
import sys

import run_full_pipeline as pipeline

PROJECT_ROOT = pipeline.PROJECT_ROOT


def main():
    print(f"{'=' * 70}\ndP-only re-run: train dP + generate predictions/plots (sg untouched)\n{'=' * 70}\n")
    pipeline.preflight_check()

    dp_ok = pipeline.run_script(
        "Pressure buildup (dP): Fourier-MIONet + U-WNO-MIONet", "Fourier-UWNO-MIONet_dP.py"
    )

    print(f"\n{'=' * 70}\nSTAGE: Generating predictions + plots\nRunning: generate_predictions_and_plots.py\n{'=' * 70}\n")
    plot_result = subprocess.run(
        [sys.executable, os.path.join(PROJECT_ROOT, "generate_predictions_and_plots.py")],
        cwd=PROJECT_ROOT,
    )

    print(f"\n{'=' * 70}\ndP-only re-run summary\n{'=' * 70}")
    print(f"  Fourier-UWNO-MIONet_dP.py: {'OK' if dp_ok else 'FAILED (see log above)'}")
    print(f"  generate_predictions_and_plots.py: {'OK' if plot_result.returncode == 0 else 'FAILED'}")
    print(
        "\nNote: generate_predictions_and_plots.py also reprocesses sg if sg checkpoints "
        "are present on disk (inference-only against sg's existing BEST checkpoints, no "
        "retraining, sg files/outputs untouched)."
    )
    print(
        "\nCheck predictions/ for updated dP y_true/y_pred/permeability arrays, and "
        "plots/field_plots/dP/ + plots/comparison/dP/ for the regenerated figures."
    )

    if not dp_ok or plot_result.returncode != 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
