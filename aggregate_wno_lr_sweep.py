"""Aggregates the wno_lr sweep: for each (task, wno_lr) cell, reports mean/std R2 and
MAE across seeds, for both Fourier-MIONet and U-WNO-MIONet, plus the easy/hard
permeability-quartile R2 gap (same methodology as analyze_heterogeneity_gap.py) --
so a single table shows whether any wno_lr value both (a) stabilizes WNO's
run-to-run variance and (b) narrows the heterogeneity-driven gap versus Fourier.

Reads per-cell JSONs produced by dp_outlier_diagnosis.py (dP) / sg_outlier_diagnosis.py
(sg), one per (task, wno_lr, seed). Expects filenames of the form
{task}_wnolr{lr}_seed{seed}.json (e.g. dp_wnolr5e-4_seed42.json,
sg_wnolr1e-4_seed43.json) in --results-dir -- matches the --state-dir naming convention
used by run_dp_wno_lr_sweep.sbatch / run_sg_wno_lr_sweep.sbatch.

Local, no GPU/cluster needed. Read-only.

Usage:
    python aggregate_wno_lr_sweep.py --results-dir sweep_results
"""
import argparse
import glob
import json
import os
import re

import numpy as np

CELL_RE = re.compile(r"^(dp|sg)_wnolr([0-9.e+-]+)_seed(\d+)\.json$")


def load_cells(results_dir):
    """Returns {(task, wno_lr): [ (seed, data_dict), ... ]}"""
    cells = {}
    for path in glob.glob(os.path.join(results_dir, "*.json")):
        m = CELL_RE.match(os.path.basename(path))
        if not m:
            continue
        task, wno_lr, seed = m.group(1), m.group(2), int(m.group(3))
        with open(path) as f:
            data = json.load(f)
        cells.setdefault((task, wno_lr), []).append((seed, data))
    return cells


def easy_hard_gap(r2, perm_std):
    n = len(perm_std)
    q = n // 4
    order = np.argsort(perm_std)
    hard_idx = order[-q:]
    easy_idx = order[:-q]
    r2 = np.asarray(r2)
    return r2[easy_idx].mean() - r2[hard_idx].mean()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="sweep_results")
    args = parser.parse_args()

    cells = load_cells(args.results_dir)
    if not cells:
        print(f"No cell JSONs matching {{dp,sg}}_wnolr<lr>_seed<seed>.json found in {args.results_dir}")
        return

    print(f"{'task':4} {'wno_lr':8} {'n_seeds':7} "
          f"{'fourier R2 mean±std':20} {'wno R2 mean±std':20} "
          f"{'fourier MAE mean±std':20} {'wno MAE mean±std':20} "
          f"{'fourier gap':12} {'wno gap':12}")

    for (task, wno_lr) in sorted(cells.keys()):
        seed_entries = cells[(task, wno_lr)]
        seeds = sorted(s for s, _ in seed_entries)

        fourier_r2 = np.array([d["fourier"]["r2"] for _, d in seed_entries]).mean(axis=1)
        wno_r2 = np.array([d["wno"]["r2"] for _, d in seed_entries]).mean(axis=1)
        fourier_mae = np.array([d["fourier"]["mae"] for _, d in seed_entries]).mean(axis=1)
        wno_mae = np.array([d["wno"]["mae"] for _, d in seed_entries]).mean(axis=1)

        fourier_gaps = []
        wno_gaps = []
        for _, d in seed_entries:
            perm_std = np.array(d["perm_std"])
            fourier_gaps.append(easy_hard_gap(d["fourier"]["r2"], perm_std))
            wno_gaps.append(easy_hard_gap(d["wno"]["r2"], perm_std))

        print(
            f"{task:4} {wno_lr:8} {len(seeds):7} "
            f"{fourier_r2.mean():8.3f}±{fourier_r2.std():<9.3f} "
            f"{wno_r2.mean():8.3f}±{wno_r2.std():<9.3f} "
            f"{fourier_mae.mean():8.4f}±{fourier_mae.std():<9.4f} "
            f"{wno_mae.mean():8.4f}±{wno_mae.std():<9.4f} "
            f"{np.mean(fourier_gaps):10.3f}  {np.mean(wno_gaps):10.3f}"
        )

    print(
        "\nReading the table: 'wno gap' < 'fourier gap' in a row is the signature the "
        "wavelet-localization hypothesis predicts (WNO degrading less from easy to hard "
        "cases). Low wno R2 std across seeds at a given wno_lr indicates that value "
        "stabilized WNO's run-to-run variance; compare against the default row "
        "(wno_lr=5e-4) to see if lowering it helped."
    )


if __name__ == "__main__":
    main()
