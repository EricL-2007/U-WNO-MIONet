"""Reads all per-config *_train_state.json files under a sweep's state-dir root and
summarizes each config's final training outcome -- a fast, no-GPU sanity check of
where all 16 wno_lr sweep configs landed, independent of the comparison_log CSV
collision (train_state.json was always per-config-safe; see aggregate_wno_lr_sweep.py
and its accompanying report for the full story).

NOTE: metrics_test entries are [mean_l2_rel, R2, MAE] aggregated over the WHOLE test
set at training time (from model.compile's metrics=[...] order) -- these are NOT
per-test-case values, so this does not give you the easy/hard permeability-quartile
split (that still requires running dp_outlier_diagnosis.py / sg_outlier_diagnosis.py
against each config's checkpoint). This script answers "what was each config's final
aggregate R2/MAE and how many steps did it actually run" -- a quick cross-check before
committing to the full per-case inference pass.

Read-only. Does not touch any checkpoint or state file.

Usage:
    python summarize_sweep_train_state.py --sweep-root pre_train_sweep \
        --out-csv sweep_train_state_summary.csv --out-json sweep_train_state_summary.json
"""
import argparse
import csv
import glob
import json
import os
import re

CONFIG_RE = re.compile(r"^(dp|sg)_wnolr([0-9.e+-]+)_seed(\d+)$")
ARCH_FILE_TAG = {
    "dp": {"fourier": "dP_intermediate_fourier_train_state.json", "wno": "dP_intermediate_wno_train_state.json"},
    "sg": {"fourier": "sg_fourier_train_state.json", "wno": "sg_wno_train_state.json"},
}


def summarize_one(json_path):
    if not os.path.exists(json_path):
        return None
    with open(json_path) as f:
        state = json.load(f)
    lh = state.get("losshistory", {})
    metrics_test = lh.get("metrics_test", [])
    steps = lh.get("steps", [])

    # metrics_test[i] corresponds to steps[i]; find the row at best_step if present,
    # else fall back to the last recorded row.
    best_step = state.get("best_step")
    row = None
    if best_step is not None and best_step in steps:
        row = metrics_test[steps.index(best_step)]
    elif metrics_test:
        row = metrics_test[-1]

    mean_l2_rel, r2, mae = (row + [None, None, None])[:3] if row else (None, None, None)

    return {
        "complete": state.get("complete"),
        "final_step": state.get("step"),
        "best_step": best_step,
        "best_loss_train": state.get("best_loss_train"),
        "best_loss_test": state.get("best_loss_test"),
        "n_recorded_checkpoints": len(steps),
        "mean_l2_rel_at_best": mean_l2_rel,
        "r2_at_best": r2,
        "mae_at_best": mae,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-root", default="pre_train_sweep")
    parser.add_argument("--out-csv", default="sweep_train_state_summary.csv")
    parser.add_argument("--out-json", default="sweep_train_state_summary.json")
    args = parser.parse_args()

    config_dirs = sorted(
        d for d in glob.glob(os.path.join(args.sweep_root, "*")) if os.path.isdir(d)
    )
    if not config_dirs:
        print(f"No config directories found under {args.sweep_root}")
        return

    rows = []
    for config_dir in config_dirs:
        name = os.path.basename(config_dir)
        m = CONFIG_RE.match(name)
        if not m:
            print(f"  skipping unrecognized directory name: {name}")
            continue
        task, wno_lr, seed = m.group(1), m.group(2), m.group(3)

        for arch, filename in ARCH_FILE_TAG[task].items():
            json_path = os.path.join(config_dir, filename)
            summary = summarize_one(json_path)
            if summary is None:
                print(f"  MISSING: {json_path}")
                continue
            row = {"task": task, "wno_lr": wno_lr, "seed": seed, "arch": arch, "config_dir": name}
            row.update(summary)
            rows.append(row)
            print(
                f"[{task}/{arch}] wno_lr={wno_lr} seed={seed}: complete={row['complete']} "
                f"final_step={row['final_step']} best_step={row['best_step']} "
                f"R2={row['r2_at_best']} MAE={row['mae_at_best']}"
            )

    if not rows:
        print("No train_state.json files found/parsed.")
        return

    with open(args.out_json, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved JSON summary to {args.out_json}")

    fieldnames = list(rows[0].keys())
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved CSV summary to {args.out_csv}")


if __name__ == "__main__":
    main()
