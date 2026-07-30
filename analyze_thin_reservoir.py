"""Local, no-GPU follow-up analysis for the thin-reservoir hypothesis. Reads the
per-case JSON produced by dp_outlier_diagnosis.py (r2, ood_rowcount, etc. per
architecture) and reports:

  1. Pearson correlation between valid reservoir thickness (96 - OODrows) and
     per-case R2, for both models, plus an R2-by-thickness-quartile breakdown.
  2. What fraction of the 80-case test set falls into the thin-reservoir regime,
     and mean/median R2 on the remaining ("normal") cases only.
  3. If a second JSON (e.g. from an ntrain=400 run) is passed via --other-json,
     the same thickness/R2 correlation for that run, for direct comparison.

Does not train, does not touch checkpoints or production code -- reads only the
JSON(s) already produced by dp_outlier_diagnosis.py.

Usage:
    python analyze_thin_reservoir.py dp_outlier_diagnosis_results.json \
        [--other-json dp_outlier_diagnosis_results_ntrain400.json] \
        [--thin-threshold 45]
"""
import argparse
import json

import numpy as np


def load(path):
    with open(path) as f:
        return json.load(f)


def report_one(data, label, thin_threshold):
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
    for arch in ("fourier", "wno"):
        if arch not in data:
            continue
        r2 = np.array(data[arch]["r2"])
        ood_rowcount = np.array(data[arch]["ood_rowcount"])
        thickness = 96 - ood_rowcount
        n = len(r2)

        corr = np.corrcoef(thickness, r2)[0, 1]
        print(f"\n[{arch}] n={n} | corr(thickness, R2) = {corr:.4f}")

        # quartile breakdown by thickness
        qs = np.percentile(thickness, [0, 25, 50, 75, 100])
        print(f"[{arch}] thickness quartile edges: {qs}")
        order = np.argsort(thickness)
        quart_size = n // 4
        for q in range(4):
            lo = q * quart_size
            hi = (q + 1) * quart_size if q < 3 else n
            idx = order[lo:hi]
            print(
                f"[{arch}] thickness Q{q+1} (rows {int(thickness[idx].min())}-{int(thickness[idx].max())}, "
                f"n={len(idx)}): mean R2={r2[idx].mean():.3f} median R2={np.median(r2[idx]):.3f}"
            )

        # thin-reservoir fraction + normal-case R2
        thin_mask = thickness < thin_threshold
        frac_thin = thin_mask.mean()
        print(f"\n[{arch}] thin-reservoir fraction (thickness < {thin_threshold}): {frac_thin:.3f} "
              f"({thin_mask.sum()}/{n} cases)")
        print(f"[{arch}] mean R2 on thin subset:   {r2[thin_mask].mean():.4f} "
              f"(median {np.median(r2[thin_mask]):.4f})" if thin_mask.any() else f"[{arch}] no thin cases")
        normal_mask = ~thin_mask
        print(f"[{arch}] mean R2 on REMAINING (normal) cases: {r2[normal_mask].mean():.4f} "
              f"(median {np.median(r2[normal_mask]):.4f}, n={normal_mask.sum()})")
        print(f"[{arch}] overall mean R2 (all {n} cases): {r2.mean():.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_path")
    parser.add_argument("--other-json", default=None, help="e.g. an ntrain=400 results JSON for comparison")
    parser.add_argument("--thin-threshold", type=int, default=45,
                         help="valid-thickness (grid rows) cutoff below which a case is 'thin'")
    args = parser.parse_args()

    data = load(args.json_path)
    report_one(data, f"ntrain=1000 (primary): {args.json_path}", args.thin_threshold)

    if args.other_json:
        other = load(args.other_json)
        report_one(other, f"comparison run: {args.other_json}", args.thin_threshold)


if __name__ == "__main__":
    main()
