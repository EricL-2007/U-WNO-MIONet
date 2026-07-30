"""Tests the project's central hypothesis: does U-WNO-MIONet's wavelet localization
narrow the R2/MAE gap (vs Fourier-MIONet) specifically on the high-permeability-
contrast subset, for both dP and sg?

Inputs (all local, no cluster/GPU needed):
  - dP: dp_outlier_diagnosis_results.json (from dp_outlier_diagnosis.py), which
    already carries per-case r2/mae/perm_std/perm_max for both architectures.
  - sg: predictions/{y_true_sg,y_pred_fmionet_sg,y_pred_uwno_sg,permeability_test_sg}.npy
    (already-correct, previously-generated sg predictions -- read-only, sg is not
    retrained or otherwise touched here).

For each task, splits the 80 test cases into the top permeability-std quartile
("hard") vs the remaining 3 quartiles ("easy"), reports R2/MAE per model per subset,
and the R2 gap (easy - hard) per model -- a smaller gap for WNO than Fourier is the
signature the wavelet-localization hypothesis predicts.

Read-only. No retraining, no fixes.
"""
import argparse
import json

import numpy as np


def sg_rsquare_mae_per_case(y_true, y_pred):
    """Per-case version of Fourier-UWNO-MIONet_sg.py's Rsquare_plume_tegother/MAE_plume
    (col0/last-timestep mask against exact 0.0, same as production)."""
    size = y_true.shape[0]
    r2 = np.zeros(size)
    mae = np.zeros(size)
    for i in range(size):
        z_axis = y_true[i, -1, :, 0]
        mask = ~np.isclose(z_axis, 0.0)
        y_true_i = y_true[i][:, mask, :]
        y_pred_i = y_pred[i][:, mask, :]
        sse = np.sum(np.square(y_true_i.flatten() - y_pred_i.flatten()))
        sst = np.sum(np.square(y_true_i.flatten() - np.mean(y_true_i.flatten())))
        r2[i] = 1 - sse / sst
        mae[i] = np.mean(np.abs(y_true_i.flatten() - y_pred_i.flatten()))
    return r2, mae


def split_report(task_label, perm_std, metrics_by_model):
    """metrics_by_model: {model_name: {"r2": arr, "mae": arr}}"""
    n = len(perm_std)
    q = n // 4
    order = np.argsort(perm_std)
    hard_idx = order[-q:]  # top quartile by perm_std
    easy_idx = order[:-q]  # remaining 3 quartiles

    print(f"\n{'=' * 70}\n{task_label}: hard (top perm_std quartile, n={len(hard_idx)}) "
          f"vs easy (remaining, n={len(easy_idx)})\n{'=' * 70}")
    print(f"perm_std range -- hard: {perm_std[hard_idx].min():.4f}-{perm_std[hard_idx].max():.4f} "
          f"| easy: {perm_std[easy_idx].min():.4f}-{perm_std[easy_idx].max():.4f}")

    gaps = {}
    for model, m in metrics_by_model.items():
        r2, mae = m["r2"], m["mae"]
        r2_hard, r2_easy = r2[hard_idx].mean(), r2[easy_idx].mean()
        mae_hard, mae_easy = mae[hard_idx].mean(), mae[easy_idx].mean()
        gap = r2_easy - r2_hard  # positive = R2 drops going into the hard subset
        gaps[model] = gap
        print(f"[{model}] easy: R2={r2_easy:8.3f} MAE={mae_easy:7.4f} | "
              f"hard: R2={r2_hard:8.3f} MAE={mae_hard:7.4f} | R2 gap (easy-hard) = {gap:8.3f}")

    if len(gaps) == 2:
        models = list(gaps.keys())
        smaller = models[0] if gaps[models[0]] < gaps[models[1]] else models[1]
        print(f"\n[{task_label}] smaller R2 gap: {smaller} "
              f"({gaps[models[0]]:.3f} vs {gaps[models[1]]:.3f}) "
              f"-> {'supports' if smaller == 'wno' else 'does NOT support'} the wavelet-localization hypothesis")
    return gaps


def do_dp(json_path):
    with open(json_path) as f:
        d = json.load(f)
    perm_std = np.array(d["perm_std"])
    metrics = {}
    for arch, label in (("fourier", "fourier"), ("wno", "wno")):
        metrics[label] = {"r2": np.array(d[arch]["r2"]), "mae": np.array(d[arch]["mae"])}
    return split_report("dP", perm_std, metrics)


def do_sg(pred_dir):
    y_true = np.load(f"{pred_dir}/y_true_sg.npy")
    y_pred_f = np.load(f"{pred_dir}/y_pred_fmionet_sg.npy")
    y_pred_w = np.load(f"{pred_dir}/y_pred_uwno_sg.npy")
    perm = np.load(f"{pred_dir}/permeability_test_sg.npy")  # (n, 96, 200)
    perm_std = perm.reshape(perm.shape[0], -1).std(axis=1)

    r2_f, mae_f = sg_rsquare_mae_per_case(y_true, y_pred_f)
    r2_w, mae_w = sg_rsquare_mae_per_case(y_true, y_pred_w)
    metrics = {
        "fourier": {"r2": r2_f, "mae": mae_f},
        "wno": {"r2": r2_w, "mae": mae_w},
    }
    print(f"\n[sg] overall mean R2 -- fourier={r2_f.mean():.4f} wno={r2_w.mean():.4f} "
          f"| overall mean MAE -- fourier={mae_f.mean():.4f} wno={mae_w.mean():.4f}")
    return split_report("sg", perm_std, metrics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dp-json", default="dp_outlier_diagnosis_results.json")
    parser.add_argument("--dp-json-ntrain400", default=None,
                         help="If available, the ntrain=400 dP results JSON for the same split.")
    parser.add_argument("--sg-pred-dir", default="predictions")
    args = parser.parse_args()

    do_dp(args.dp_json)

    if args.dp_json_ntrain400:
        print(f"\n\n{'#' * 70}\n# ntrain=400 comparison\n{'#' * 70}")
        do_dp(args.dp_json_ntrain400)
    else:
        print(f"\n\n[ntrain=400 comparison] SKIPPED -- no --dp-json-ntrain400 provided.")

    do_sg(args.sg_pred_dir)


if __name__ == "__main__":
    main()
