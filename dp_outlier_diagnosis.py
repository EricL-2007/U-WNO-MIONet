"""Inference-only diagnostic for the dP R2-vs-MAE puzzle (ntrain=400 -> 1000: MAE
improved but R2 got worse). Loads EXISTING best checkpoints for both dP models
(Fourier-MIONet, U-WNO-MIONet) at the ntrain=1000 scale, runs inference on the full
80-case test set, and reports:

  1. Per-case R2 and L2 relative error for both models (no retraining).
  2. The worst N cases by R2 (default N=10), with their scalar MIO features,
     permeability-field max/std, and column 6 (found to correlate r=-0.90 with
     OOD-row-count in the prior proxy-only analysis) -- checked for overlap against
     the highest-mask-leak cases.
  3. R2 recomputed with a locally-corrected out-of-domain mask (per row AND per
     column, not just column 0 broadcast across all 200 columns) to test whether the
     confirmed column-0-only mask leak (~4% avg / 23% max of padded cells scored as
     real predictions) is a meaningful chunk of the R2 regression or a red herring.
     This does NOT touch the production masking code in Fourier-UWNO-MIONet_dP*.py --
     the corrected mask is computed locally in this script only.

Does not train. Does not modify pre_train/ or any production script.

Usage (from fourier-mionet-gcs/, with the project venv active, on a GPU node):
    python dp_outlier_diagnosis.py --top-n 10
"""
import argparse
import glob
import importlib.util
import json
import os

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR_DEFAULT = os.path.join(PROJECT_ROOT, "pre_train")

# Candidate checkpoint filename stems, most-specific (intermediate, ntrain=1000 run)
# first -- Fourier-UWNO-MIONet_dP_intermediate.py writes "dP_intermediate_*_model.ckpt
# -BEST.pt"; Fourier-UWNO-MIONet_dP.py (ntrain=400 default) writes "dP_*_model.ckpt
# -BEST.pt". Try both since it's unclear from this machine which naming the actual
# Delta ntrain=1000 run used.
CKPT_STEM_CANDIDATES = {
    "fourier": ["dP_intermediate_fourier_model.ckpt", "dP_fourier_model.ckpt"],
    "wno": ["dP_intermediate_wno_model.ckpt", "dP_wno_model.ckpt"],
}


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_checkpoint(ckpt_dir, explicit_path, arch):
    if explicit_path:
        if not os.path.exists(explicit_path):
            raise FileNotFoundError(f"--{arch}-ckpt path does not exist: {explicit_path}")
        return explicit_path
    for stem in CKPT_STEM_CANDIDATES[arch]:
        best_path = os.path.join(ckpt_dir, f"{stem}-BEST.pt")
        if os.path.exists(best_path):
            return best_path
    # Fall back to the highest-numbered periodic snapshot for either naming.
    candidates = []
    for stem in CKPT_STEM_CANDIDATES[arch]:
        candidates.extend(glob.glob(os.path.join(ckpt_dir, f"{stem}-*.pt")))
    candidates = [c for c in candidates if not c.endswith("-BEST.pt")]
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint found for '{arch}' in {ckpt_dir} "
            f"(tried stems: {CKPT_STEM_CANDIDATES[arch]}). Pass --{arch}-ckpt explicitly."
        )

    def step_of(p):
        try:
            return int(p.rsplit("-", 1)[1].split(".")[0])
        except (IndexError, ValueError):
            return -1

    candidates.sort(key=step_of)
    print(f"  WARNING: no -BEST.pt found for '{arch}'; using highest periodic snapshot: {candidates[-1]}")
    return candidates[-1]


def load_trained_model(module, arch, ckpt_path, data_bundle):
    decoder_builder = module._build_fourier_decoder if arch == "fourier" else module._build_wavelet_decoder
    output_transform = module.make_output_transform(data_bundle["mean"], data_bundle["std"])

    module.reset_seed(module.SEED)
    net = module.build_net(decoder_builder, output_transform)
    data = module.dde.data.Quadruple(
        data_bundle["x_train"], data_bundle["y_train"], data_bundle["x_test"], data_bundle["y_test"]
    )
    model = module.dde.Model(data, net)
    model.compile(
        "adam",
        lr=module.LR,
        loss="mean l2 relative error",
        decay=("step", module.DECAY_STEP, module.DECAY_GAMMA),
        metrics=["mean l2 relative error", module.Rsquare_plume_tegother, module.MAE_plume],
    )
    model.restore(ckpt_path, verbose=1)
    return model


# ============================================================
# Per-case metrics (same math as the production Rsquare_plume_tegother / MAE_plume /
# max_l2_relative_error, but returning one value per test case instead of an
# already-averaged scalar).
# ============================================================
def per_case_metrics(y_true, y_pred, sentinel):
    """y_true, y_pred: (ntest, 24, 96, 200). Returns dict of per-case arrays using the
    PRODUCTION mask (column-0-only, broadcast across all 200 columns)."""
    ntest = y_true.shape[0]
    r2 = np.zeros(ntest)
    mae = np.zeros(ntest)
    l2_rel = np.zeros(ntest)
    leak_frac = np.zeros(ntest)
    ood_rowcount = np.zeros(ntest)

    for i in range(ntest):
        z_axis = y_true[i, -1, :, 0]
        mask_row = ~np.isclose(z_axis, sentinel, atol=1e-3)
        ood_rowcount[i] = (~mask_row).sum()

        y_true_i = y_true[i][:, mask_row, :]
        y_pred_i = y_pred[i][:, mask_row, :]
        sse = np.sum(np.square(y_true_i.flatten() - y_pred_i.flatten()))
        sst = np.sum(np.square(y_true_i.flatten() - np.mean(y_true_i.flatten())))
        r2[i] = 1 - sse / sst
        mae[i] = np.mean(np.abs(y_true_i.flatten() - y_pred_i.flatten()))

        flat_true = y_true[i].reshape(-1)
        flat_pred = y_pred[i].reshape(-1)
        denom = np.linalg.norm(flat_true)
        denom = denom if denom != 0 else 1e-12
        l2_rel[i] = np.linalg.norm(flat_true - flat_pred) / denom

        # leak diagnostic: within rows the mask calls "valid", what fraction of
        # (row, col) cells at the final timestep are actually still exactly the
        # sentinel (i.e. genuinely out-of-domain, leaked past the column-0 proxy)?
        last_t = y_true[i, -1]  # (96, 200)
        valid_rows = np.where(mask_row)[0]
        if valid_rows.size:
            leaked = np.isclose(last_t[valid_rows, :], sentinel, atol=1e-3).sum()
            leak_frac[i] = leaked / (96 * 200)

    return {
        "r2": r2,
        "mae": mae,
        "l2_rel": l2_rel,
        "leak_frac": leak_frac,
        "ood_rowcount": ood_rowcount,
    }


def per_case_r2_corrected_mask(y_true, y_pred, sentinel):
    """Same R2, but with the mask built PER ROW AND PER COLUMN at the final timestep
    (elementwise, not the column-0-only proxy broadcast across all 200 columns). Local
    to this script -- does not touch production masking code."""
    ntest = y_true.shape[0]
    r2 = np.zeros(ntest)
    for i in range(ntest):
        mask2d = ~np.isclose(y_true[i, -1], sentinel, atol=1e-3)  # (96, 200), elementwise
        mask3d = np.broadcast_to(mask2d, y_true[i].shape)  # (24, 96, 200)

        y_true_i = y_true[i][mask3d]
        y_pred_i = y_pred[i][mask3d]
        sse = np.sum(np.square(y_true_i - y_pred_i))
        sst = np.sum(np.square(y_true_i - np.mean(y_true_i)))
        r2[i] = 1 - sse / sst
    return r2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-dir", default=CKPT_DIR_DEFAULT)
    parser.add_argument("--fourier-ckpt", default=None, help="Explicit path, overrides auto-detection.")
    parser.add_argument("--wno-ckpt", default=None, help="Explicit path, overrides auto-detection.")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument(
        "--module-path",
        default=os.path.join(PROJECT_ROOT, "Fourier-UWNO-MIONet_dP_intermediate.py"),
        help="Which training script to import model-building/data-loading code from "
        "(its module-level NTRAIN/NTEST control the data scale used here).",
    )
    # No default on purpose: a shared default filename is exactly what caused the
    # sweep's comparison_log_*.csv collision (same root cause -- a hardcoded/shared
    # output path across repeated invocations silently overwrites prior runs' results).
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    module = _load_module(args.module_path, "dp_intermediate_module")
    print(f"[config] using {args.module_path} (NTRAIN={module.NTRAIN}, NTEST={module.NTEST}, "
          f"DP_WNO_LAYERS={module.DP_WNO_LAYERS}, DP_WNO_WIDTH={module.DP_WNO_WIDTH})")

    fourier_ckpt = find_checkpoint(args.ckpt_dir, args.fourier_ckpt, "fourier")
    wno_ckpt = find_checkpoint(args.ckpt_dir, args.wno_ckpt, "wno")
    print(f"[checkpoints] fourier={fourier_ckpt}")
    print(f"[checkpoints] wno={wno_ckpt}")

    print("\nLoading + normalizing data (train-only-fit normalizers, no training)...")
    data_bundle = module.load_and_normalize_data()
    ntest = data_bundle["ntest"]
    sentinel = module.OUT_OF_DOMAIN_SENTINEL

    y_true = data_bundle["y_test"].reshape(ntest, 24, 96, 200)

    # scalar MIO features + permeability field, unnormalized, for the worst-case report
    x_test_field_raw = data_bundle["x_test_field_raw"]  # (ntest, 96, 200, 5) post field_input selection
    permeability = x_test_field_raw[..., 0]  # channel 0, consistent with generate_predictions_and_plots.py
    perm_max = permeability.reshape(ntest, -1).max(axis=1)
    perm_std = permeability.reshape(ntest, -1).std(axis=1)

    # dP_test_a_MIO.npy, last `ntest` rows, matching get_data()'s -ntest: slice
    mio_raw = np.load(os.path.join(PROJECT_ROOT, "dP_test_a_MIO.npy"))[-ntest:, :]

    results_by_arch = {}
    for arch, ckpt_path in (("fourier", fourier_ckpt), ("wno", wno_ckpt)):
        print(f"\n{'=' * 70}\nRunning inference: {arch}\n{'=' * 70}")
        model = load_trained_model(module, arch, ckpt_path, data_bundle)
        y_pred = module.predict_in_chunks(
            model, data_bundle["x_test"], branch_chunk=module.BATCH_SIZE, time_chunk=module.TIMESTEP_BATCH_SIZE
        ).reshape(ntest, 24, 96, 200)

        metrics = per_case_metrics(y_true, y_pred, sentinel)
        metrics["r2_corrected_mask"] = per_case_r2_corrected_mask(y_true, y_pred, sentinel)
        results_by_arch[arch] = metrics

        print(f"[{arch}] mean R2 (production mask) = {metrics['r2'].mean():.4f}")
        print(f"[{arch}] mean R2 (corrected mask)   = {metrics['r2_corrected_mask'].mean():.4f}")
        print(f"[{arch}] mean MAE (production mask) = {metrics['mae'].mean():.4f}")
        print(f"[{arch}] mean L2 rel = {metrics['l2_rel'].mean():.4f} | max L2 rel = {metrics['l2_rel'].max():.4f}")

    # ============================================================
    # Worst-N-by-R2 report + mask-leak overlap check
    # ============================================================
    top_n = args.top_n
    for arch in ("fourier", "wno"):
        m = results_by_arch[arch]
        worst_idx = np.argsort(m["r2"])[:top_n]  # lowest R2 = worst
        leak_top_idx = set(np.argsort(m["leak_frac"])[-top_n:].tolist())
        overlap = set(worst_idx.tolist()) & leak_top_idx

        print(f"\n{'=' * 70}\n{arch}: worst {top_n} cases by R2 (production mask)\n{'=' * 70}")
        print(f"Overlap with top-{top_n} highest-mask-leak cases: {len(overlap)}/{top_n} -> {sorted(overlap)}")
        print(
            f"{'idx':>4} {'R2':>8} {'R2_corr':>8} {'MAE':>7} {'L2rel':>7} "
            f"{'leak%':>7} {'OODrows':>7} {'perm_max':>9} {'perm_std':>9} {'col6':>8}"
        )
        for i in worst_idx:
            print(
                f"{i:4d} {m['r2'][i]:8.3f} {m['r2_corrected_mask'][i]:8.3f} {m['mae'][i]:7.3f} "
                f"{m['l2_rel'][i]:7.3f} {100*m['leak_frac'][i]:7.2f} {int(m['ood_rowcount'][i]):7d} "
                f"{perm_max[i]:9.4f} {perm_std[i]:9.4f} {mio_raw[i, 6]:8.2f}"
            )
            print(f"      MIO (all 7 cols): {mio_raw[i].tolist()}")

    # ============================================================
    # Does fixing the mask meaningfully change R2?
    # ============================================================
    print(f"\n{'=' * 70}\nCorrected-mask R2 vs production-mask R2 (all {ntest} cases)\n{'=' * 70}")
    for arch in ("fourier", "wno"):
        m = results_by_arch[arch]
        r2_prod_mean = m["r2"].mean()
        r2_corr_mean = m["r2_corrected_mask"].mean()
        delta = r2_corr_mean - r2_prod_mean
        print(
            f"[{arch}] production mask mean R2 = {r2_prod_mean:.4f} | "
            f"corrected mask mean R2 = {r2_corr_mean:.4f} | delta = {delta:+.4f}"
        )
        per_case_delta = m["r2_corrected_mask"] - m["r2"]
        print(
            f"[{arch}] per-case R2 delta (corrected - production): "
            f"mean={per_case_delta.mean():+.4f} max={per_case_delta.max():+.4f} min={per_case_delta.min():+.4f}"
        )

    # ============================================================
    # Save raw per-case arrays for further analysis
    # ============================================================
    out = {
        arch: {k: v.tolist() for k, v in m.items()}
        for arch, m in results_by_arch.items()
    }
    out["mio_raw"] = mio_raw.tolist()
    out["perm_max"] = perm_max.tolist()
    out["perm_std"] = perm_std.tolist()
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved full per-case results to {args.out_json}")


if __name__ == "__main__":
    main()
