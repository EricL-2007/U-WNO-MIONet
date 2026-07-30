"""Inference-only per-case diagnostic for sg (gas saturation), mirroring
dp_outlier_diagnosis.py's structure so both tasks can be aggregated the same way across
the wno_lr sweep. Loads EXISTING checkpoints for both sg models (Fourier-MIONet,
U-WNO-MIONet) from a given --state-dir, runs inference on the full test set, and saves
per-case R2/MAE/L2-rel plus permeability-field stats (perm_max, perm_std) to JSON --
the same schema dp_outlier_diagnosis.py produces, so analyze_heterogeneity_gap.py and
aggregate_wno_lr_sweep.py can read either without modification.

Uses the now-fixed (elementwise, per row-and-column) mask -- same as the corrected
Rsquare_plume_tegother/MAE_plume in Fourier-UWNO-MIONet_sg.py -- so no separate
"corrected mask" comparison is needed here (unlike dp_outlier_diagnosis.py, which
predates the fix and validated it locally).

Does not train. Does not modify any checkpoint or production code.

Usage (from fourier-mionet-gcs/, with the project venv active, on a GPU node):
    python sg_outlier_diagnosis.py --state-dir pre_train_sweep/sg_wnolr5e-4_seed42 \
        --out-json sg_wnolr5e-4_seed42_results.json
"""
import argparse
import glob
import importlib.util
import json
import os

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CKPT_STEM = {"fourier": "sg_fourier_model.ckpt", "wno": "sg_wno_model.ckpt"}


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
    stem = CKPT_STEM[arch]
    best_path = os.path.join(ckpt_dir, f"{stem}-BEST.pt")
    if os.path.exists(best_path):
        return best_path
    candidates = [c for c in glob.glob(os.path.join(ckpt_dir, f"{stem}-*.pt")) if not c.endswith("-BEST.pt")]
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint found for '{arch}' in {ckpt_dir} (tried stem: {stem}). "
            f"Pass --{arch}-ckpt explicitly."
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


def per_case_metrics(y_true, y_pred):
    """Per-case R2/MAE/L2-rel using the FIXED elementwise (row+col) mask, matching the
    corrected production Rsquare_plume_tegother/MAE_plume in Fourier-UWNO-MIONet_sg.py."""
    ntest = y_true.shape[0]
    r2 = np.zeros(ntest)
    mae = np.zeros(ntest)
    l2_rel = np.zeros(ntest)
    for i in range(ntest):
        mask = ~np.isclose(y_true[i, -1], 0.0)
        y_true_i = y_true[i][:, mask]
        y_pred_i = y_pred[i][:, mask]
        sse = np.sum(np.square(y_true_i.flatten() - y_pred_i.flatten()))
        sst = np.sum(np.square(y_true_i.flatten() - np.mean(y_true_i.flatten())))
        r2[i] = 1 - sse / sst
        mae[i] = np.mean(np.abs(y_true_i.flatten() - y_pred_i.flatten()))

        flat_true = y_true[i].reshape(-1)
        flat_pred = y_pred[i].reshape(-1)
        denom = np.linalg.norm(flat_true)
        denom = denom if denom != 0 else 1e-12
        l2_rel[i] = np.linalg.norm(flat_true - flat_pred) / denom
    return {"r2": r2, "mae": mae, "l2_rel": l2_rel}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", required=True, help="e.g. pre_train_sweep/sg_wnolr5e-4_seed42")
    parser.add_argument("--fourier-ckpt", default=None)
    parser.add_argument("--wno-ckpt", default=None)
    parser.add_argument(
        "--module-path",
        default=os.path.join(PROJECT_ROOT, "Fourier-UWNO-MIONet_sg.py"),
    )
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    module = _load_module(args.module_path, "sg_module")
    print(f"[config] using {args.module_path} (NTRAIN={module.NTRAIN}, NTEST={module.NTEST})")

    fourier_ckpt = find_checkpoint(args.state_dir, args.fourier_ckpt, "fourier")
    wno_ckpt = find_checkpoint(args.state_dir, args.wno_ckpt, "wno")
    print(f"[checkpoints] fourier={fourier_ckpt}")
    print(f"[checkpoints] wno={wno_ckpt}")

    print("\nLoading + normalizing data (train-only-fit normalizers, no training)...")
    data_bundle = module.load_and_normalize_data()
    ntest = data_bundle["ntest"]

    y_true = data_bundle["y_test"].reshape(ntest, 24, 96, 200)
    x_test_field_raw = data_bundle["x_test_field_raw"]
    permeability = x_test_field_raw[..., 0]
    perm_max = permeability.reshape(ntest, -1).max(axis=1)
    perm_std = permeability.reshape(ntest, -1).std(axis=1)
    mio_raw = np.load(os.path.join(PROJECT_ROOT, "sg_test_a_MIO.npy"))[-ntest:, :]

    results_by_arch = {}
    for arch, ckpt_path in (("fourier", fourier_ckpt), ("wno", wno_ckpt)):
        print(f"\n{'=' * 70}\nRunning inference: {arch}\n{'=' * 70}")
        model = load_trained_model(module, arch, ckpt_path, data_bundle)
        y_pred = module.predict_in_chunks(
            model, data_bundle["x_test"], branch_chunk=module.BATCH_SIZE, time_chunk=module.TIMESTEP_BATCH_SIZE
        ).reshape(ntest, 24, 96, 200)

        metrics = per_case_metrics(y_true, y_pred)
        results_by_arch[arch] = metrics
        print(f"[{arch}] mean R2={metrics['r2'].mean():.4f} mean MAE={metrics['mae'].mean():.4f} "
              f"mean L2rel={metrics['l2_rel'].mean():.4f} max L2rel={metrics['l2_rel'].max():.4f}")

    out = {arch: {k: v.tolist() for k, v in m.items()} for arch, m in results_by_arch.items()}
    out["mio_raw"] = mio_raw.tolist()
    out["perm_max"] = perm_max.tolist()
    out["perm_std"] = perm_std.tolist()
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved full per-case results to {args.out_json}")


if __name__ == "__main__":
    main()
