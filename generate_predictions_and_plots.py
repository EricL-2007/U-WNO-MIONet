"""Generate predictions from EXISTING checkpoints and produce the full set of plots.

Does NOT train anything and does NOT require the full 1200-iteration run to have
finished — it works against whatever sg_fourier_model.ckpt-*.pt / sg_wno_model.ckpt-*.pt
(and dP_* equivalents, if present) are already on disk in pre_train/, picking the
highest-step checkpoint available for each model.

Run with:  python generate_predictions_and_plots.py
"""
import glob
import importlib.util
import json
import os
import re

import numpy as np

import plot_comparison
import plot_fields

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(PROJECT_ROOT, "pre_train")
PRED_DIR = os.path.join(PROJECT_ROOT, "predictions")
PLOTS_FIELD_DIR = os.path.join(PROJECT_ROOT, "plots", "field_plots")
PLOTS_COMPARISON_DIR = os.path.join(PROJECT_ROOT, "plots", "comparison")
COMPARISON_CSV_SG = os.path.join(PROJECT_ROOT, "comparison_log.csv")
COMPARISON_CSV_DP = os.path.join(PROJECT_ROOT, "comparison_log_dp.csv")
META_PATH = os.path.join(PRED_DIR, "meta.json")

STEP_RE = re.compile(r"-(\d+)\.pt$")

# (task, arch) -> checkpoint glob pattern, matching what Fourier-UWNO-MIONet_sg.py's
# MODEL_SPECS[*]["ckpt"] writes (sg_fourier_model.ckpt-<step>.pt / sg_wno_model.ckpt-
# <step>.pt), and the equivalent dP_* naming if a parallel pressure script/checkpoints
# ever exists.
CKPT_PATTERNS = {
    ("sg", "fourier"): "sg_fourier_model.ckpt-*.pt",
    ("sg", "wno"): "sg_wno_model.ckpt-*.pt",
    ("dP", "fourier"): "dP_fourier_model.ckpt-*.pt",
    ("dP", "wno"): "dP_wno_model.ckpt-*.pt",
}

ARCH_LABEL = {"fourier": "Fourier-MIONet", "wno": "U-WNO-MIONet"}
ARCH_FILE_TAG = {"fourier": "fmionet", "wno": "uwno"}

# The 24 stored timesteps are indexed 0-23; the dataset only exposes a normalized
# [0, 1] trunk coordinate (see get_data()'s `xrt`), not real day/year values, so there
# is no way to recover an exact physical-time mapping from the code. These indices are
# early/mid/final placeholders approximating the requested ~37 day / ~1.8 yr / ~30 yr
# snapshots — edit if you know the true per-step schedule.
TIME_INDICES = [2, 8, 23]
TIME_LABELS = ["early (step 2)", "mid (step 8)", "final (step 23)"]
RADIAL_Z_INDEX = 48  # mid-depth
RADIAL_TIME_INDEX = 18  # mid-to-late, plume front well developed


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_best_checkpoint(pattern):
    paths = glob.glob(os.path.join(CKPT_DIR, pattern))
    best_path, best_step = None, -1
    for p in paths:
        m = STEP_RE.search(p)
        if not m:
            continue
        step = int(m.group(1))
        if step > best_step:
            best_step, best_path = step, p
    return best_path, best_step


def find_all_checkpoints():
    found = {}
    for (task, arch), pattern in CKPT_PATTERNS.items():
        path, step = find_best_checkpoint(pattern)
        if path is not None:
            found[(task, arch)] = (path, step)
    return found


def load_meta():
    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            return json.load(f)
    return {}


def save_meta(meta):
    os.makedirs(PRED_DIR, exist_ok=True)
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)


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


def get_predictions(task, arch, module, ckpt_path, step, data_bundle, meta, written_files):
    """Run inference (no training) and cache the result under predictions/, keyed by
    checkpoint step so this never reruns unless a newer checkpoint shows up."""
    npy_path = os.path.join(PRED_DIR, f"y_pred_{ARCH_FILE_TAG[arch]}_{task.lower()}.npy")
    meta_key = f"{task}_{arch}_step"

    if meta.get(meta_key) == step and os.path.exists(npy_path):
        print(f"  [{task}/{arch}] step {step} already cached -> {npy_path} (skipping inference)")
        return np.load(npy_path)

    print(f"  [{task}/{arch}] running inference from step {step} checkpoint: {ckpt_path}")
    model = load_trained_model(module, arch, ckpt_path, data_bundle)
    y_pred = module.predict_in_chunks(
        model, data_bundle["x_test"], branch_chunk=module.BATCH_SIZE, time_chunk=module.TIMESTEP_BATCH_SIZE
    )
    y_pred = y_pred.reshape(-1, 24, 96, 200)
    np.save(npy_path, y_pred)
    meta[meta_key] = step
    written_files.append(npy_path)
    return y_pred


def pick_case_indices(permeability_test, seed):
    """2 cases spanning the permeability-heterogeneity range (min/max std across the
    field) plus 1 additional random case, so plots aren't all drawn from one regime."""
    het_score = permeability_test.reshape(permeability_test.shape[0], -1).std(axis=1)
    idx_min_het = int(np.argmin(het_score))
    idx_max_het = int(np.argmax(het_score))
    rng = np.random.RandomState(seed)
    remaining = [i for i in range(permeability_test.shape[0]) if i not in (idx_min_het, idx_max_het)]
    idx_random = int(rng.choice(remaining)) if remaining else idx_min_het
    return {
        "min_heterogeneity": idx_min_het,
        "max_heterogeneity": idx_max_het,
        "random": idx_random,
    }


def process_task(task, module, found_for_task, field_name, apply_mask, meta):
    """Run inference (as needed) + generate every field plot for one task (sg or dP)."""
    written_files = []
    print(f"\n{'=' * 70}\n{task}: loading and normalizing data (no training)\n{'=' * 70}")
    data_bundle = module.load_and_normalize_data()

    y_true = data_bundle["y_test"].reshape(-1, 24, 96, 200)
    y_true_path = os.path.join(PRED_DIR, f"y_true_{task.lower()}.npy")
    np.save(y_true_path, y_true)
    written_files.append(y_true_path)

    permeability_test = data_bundle["x_test_field_raw"][..., 0]  # (N, 96, 200), unnormalized
    perm_path = os.path.join(PRED_DIR, f"permeability_test_{task.lower()}.npy")
    np.save(perm_path, permeability_test)
    written_files.append(perm_path)

    y_preds = {}
    for arch, (ckpt_path, step) in found_for_task.items():
        y_pred = get_predictions(task, arch, module, ckpt_path, step, data_bundle, meta, written_files)
        y_preds[ARCH_LABEL[arch]] = y_pred

    missing = [ARCH_LABEL[a] for a in ("fourier", "wno") if a not in found_for_task]
    if missing:
        print(f"  Note: no checkpoint found for {', '.join(missing)} — plotting only {list(y_preds.keys())}.")

    mask = plot_fields.build_plume_mask(y_true) if apply_mask else None

    case_indices = pick_case_indices(permeability_test, seed=module.SEED)
    print(f"  Selected test cases: {case_indices}")

    field_dir = os.path.join(PLOTS_FIELD_DIR, task.lower())
    os.makedirs(field_dir, exist_ok=True)

    for case_name, case_idx in case_indices.items():
        written_files.append(
            plot_fields.plot_field_snapshots(
                y_true, y_preds, case_idx, TIME_INDICES,
                os.path.join(field_dir, f"snapshots_{case_name}_case{case_idx}.png"),
                field_name=field_name, time_labels=TIME_LABELS,
            )
        )
        written_files.append(
            plot_fields.plot_radial_profile(
                y_true, y_preds, case_idx, RADIAL_Z_INDEX, RADIAL_TIME_INDEX,
                os.path.join(field_dir, f"radial_profile_{case_name}_case{case_idx}.png"),
                field_name=field_name,
            )
        )
        written_files.append(
            plot_fields.plot_input_context(
                permeability_test, case_idx,
                os.path.join(field_dir, f"input_context_{case_name}_case{case_idx}.png"),
            )
        )

    for model_name, y_pred in y_preds.items():
        tag = model_name.replace(" ", "").replace("-", "").lower()
        written_files.append(
            plot_fields.plot_scatter_true_vs_pred(
                y_true, y_pred, os.path.join(field_dir, f"scatter_{tag}.png"),
                model_name, field_name=field_name, mask=mask,
            )
        )
        written_files.append(
            plot_fields.plot_r2_histogram(
                y_true, y_pred, os.path.join(field_dir, f"r2_histogram_{tag}.png"),
                model_name, field_name=field_name, mask=mask,
            )
        )
        written_files.append(
            plot_fields.plot_mean_error_map(
                y_true, y_pred, os.path.join(field_dir, f"mean_error_map_{tag}.png"),
                model_name, field_name=field_name,
            )
        )

    return written_files


def main():
    print(f"{'=' * 70}\nScanning {CKPT_DIR} for checkpoints (no training will be run)\n{'=' * 70}")
    found = find_all_checkpoints()
    for (task, arch), (path, step) in found.items():
        print(f"  {task}/{arch}: using highest available step {step} -> {os.path.relpath(path, PROJECT_ROOT)}")

    if not found:
        print("No checkpoints found matching sg_fourier/sg_wno/dP_fourier/dP_wno patterns in pre_train/. Nothing to do.")
        return

    os.makedirs(PRED_DIR, exist_ok=True)
    os.makedirs(PLOTS_FIELD_DIR, exist_ok=True)
    os.makedirs(PLOTS_COMPARISON_DIR, exist_ok=True)
    meta = load_meta()
    written_files = []

    sg_found = {arch: v for (task, arch), v in found.items() if task == "sg"}
    if sg_found:
        sg_module = _load_module(os.path.join(PROJECT_ROOT, "Fourier-UWNO-MIONet_sg.py"), "fourier_mionet_sg")
        written_files += process_task(
            "sg", sg_module, sg_found, field_name="Gas Saturation", apply_mask=True, meta=meta
        )
    else:
        print("\nNo sg_fourier_model.ckpt-*.pt / sg_wno_model.ckpt-*.pt checkpoints found — skipping gas-saturation predictions/plots.")

    dP_found = {arch: v for (task, arch), v in found.items() if task == "dP"}
    if dP_found:
        dP_path = os.path.join(PROJECT_ROOT, "Fourier-UWNO-MIONet_dP.py")
        dP_module = _load_module(dP_path, "fourier_mionet_dp") if os.path.exists(dP_path) else None
        required_attrs = [
            "load_and_normalize_data", "build_net", "_build_fourier_decoder",
            "_build_wavelet_decoder", "predict_in_chunks", "Rsquare_plume_tegother",
            "MAE_plume", "reset_seed", "SEED", "LR", "DECAY_STEP", "DECAY_GAMMA",
            "BATCH_SIZE", "TIMESTEP_BATCH_SIZE",
        ]
        if dP_module is not None and all(hasattr(dP_module, a) for a in required_attrs):
            written_files += process_task(
                "dP", dP_module, dP_found, field_name="Pressure (bar)", apply_mask=False, meta=meta
            )
        else:
            print(
                "\ndP_fourier_model.ckpt-*.pt / dP_wno_model.ckpt-*.pt checkpoints exist, but "
                "Fourier-UWNO-MIONet_dP.py hasn't been refactored to the same dual-model interface as "
                "Fourier-UWNO-MIONet_sg.py (load_and_normalize_data/build_net/_build_*_decoder/"
                "predict_in_chunks/etc.) — skipping pressure predictions/plots. Ask for that "
                "refactor if you want this supported."
            )
    else:
        print("\nNo dP_fourier_model.ckpt-*.pt / dP_wno_model.ckpt-*.pt checkpoints found — skipping pressure predictions/plots.")

    save_meta(meta)

    for task_label, csv_path in (("sg", COMPARISON_CSV_SG), ("dP", COMPARISON_CSV_DP)):
        if os.path.exists(csv_path):
            out_dir = os.path.join(PLOTS_COMPARISON_DIR, task_label)
            print(f"\n{'=' * 70}\nPlotting {task_label} training curves from {csv_path}\n{'=' * 70}")
            written_files += plot_comparison.plot_all_comparisons(csv_path, out_dir)
        else:
            print(
                f"\n{csv_path} not found (it's only written once a full {task_label} dual-model "
                "training run completes) — skipping its loss/R2/MAE-vs-step comparison curves. "
                "Everything checkpoint-based above still ran."
            )

    print(f"\n{'=' * 70}\nDone. Files written:\n{'=' * 70}")
    for f in written_files:
        print(" -", os.path.relpath(f, PROJECT_ROOT))


if __name__ == "__main__":
    main()
