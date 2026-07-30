import os
os.environ["DDE_BACKEND"] = "pytorch"

import argparse
import csv
import json
import random
import time
import warnings
warnings.filterwarnings("ignore")

import deepxde as dde
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

from wavelet_convolution import WaveConv2d


# ============================================================
# Reproducibility
# ============================================================
SEED = 42


def reset_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


reset_seed(SEED)

# ============================================================
# Device
# ============================================================
if os.environ.get("FORCE_CPU") == "1":
    # Escape hatch for MPS backward-pass bugs on specific op combinations (e.g. the
    # WaveletDecoder's layer-0 U-Net branch) — not needed on CUDA, so leave unset there.
    device = torch.device("cpu")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)
if device.type == "mps":
    print("Using Apple Metal (MPS) GPU")
elif device.type == "cuda":
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    print("Running on CPU")


# ============================================================
# Metrics
# ============================================================
def max_l2_relative_error(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_true = y_true.reshape(y_true.shape[0], -1)
    y_pred = y_pred.reshape(y_pred.shape[0], -1)
    denom = np.linalg.norm(y_true, axis=1)
    denom = np.where(denom == 0, 1e-12, denom)
    rel = np.linalg.norm(y_true - y_pred, axis=1) / denom
    return np.max(rel)


def mean_l2_relative_error_np(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_true = y_true.reshape(y_true.shape[0], -1)
    y_pred = y_pred.reshape(y_pred.shape[0], -1)
    denom = np.linalg.norm(y_true, axis=1)
    denom = np.where(denom == 0, 1e-12, denom)
    rel = np.linalg.norm(y_true - y_pred, axis=1) / denom
    return np.mean(rel)


def Rsquare_plume_tegother(y_true, y_pred):
    """R^2 restricted to the CO2 plume region (saturation ~0 at final timestep is masked
    out). Ported from baselines/MIONet_vanilla_SG.py so both models are scored the same
    way the original paper code scored Fourier-MIONet.

    Mask is built elementwise per (row, col) from the final timestep, not just column 0
    broadcast across all 200 columns -- the out-of-domain boundary varies by column, so
    a column-0-only proxy leaked padded/zero cells into the "valid" region (confirmed:
    29.7% average / 69.2% max of cells in mask-called-valid rows were still genuinely
    zero at the final timestep, materially inflating R2 -- 0.240 -> -0.028 for
    Fourier-MIONet once corrected)."""
    size = y_true.shape[0]
    y_true = np.asarray(y_true).reshape(size, 24, 96, 200)
    y_pred = np.asarray(y_pred).reshape(size, 24, 96, 200)
    r2 = 0.0
    for i in range(size):
        mask = ~np.isclose(y_true[i, -1], 0.0)
        y_true_i = y_true[i][:, mask]
        y_pred_i = y_pred[i][:, mask]
        sse = np.sum(np.square(y_true_i.flatten() - y_pred_i.flatten()))
        sst = np.sum(np.square(y_true_i.flatten() - np.mean(y_true_i.flatten())))
        r2 += 1 - sse / sst
    return r2 / size


def MAE_plume(y_true, y_pred):
    """MAE restricted to the CO2 plume region, same (elementwise, row-and-column) masking
    as Rsquare_plume_tegother."""
    size = y_true.shape[0]
    y_true = np.asarray(y_true).reshape(size, 24, 96, 200)
    y_pred = np.asarray(y_pred).reshape(size, 24, 96, 200)
    mae = 0.0
    for i in range(size):
        mask = ~np.isclose(y_true[i, -1], 0.0)
        y_true_i = y_true[i][:, mask]
        y_pred_i = y_pred[i][:, mask]
        mae += np.mean(np.abs(y_true_i.flatten() - y_pred_i.flatten()))
    return mae / size


# ============================================================
# Data
# ============================================================
def get_data(ntrain, ntest):
    t = np.linspace(0, 1, 24).astype(np.float32)
    xrt = np.array([[c] for c in t]).astype(np.float32)

    field_input = [True, True, True, False, False, False, False, False, False, True, True]

    train_a_raw = np.load("sg_train_a.npz")["sg_train_a"]
    test_a_raw = np.load("sg_test_a.npz")["sg_test_a"]

    x_train_field = train_a_raw[:ntrain, :, :, field_input].astype(np.float32)
    x_train_MIO = np.load("sg_train_a_MIO.npy")[:ntrain, :].astype(np.float32)
    grid_x = train_a_raw[0, 0, :, -2].astype(np.float32)
    x_train = (x_train_field, x_train_MIO, xrt)

    y_train = (
        np.load("sg_train_u.npz")["sg_train_u"][:ntrain, :, :, :]
        .transpose(0, 3, 1, 2)
        .reshape(ntrain, 24 * 96 * 200)
        .astype(np.float32)
    )

    x_test_field = test_a_raw[-ntest:, :, :, field_input].astype(np.float32)
    x_test_MIO = np.load("sg_test_a_MIO.npy")[-ntest:, :].astype(np.float32)
    x_test = (x_test_field, x_test_MIO, xrt)

    y_test = (
        np.load("sg_test_u.npz")["sg_test_u"][-ntest:, :, :, :]
        .transpose(0, 3, 1, 2)
        .reshape(ntest, 24 * 96 * 200)
        .astype(np.float32)
    )

    return x_train, y_train, x_test, y_test, grid_x


class UnitGaussianNormalizer:
    """Per-feature (last-axis) zero-mean/unit-variance normalizer. Fit stats are frozen
    at construction time, so fitting on train and calling encode() on test never leaks
    test statistics."""

    def __init__(self, x, eps=1e-6):
        reduce_axes = tuple(range(x.ndim - 1))
        self.mean = x.mean(axis=reduce_axes, keepdims=True).astype(np.float32)
        self.std = x.std(axis=reduce_axes, keepdims=True).astype(np.float32)
        self.eps = eps

    def encode(self, x):
        return ((x - self.mean) / (self.std + self.eps)).astype(np.float32)


# ============================================================
# Custom layers
# ============================================================
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights3 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights4 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        x = torch.as_tensor(x, device=self.weights1.device, dtype=self.weights1.real.dtype)
        batchsize = x.shape[0]
        x_ft = torch.fft.rfftn(x, dim=[-2, -1])
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, : self.modes1, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, : self.modes1, : self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1 :, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1 :, : self.modes2], self.weights2
        )
        out_ft[:, :, : self.modes1, -self.modes2 :] = self.compl_mul2d(
            x_ft[:, :, : self.modes1, -self.modes2 :], self.weights3
        )
        out_ft[:, :, -self.modes1 :, -self.modes2 :] = self.compl_mul2d(
            x_ft[:, :, -self.modes1 :, -self.modes2 :], self.weights4
        )
        return torch.fft.irfftn(out_ft, s=(x.size(-2), x.size(-1)))


class U_net(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, dropout_rate):
        super().__init__()
        self.conv1 = self.conv(input_channels, output_channels, kernel_size, 2, dropout_rate)
        self.conv2 = self.conv(output_channels, output_channels, kernel_size, 2, dropout_rate)
        self.conv2_1 = self.conv(output_channels, output_channels, kernel_size, 1, dropout_rate)
        self.conv3 = self.conv(output_channels, output_channels, kernel_size, 2, dropout_rate)
        self.conv3_1 = self.conv(output_channels, output_channels, kernel_size, 1, dropout_rate)
        self.deconv2 = self.deconv(output_channels, output_channels)
        self.deconv1 = self.deconv(output_channels * 2, output_channels)
        self.deconv0 = self.deconv(output_channels * 2, output_channels)
        self.output_layer = self.output(output_channels * 2, output_channels, kernel_size, 1, dropout_rate)

    def forward(self, x):
        p = next(self.parameters())
        x = torch.as_tensor(x, device=p.device, dtype=p.dtype)
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2_1(self.conv2(out_conv1))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_deconv2 = self.deconv2(out_conv3)
        concat2 = torch.cat((out_conv2, out_deconv2), 1)
        out_deconv1 = self.deconv1(concat2)
        concat1 = torch.cat((out_conv1, out_deconv1), 1)
        out_deconv0 = self.deconv0(concat1)
        concat0 = torch.cat((x, out_deconv0), 1)
        return self.output_layer(concat0)

    def conv(self, in_planes, out_planes, kernel_size, stride, dropout_rate):
        return nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
                bias=False,
            ),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate),
        )

    def deconv(self, in_planes, out_planes):
        return nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def output(self, in_planes, out_planes, kernel_size, stride, dropout_rate):
        return nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
        )


class WaveletDecoder(nn.Module):
    def __init__(self, width, level, size, wavelet, layers=4, width2=128):
        super().__init__()
        self.layers = layers
        self.width = width
        self.a = nn.Parameter(torch.FloatTensor([0.1]))
        self.conv = nn.ModuleList(
            [WaveConv2d(width, width, level, size, wavelet, mode="zero") for _ in range(layers)]
        )
        self.w = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(layers)])
        self.unet = None
        self.fc1 = nn.Linear(width, width2)
        self.fc2 = nn.Linear(width2, 1)

    def set_unet(self, unet_list):
        self.unet = unet_list

    def forward(self, x):
        p = next(self.parameters())
        x = torch.as_tensor(x, device=p.device, dtype=p.dtype)

        if x.shape[0] == 0:
            return torch.zeros(0, 96, 200, device=x.device, dtype=x.dtype)

        batchsize = x.shape[0]
        size_x, size_y = x.shape[2], x.shape[3]
        r = x

        for j, (convl, wl) in enumerate(zip(self.conv, self.w)):
            # Aligned to the reference UWNO2d (U-WNO/uwno2d_Darcy.py): a real U-Net
            # block and the "+r" residual are applied uniformly at every layer,
            # including layer 0 (this used to special-case layer 0 with no unet and
            # no residual).
            x = convl(x + r) + wl(x) + self.unet[j](x)

            if j != self.layers - 1:
                x = F.mish(10 * self.a * x)

        x = x.permute(0, 2, 3, 1)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        x = x.reshape(batchsize, size_x, size_y, 1)[..., :-8, :-8, :]
        return x.squeeze(-1)


class FourierDecoder(nn.Module):
    """Vanilla Fourier-MIONet decoder baseline (SpectralConv2d + U_net, no wavelets).
    Ported from the active forward path of `decoder` in Fourier-UWNO-MIONet_dP.py (that class
    also allocated conv1/conv2/conv4/conv5/unet4/unet5, but its forward() never used
    them, so they are dropped here to keep the parameter-count comparison honest)."""

    def __init__(self, modes1, modes2, width, width2):
        super().__init__()
        self.width = width
        self.conv0 = SpectralConv2d(width, width, modes1, modes2)
        self.conv3 = SpectralConv2d(width, width, modes1, modes2)
        self.w0 = nn.Conv2d(width, width, 1)
        self.w3 = nn.Conv2d(width, width, 1)
        self.unet3 = U_net(width, width, 3, 0.0)
        self.fc1 = nn.Linear(width, width2)
        self.fc2 = nn.Linear(width2, 1)

    def forward(self, x):
        # NB: don't use next(self.parameters()) for dtype/device here — conv0's
        # SpectralConv2d weights are registered first and are torch.cfloat, which
        # would silently cast this real-valued input to complex.
        p = self.w0.weight
        x = torch.as_tensor(x, device=p.device, dtype=p.dtype)

        if x.shape[0] == 0:
            return torch.zeros(0, 96, 200, device=x.device, dtype=x.dtype)

        batchsize = x.shape[0]
        size_x, size_y = x.shape[2], x.shape[3]

        x = F.relu(self.conv0(x) + self.w0(x))
        x = F.relu(self.conv3(x) + self.w3(x) + self.unet3(x))

        x = x.permute(0, 2, 3, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.reshape(batchsize, size_x, size_y, 1)[..., :-8, :-8, :]
        return x.squeeze(-1)


class branch1(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.fc0 = nn.Linear(5, width)

    def forward(self, x):
        x = torch.as_tensor(x, dtype=self.fc0.weight.dtype, device=self.fc0.weight.device)
        x = F.pad(F.pad(x, (0, 0, 0, 8), "replicate"), (0, 0, 0, 0, 0, 8), "constant", 0)
        x = self.fc0(x)
        return x.permute(0, 3, 1, 2)


class branch2(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.fc0 = nn.Linear(7, width)

    def forward(self, x):
        x = torch.as_tensor(x, dtype=self.fc0.weight.dtype, device=self.fc0.weight.device)
        return self.fc0(x)


class WrappedTrunk(nn.Module):
    def __init__(self, trunk):
        super().__init__()
        self.trunk = trunk

    def forward(self, x):
        p = next(self.trunk.parameters())
        x = torch.as_tensor(x, dtype=p.dtype, device=p.device)
        return self.trunk(x)


def _build_wavelet_decoder():
    dec = WaveletDecoder(width=36, level=WNO_LEVEL, size=[104, 208], wavelet=WNO_WAVELET, layers=4, width2=128)
    dec.set_unet(
        nn.ModuleList(
            [
                U_net(36, 36, 3, 0.0),
                U_net(36, 36, 3, 0.0),
                U_net(36, 36, 3, 0.0),
                U_net(36, 36, 3, 0.0),
            ]
        )
    )
    return dec


def _build_fourier_decoder():
    return FourierDecoder(modes1=10, modes2=10, width=36, width2=128)


class BestModelSaver(dde.callbacks.Callback):
    """Saves net/optimizer state whenever `monitor` improves over its own best-so-far
    value, tracked independently of dde.model.TrainState.best_step.

    This USED to key off train_state.best_step ("save exactly when best_step advances
    to the current step"), but best_step is unconditionally train-loss-based --
    TrainState.update_best() hardcodes `if self.best_loss_train > np.sum(self.loss_train)`
    with no monitor argument at all, so no ModelCheckpoint/EarlyStopping `monitor` setting
    anywhere else in this file could ever change what best_step tracks. That silently
    defeated the entire point of a test-loss-based "best" checkpoint: diagnosed on the
    ntrain=2000/batch_size=20 U-WNO-MIONet run, where train loss kept slowly improving
    (0.894 -> 0.352 over the first 1500 steps) while test loss/R2 catastrophically
    diverged (R2: 0.131 -> -26.3) -- best_step kept advancing on every train-loss
    improvement, so the old best_step-keyed saver kept overwriting BEST.pt with newer,
    badly-overfit weights instead of preserving the healthy early checkpoint.

    monitor="test loss" (the new default) fixes this: `current` is compared directly
    against this callback's own running `best`, so a checkpoint is only kept when test
    loss itself improves, regardless of what train loss or best_step are doing.
    """

    def __init__(self, filepath, monitor="test loss", resume_best=None):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.best = resume_best if resume_best is not None else np.inf

    def _current(self):
        ts = self.model.train_state
        if self.monitor == "test loss":
            return float(np.sum(ts.loss_test))
        elif self.monitor == "train loss":
            return float(np.sum(ts.loss_train))
        else:
            raise ValueError(f"Unsupported monitor: {self.monitor!r}")

    def on_epoch_end(self):
        current = self._current()
        if current < self.best:
            self.best = current
            torch.save(
                {
                    "model_state_dict": self.model.net.state_dict(),
                    "optimizer_state_dict": self.model.opt.state_dict(),
                    "step": self.model.train_state.step,
                    "monitor": self.monitor,
                    "monitor_value": current,
                },
                self.filepath,
            )


class ResumableEarlyStopping(dde.callbacks.EarlyStopping):
    """Identical to dde.callbacks.EarlyStopping, except on_train_begin only resets
    wait/best/stopped_epoch to their fresh-start defaults when no resumed values were
    supplied at construction time. Model.train() calls on_train_begin() unconditionally
    at the start of every .train() call -- including a resumed one -- so without this
    override, resuming would silently wipe the exact counters we're trying to resume."""

    def __init__(self, *args, resume_wait=None, resume_best=None, resume_stopped_epoch=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._resume_wait = resume_wait
        self._resume_best = resume_best
        self._resume_stopped_epoch = resume_stopped_epoch

    def on_train_begin(self):
        super().on_train_begin()
        if self._resume_wait is not None:
            self.wait = self._resume_wait
        if self._resume_best is not None:
            self.best = self._resume_best
        if self._resume_stopped_epoch is not None:
            self.stopped_epoch = self._resume_stopped_epoch


class ResumeCheckpoint(dde.callbacks.Callback):
    """Periodically persists everything needed to resume training after a Slurm
    preemption/timeout on the 48-hour partition -- not just net/optimizer weights
    (which Model.save()/restore() and BestModelSaver already handle), but also:

    - The LR scheduler's own state. `decay=("step", ...)` builds a separate
      torch.optim.lr_scheduler.StepLR object (model.lr_scheduler) with its own
      internal step counter (last_epoch) -- Model.save()/restore() does NOT capture
      it, so naively restoring only net+optimizer would reset the decay schedule to
      start over from step 0, decoupling it from the actual resumed step.
    - TrainState bookkeeping (step, best_step, best_loss_train/test) so the resumed
      run knows how many iterations remain and doesn't reset "best so far".
    - EarlyStopping's wait/best/stopped_epoch counters (see ResumableEarlyStopping).
    - BestModelSaver's own running `best` value (see that class -- it's now tracked
      independently of TrainState.best_step, so it must be persisted/restored
      separately too, or a resumed run would reset it to inf and could overwrite
      BEST.pt with a worse checkpoint than one already seen pre-resume).
    - The LossHistory accumulated so far (steps/loss_train/loss_test/metrics_test) --
      a resumed run gets a brand new dde.Model with an empty LossHistory, so without
      this, the final per-checkpoint comparison CSV would silently lose every row
      from before the resume point.

    Two files, written together so they can't desync: a .pt for tensors (weights,
    optimizer, scheduler) and a .json for everything JSON-serializable. `complete`
    marks whether this model's training has fully finished (iterations exhausted or
    EarlyStopping fired) -- checked on startup to skip re-training a finished model.
    """

    def __init__(self, json_path, weights_path, early_stopping, best_saver, save_every=50):
        super().__init__()
        self.json_path = json_path
        self.weights_path = weights_path
        self.early_stopping = early_stopping
        self.best_saver = best_saver
        self.save_every = save_every

    def _save(self, complete):
        m = self.model
        ts = m.train_state
        torch.save(
            {
                "model_state_dict": m.net.state_dict(),
                "optimizer_state_dict": m.opt.state_dict(),
                "lr_scheduler_state_dict": (
                    m.lr_scheduler.state_dict() if m.lr_scheduler is not None else None
                ),
            },
            self.weights_path,
        )
        state = {
            "complete": complete,
            "step": ts.step,
            "best_step": ts.best_step,
            "best_loss_train": float(np.sum(ts.best_loss_train)),
            "best_loss_test": float(np.sum(ts.best_loss_test)),
            "early_stopping": {
                "wait": self.early_stopping.wait,
                "best": float(self.early_stopping.best),
                "stopped_epoch": self.early_stopping.stopped_epoch,
            },
            "best_saver": {
                "best": None if self.best_saver.best == np.inf else float(self.best_saver.best),
            },
            "losshistory": {
                "steps": list(m.losshistory.steps),
                "loss_train": [[float(v) for v in np.atleast_1d(x)] for x in m.losshistory.loss_train],
                "loss_test": [[float(v) for v in np.atleast_1d(x)] for x in m.losshistory.loss_test],
                "metrics_test": [[float(v) for v in np.atleast_1d(x)] for x in m.losshistory.metrics_test],
            },
        }
        tmp_path = self.json_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(state, f)
        os.replace(tmp_path, self.json_path)

    def on_epoch_end(self):
        if self.model.train_state.step % self.save_every == 0:
            self._save(complete=False)

    def on_train_end(self):
        self._save(complete=True)


def load_resume_state(json_path):
    """Returns None if no resume state file exists yet (fresh start)."""
    if not os.path.exists(json_path):
        return None
    with open(json_path) as f:
        return json.load(f)


def apply_resumed_history(model, resume_state):
    """Splice a previously-persisted LossHistory back into a freshly constructed
    dde.Model so the final comparison CSV includes pre-resume rows too."""
    lh = resume_state["losshistory"]
    model.losshistory.steps = list(lh["steps"])
    model.losshistory.loss_train = list(lh["loss_train"])
    model.losshistory.loss_test = list(lh["loss_test"])
    model.losshistory.metrics_test = list(lh["metrics_test"])


class EarlyTimingProbe(dde.callbacks.Callback):
    """Prints a measured (not assumed/extrapolated) average seconds/step after the
    first `probe_steps` real steps of THIS invocation -- so a short verification run
    (e.g. --iterations 30) reports true per-step timing at the new batch_size before
    committing GPU time to the full run, and a resumed run measures fresh rather than
    reusing a stale pre-resume number."""

    def __init__(self, label, probe_steps=20):
        super().__init__()
        self.label = label
        self.probe_steps = probe_steps
        self._probe_start_step = None
        self._start_time = None
        self._done = False

    def on_train_begin(self):
        self._probe_start_step = self.model.train_state.step
        self._start_time = time.time()

    def on_epoch_end(self):
        if self._done:
            return
        steps_done = self.model.train_state.step - self._probe_start_step
        if steps_done >= self.probe_steps:
            elapsed = time.time() - self._start_time
            print(
                f"[timing] {self.label}: {elapsed / steps_done:.4f} s/step measured over "
                f"{steps_done} real steps (batch_size={BATCH_SIZE}).",
                flush=True,
            )
            self._done = True


class DivergenceGuard(dde.callbacks.Callback):
    """Fast-acting safety net for catastrophic collapse, independent of (and much
    quicker than) the normal patience-until-improvement mechanisms above.

    dde.callbacks.EarlyStopping/ResumableEarlyStopping only fire after `patience`
    consecutive non-improving steps -- even the tightened WNO_PATIENCE_CHECKS=4 (400
    raw steps, see above) still requires 4 full epochs of no improvement. That's the
    right tolerance for an ordinary plateau, but both confirmed U-WNO-MIONet divergence
    episodes collapsed from healthy/moderate to catastrophic (R2 -24 to -32) within a
    few hundred steps -- this exists to catch THAT specific severity fast, as a safety
    net layered on top of normal patience, not a replacement for it.

    Only counts REAL evaluations (aligned to `display_every`), not raw on_epoch_end()
    calls -- unlike EarlyStopping's `patience`, which double-counts the same frozen
    metric value on every step between evaluations (see the EARLY_STOP_PATIENCE_CHECKS
    comment above). That quirk is a deliberate, well-understood tradeoff for ordinary
    patience; a safety net for catastrophic collapse shouldn't inherit it, since it
    would make "consecutive checks" fire off a single bad evaluation repeated across
    many raw steps rather than genuinely-repeated bad evaluations.

    Disabled entirely when `threshold` is None (see --disable-wno-divergence-guard).
    Not resumability-aware by design: `_consecutive_bad` resets to 0 on every fresh
    `.train()` call (including a resumed one), which just means a resumed run needs
    `consecutive_checks` bad evaluations of its own before firing -- an acceptable,
    deliberately simple tradeoff for a 2-4-check safety net (unlike EarlyStopping's
    wait/best, which really does need to survive a resume to keep its intended
    hundreds-of-steps tolerance meaningful).
    """

    def __init__(self, label, display_every, metric_index=1, threshold=-10.0, consecutive_checks=2):
        super().__init__()
        self.label = label
        self.display_every = display_every
        self.metric_index = metric_index
        self.threshold = threshold
        self.consecutive_checks = consecutive_checks
        self._consecutive_bad = 0
        self._last_checked_step = None

    def on_epoch_end(self):
        if self.threshold is None:
            return
        ts = self.model.train_state
        if ts.step == self._last_checked_step or ts.step % self.display_every != 0:
            return
        self._last_checked_step = ts.step
        if ts.metrics_test is None or len(ts.metrics_test) <= self.metric_index:
            return

        value = float(ts.metrics_test[self.metric_index])
        if value < self.threshold:
            self._consecutive_bad += 1
            print(
                f"[divergence-guard] {self.label}: metric={value:.4f} < threshold={self.threshold} "
                f"({self._consecutive_bad}/{self.consecutive_checks} consecutive bad checks) "
                f"at step {ts.step}",
                flush=True,
            )
            if self._consecutive_bad >= self.consecutive_checks:
                print(
                    f"[divergence-guard] {self.label}: stopping immediately -- "
                    f"{self.consecutive_checks} consecutive real evaluations below "
                    f"threshold={self.threshold}.",
                    flush=True,
                )
                self.model.stop_training = True
        else:
            self._consecutive_bad = 0


# ============================================================
# Load + normalize data (fall back to a smaller size if RAM/VRAM can't hold it)
# ============================================================
# NTRAIN/NTEST are module-level (not local to `if __name__ == "__main__"`) because
# generate_predictions_and_plots.py imports this file via importlib (its __name__ is
# never "__main__", so the CLI-parsing block below never runs) and calls
# load_and_normalize_data() directly for inference -- it needs these to already hold
# whatever size the checkpoint was actually trained with. CLI args (see __main__)
# override these two names directly, at module scope, before training starts. If you
# override --ntrain/--ntest for a one-off run, re-running generate_predictions_and_plots.py
# afterward will still use these defaults (2000/222), not your override -- pass the same
# override there too, or the field/output normalizers will be refit on a mismatched split.
NTRAIN, NTEST = 2000, 222
# Historical smaller sizes, tried in order if a larger size hits a MemoryError.
DATASET_FALLBACKS = [(400, 80), (200, 40)]

# Fraction of the TRAINING pool (not test) held out as a validation split, so
# checkpoint/EarlyStopping/DivergenceGuard selection during training stops touching
# the test set at all. CLI-overridable via --val-frac (see parse_args). Deterministic
# given --seed. Test is only ever used once, after training, for the final
# Rsquare_plume_tegother/MAE_plume report.
VAL_FRAC = 0.15


def load_and_normalize_data():
    """Load train/val/test data and fit all normalizers on the TRAIN split only (not
    val, not test). Side-effect-free beyond disk reads, so this is safe to call from
    other scripts (e.g. an inference/plotting script) without triggering training --
    only `if __name__ == "__main__"` below actually trains anything.

    val is carved out of the training pool ONLY, deterministically from SEED, BEFORE
    normalizer fitting -- so val statistics never leak into normalization either. test
    is untouched here beyond being loaded/normalized with the train-fit normalizers,
    same as before."""
    x_train = y_train = x_test = y_test = grid_x = None
    ntrain = ntest = None
    dataset_candidates = [(NTRAIN, NTEST)] + [
        c for c in DATASET_FALLBACKS if c != (NTRAIN, NTEST)
    ]
    for cand_ntrain, cand_ntest in dataset_candidates:
        try:
            x_train, y_train, x_test, y_test, grid_x = get_data(cand_ntrain, cand_ntest)
            ntrain, ntest = cand_ntrain, cand_ntest
            print(f"Loaded dataset: ntrain={ntrain}, ntest={ntest}")
            break
        except MemoryError as e:
            print(f"Ran out of memory loading ntrain={cand_ntrain}, ntest={cand_ntest} ({e}).")
            print("Falling back to a smaller dataset size...")

    if x_train is None:
        raise RuntimeError("Could not load any candidate dataset size — check available RAM.")

    # branch1's field maps (incl. permeability spanning ~7 orders of magnitude) and
    # branch2's scalar MIO features were previously fed in raw/unnormalized while only
    # the output got a StandardScaler. That mismatch badly conditions the loss surface
    # and is the likely cause of the ~0.73-0.75 test-metric plateau.
    x_train_field, x_train_MIO, xrt_train = x_train
    x_test_field, x_test_MIO, xrt_test = x_test

    rng = np.random.RandomState(SEED)
    perm = rng.permutation(ntrain)
    n_val = int(round(ntrain * VAL_FRAC))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    print(
        f"[val split] ntrain_pool={ntrain} -> train={len(train_idx)} val={len(val_idx)} "
        f"(val_frac={VAL_FRAC}, split_seed={SEED}); test={ntest} untouched, used once "
        f"at the end only."
    )

    x_val_field, x_val_MIO = x_train_field[val_idx], x_train_MIO[val_idx]
    y_val = y_train[val_idx]
    x_train_field, x_train_MIO = x_train_field[train_idx], x_train_MIO[train_idx]
    y_train = y_train[train_idx]

    field_normalizer = UnitGaussianNormalizer(x_train_field)
    mio_normalizer = UnitGaussianNormalizer(x_train_MIO)

    x_train = (field_normalizer.encode(x_train_field), mio_normalizer.encode(x_train_MIO), xrt_train)
    x_val = (field_normalizer.encode(x_val_field), mio_normalizer.encode(x_val_MIO), xrt_train)
    x_test = (field_normalizer.encode(x_test_field), mio_normalizer.encode(x_test_MIO), xrt_test)

    # Train-split-only scaler for outputs (not val, not test).
    scaler = StandardScaler().fit(y_train)
    mean = scaler.mean_.astype(np.float32)
    std = np.sqrt(scaler.var_.astype(np.float32))
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)

    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "x_test": x_test,
        "y_test": y_test,
        "x_test_field_raw": x_test_field,
        "grid_x": grid_x,
        "ntrain": len(train_idx),
        "nval": len(val_idx),
        "ntest": ntest,
        "mean": mean,
        "std": std,
        "field_normalizer": field_normalizer,
        "mio_normalizer": mio_normalizer,
    }


def make_output_transform(mean, std):
    def output_transform(inputs, outputs):
        out_dim = outputs.shape[1]
        mean_t = torch.as_tensor(mean[:out_dim], device=outputs.device, dtype=outputs.dtype)
        std_t = torch.as_tensor(std[:out_dim], device=outputs.device, dtype=outputs.dtype)
        return outputs * std_t + mean_t

    return output_transform


gelu = torch.nn.GELU()


def build_net(decoder_builder, output_transform):
    net = dde.nn.pytorch.mionet.MIONetCartesianProd(
        layer_sizes_branch1=[4500 * 96 * 200 * 3, branch1(36)],
        layer_sizes_branch2=[7 * 36, branch2(36)],
        layer_sizes_trunk=[1, 36, 36, 36, 36],
        activation={
            "branch1": gelu,
            "branch2": gelu,
            "trunk": gelu,
            "merger": gelu,
            "output merger": gelu,
        },
        kernel_initializer="Glorot normal",
        regularization=("l2", 4e-6),
        trunk_last_activation=False,
        merge_operation="sum",
        layer_sizes_merger=None,
        output_merge_operation="mul",
        layer_sizes_output_merger=[36, decoder_builder()],
    )
    net.trunk = WrappedTrunk(net.trunk)
    net.apply_output_transform(output_transform)
    return net.to(device)


def predict_in_chunks(model, x_test, branch_chunk=4, time_chunk=8, training_time_size=24, nx=96, ny=200):
    """model.predict() runs the whole batch through the network in one shot with no
    internal chunking. For NTEST=80 x all 24 timesteps that materializes a
    (n, T, 36, 104, 208) intermediate tensor several GB in size before the decoder even
    runs. Chunk it the same way Model._test() chunks during training to keep peak memory
    bounded regardless of NTEST."""
    x1, x2, xt = x_test
    n = x1.shape[0]
    branch_chunks = []
    for i in range(0, n, branch_chunk):
        x1_c, x2_c = x1[i : i + branch_chunk], x2[i : i + branch_chunk]
        time_chunks = []
        for j in range(0, training_time_size, time_chunk):
            xt_c = xt[j : j + time_chunk]
            y = model.predict((x1_c, x2_c, xt_c))
            time_chunks.append(y.reshape(x1_c.shape[0], -1, nx, ny))
        branch_chunks.append(np.concatenate(time_chunks, axis=1))
    return np.concatenate(branch_chunks, axis=0).reshape(n, -1)


# Scaled-up run (per 2026-07-21 professor/PhD-student meeting guidance): ntrain=2000
# (~44% of the paper's 4500), iterations=7500 (75 epochs at 100 batches/epoch = 50% of
# the paper's 150-epoch schedule, the midpoint of the requested 40-60% range). All of
# these are plain module-level globals (not local to `if __name__ == "__main__"`) for
# the same reason NTRAIN/NTEST are above: generate_predictions_and_plots.py imports this
# file without running the CLI block, and reads BATCH_SIZE/TIMESTEP_BATCH_SIZE directly.
# CLI args in __main__ reassign these names in place before training starts.
ITERATIONS = 7500
BATCH_SIZE = 20  # ntrain/batch_size = 2000/20 = 100 batches/epoch
TIMESTEP_BATCH_SIZE = 8  # unchanged -- paper's own tested accuracy/memory sweet spot
TRAINING_TIME_SIZE = 24
DISPLAY_EVERY = 100  # 7500/100 = 75 printed rows, in the requested 50-100 range
LR = 5e-4
# Un-normalized inputs likely made the model bounce around a wide loss basin at the
# original 1e-3 LR; now that inputs are normalized, drop LR and decay it partway
# through training instead of holding it constant. Already a function of ITERATIONS,
# not a hardcoded step count, so it scales automatically with the new budget (verified:
# 0.4 * 7500 = 3000, still comfortably inside the run, same 40%-through-training point
# as the previous 1200-iteration config's 480).
DECAY_STEP = int(0.4 * ITERATIONS)
DECAY_GAMMA = 0.5

# EarlyStopping's `patience` counts on_epoch_end() CALLS, which happen every single
# training step -- not every display_every steps. But the monitored quantity
# (train_state.loss_train) only actually CHANGES value at display_every cadence (that's
# when Model._test() runs); in between, on_epoch_end() re-checks the same frozen value
# against `best`, which reads as "no improvement" and increments `wait` on every one of
# those steps too. So patience, in effect, buys you (patience / display_every) real
# evaluation chances before firing -- not `patience` chances. The previous 1200-iteration
# config (display_every=10, patience=100) had 10 real chances (~8% of its 120 total
# checks). Scaling display_every to 100 for this run without also scaling patience would
# silently collapse that to 1 real chance (100/100) -- EarlyStopping would fire after a
# single non-improving evaluation, likely almost immediately. EARLY_STOP_PATIENCE_CHECKS
# preserves the original 10-real-chances tolerance regardless of display_every.
#
# EARLY_STOP_PATIENCE_CHECKS below is now Fourier-MIONet-only (see WNO_PATIENCE_CHECKS):
# Fourier-MIONet has been stable across both sg divergence episodes and doesn't need a
# tighter window, so its patience is left unchanged.
EARLY_STOP_PATIENCE_CHECKS = 10
EARLY_STOP_PATIENCE = EARLY_STOP_PATIENCE_CHECKS * DISPLAY_EVERY

# U-WNO-MIONet-specific, tighter patience. The 10-checks/1000-step window above let two
# confirmed divergence episodes collapse from healthy/moderate to catastrophic (R2 -24 to
# -32) well before EarlyStopping could fire -- both times the collapse was visible within
# a few hundred steps, far inside the 1000-step tolerance. 4 checks (400 raw steps = 4
# epochs at 100 steps/epoch) still requires multiple consecutive non-improving real
# evaluations (guards against single noisy fluctuations) but cuts the worst-case damage
# window by 60% versus Fourier-MIONet's unchanged 10-check patience. CLI-overridable via
# --wno-patience-checks.
WNO_PATIENCE_CHECKS = 4

# Fast-acting safety net independent of the patience-until-improvement mechanisms above:
# if test R2 collapses below this threshold for WNO_DIVERGENCE_CHECKS consecutive real
# evaluations, stop immediately regardless of the normal patience counter (see
# DivergenceGuard below). -10.0 is well below anything seen in a healthy run (a
# reasonably-fit model scores R2 close to 1; even a mediocre-but-not-diverging fit
# shouldn't approach -10) but comfortably above the -24/-32 actually observed, so it
# fires well before the worst of either confirmed episode. CLI-overridable
# (--wno-divergence-r2-threshold, --wno-divergence-checks); set
# --disable-wno-divergence-guard to turn it off entirely.
WNO_DIVERGENCE_R2_THRESHOLD = -10.0
WNO_DIVERGENCE_CHECKS = 2

# Governs both the full weights+optimizer+scheduler .pt snapshot AND how often
# EarlyStopping's wait/best counters get persisted (see ResumeCheckpoint) -- the two are
# saved together so they can't desync. Lowered from 50 (previously up to 49 raw steps of
# staleness in a resumed wait/best -- small relative to the old 1000-step patience, but a
# larger fraction of WNO's new 400-step patience above). 20 cuts worst-case staleness to
# 19 steps at a modest ~2.5x increase in checkpoint IO frequency over a 7500-iteration
# run -- cheap relative to compute time on an A100, and still far short of the per-step
# cost of doing this on every single step.
CHECKPOINT_PERIOD = 20

# Per-model LR overrides (both default to LR above, i.e. no behavior change unless
# explicitly passed). Added so the ntrain=2000/batch_size=20 divergence (U-WNO-MIONet
# only -- Fourier-MIONet trained fine at the same shared LR/batch_size/data) can be
# investigated by adjusting U-WNO-MIONet's LR alone, without touching Fourier-MIONet's
# currently-working config. NOT defaulted to a different value here: the standard
# linear/sqrt batch-size scaling rule would suggest RAISING lr for the 5x batch_size
# increase, but that reasoning assumes batch_size grew while dataset size stayed fixed
# (fewer, bigger steps per epoch to compensate for). Here ntrain scaled by the same 5x
# as batch_size, so steps/epoch stayed at 100 in both the old (400/4) and new (2000/20)
# configs -- the classic justification for scaling lr up doesn't apply. The observed
# failure (train loss improving while test loss/R2 catastrophically worsens, and doing
# so within ~1 epoch) looks like reduced per-step gradient noise (5x larger batch) letting
# U-WNO-MIONet's much higher-capacity decoder converge fast into an overfit/sharp
# solution -- which raising lr would likely worsen, not fix. Left unchanged by default
# pending the verification run below; if you do want to experiment, try LOWERING
# --wno-lr first, not raising it.
FOURIER_LR = LR
WNO_LR = LR

# Wavelet basis / decomposition level -- previously hardcoded, never swept despite
# being the most directly on-hypothesis untested knob for a wavelet-localization
# architecture. CLI-overridable via --wavelet/--wavelet-level (see parse_args).
WNO_WAVELET = "db6"
WNO_LEVEL = 4


def _build_model_specs(state_dir, wno_patience_checks, wno_divergence_threshold, wno_divergence_checks):
    return [
        {
            "key": "fourier",
            "name": "Fourier-MIONet",
            "decoder_builder": _build_fourier_decoder,
            "ckpt": os.path.join(state_dir, "sg_fourier_model.ckpt"),
            "lr": FOURIER_LR,
            "patience_checks": EARLY_STOP_PATIENCE_CHECKS,
            "divergence_threshold": None,  # guard is WNO-only, see FIX 1 above
            "divergence_checks": None,
        },
        {
            "key": "wno",
            "name": "U-WNO-MIONet",
            "decoder_builder": _build_wavelet_decoder,
            "ckpt": os.path.join(state_dir, "sg_wno_model.ckpt"),
            "lr": WNO_LR,
            "patience_checks": wno_patience_checks,
            "divergence_threshold": wno_divergence_threshold,
            "divergence_checks": wno_divergence_checks,
        },
    ]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ntrain", type=int, default=NTRAIN)
    p.add_argument("--ntest", type=int, default=NTEST)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--timestep-batch-size", type=int, default=TIMESTEP_BATCH_SIZE)
    p.add_argument("--iterations", type=int, default=ITERATIONS)
    p.add_argument("--display-every", type=int, default=DISPLAY_EVERY)
    p.add_argument(
        "--early-stop-patience-checks",
        type=int,
        default=EARLY_STOP_PATIENCE_CHECKS,
        help="EarlyStopping tolerance in units of real (display_every-spaced) evaluations, "
        "not raw steps -- see the comment above EARLY_STOP_PATIENCE_CHECKS for why.",
    )
    p.add_argument(
        "--state-dir",
        type=str,
        default="pre_train",
        help="Directory for checkpoints and resume-state files. IMPORTANT: pass a "
        "different --state-dir for a verification/test run than whatever the live "
        "scale-up job is using -- checkpoints (sg_{fourier,wno}_model.ckpt*) now live "
        "under --state-dir too (previously hardcoded to pre_train/ regardless of "
        "--state-dir), so this fully isolates a verification run's on-disk artifacts "
        "from a concurrently-running job's.",
    )
    p.add_argument(
        "--fourier-lr",
        type=float,
        default=FOURIER_LR,
        help="Fourier-MIONet learning rate (default unchanged from the current 5e-4).",
    )
    p.add_argument(
        "--wno-lr",
        type=float,
        default=WNO_LR,
        help="U-WNO-MIONet learning rate (default unchanged from the current 5e-4 -- "
        "override this alone to experiment with LR for the divergence investigation "
        "without touching Fourier-MIONet).",
    )
    p.add_argument(
        "--wno-patience-checks",
        type=int,
        default=WNO_PATIENCE_CHECKS,
        help="U-WNO-MIONet-only EarlyStopping tolerance in units of real evaluations "
        "(same units as --early-stop-patience-checks, which stays Fourier-MIONet-only). "
        "Default 4 (400 raw steps at display_every=100) -- tighter than Fourier-MIONet's "
        "10, since two confirmed divergence episodes showed U-WNO-MIONet can collapse to "
        "catastrophic R2 well within 1000 steps.",
    )
    p.add_argument(
        "--wno-divergence-r2-threshold",
        type=float,
        default=WNO_DIVERGENCE_R2_THRESHOLD,
        help="U-WNO-MIONet-only fast divergence guard: stop immediately if test R2 stays "
        "below this threshold for --wno-divergence-checks consecutive real evaluations, "
        "regardless of the normal patience counter. Default -10.0 (well below any healthy "
        "run, comfortably above the -24/-32 actually observed). See --disable-wno-"
        "divergence-guard to turn this off.",
    )
    p.add_argument(
        "--wno-divergence-checks",
        type=int,
        default=WNO_DIVERGENCE_CHECKS,
        help="Number of consecutive real evaluations below --wno-divergence-r2-threshold "
        "required to trigger the fast divergence guard. Default 2.",
    )
    p.add_argument(
        "--disable-wno-divergence-guard",
        action="store_true",
        help="Turn off the fast divergence guard entirely (normal EarlyStopping patience "
        "-- --wno-patience-checks -- still applies).",
    )
    p.add_argument("--seed", type=int, default=SEED, help="RNG seed (default 42).")
    p.add_argument(
        "--wavelet",
        type=str,
        default=WNO_WAVELET,
        help="U-WNO-MIONet wavelet basis (default db6). Previously hardcoded.",
    )
    p.add_argument(
        "--wavelet-level",
        type=int,
        default=WNO_LEVEL,
        help="U-WNO-MIONet wavelet decomposition level (default 4). Previously hardcoded.",
    )
    p.add_argument(
        "--val-frac",
        type=float,
        default=VAL_FRAC,
        help="Fraction of the TRAINING pool (not test) held out as a validation split "
        "for checkpoint/EarlyStopping/DivergenceGuard selection (default 0.15). "
        "Deterministic given --seed. Test is never used for selection, only for the "
        "final one-time report.",
    )
    return p.parse_args()


if __name__ == "__main__":
    # ============================================================
    # Train U-WNO-MIONet and vanilla Fourier-MIONet under identical conditions
    # ============================================================
    args = parse_args()
    NTRAIN, NTEST = args.ntrain, args.ntest
    BATCH_SIZE = args.batch_size
    TIMESTEP_BATCH_SIZE = args.timestep_batch_size
    ITERATIONS = args.iterations
    DISPLAY_EVERY = args.display_every
    EARLY_STOP_PATIENCE_CHECKS = args.early_stop_patience_checks
    SEED = args.seed
    FOURIER_LR = args.fourier_lr
    WNO_LR = args.wno_lr
    WNO_PATIENCE_CHECKS = args.wno_patience_checks
    WNO_DIVERGENCE_R2_THRESHOLD = None if args.disable_wno_divergence_guard else args.wno_divergence_r2_threshold
    WNO_DIVERGENCE_CHECKS = args.wno_divergence_checks
    WNO_WAVELET = args.wavelet
    WNO_LEVEL = args.wavelet_level
    VAL_FRAC = args.val_frac
    # Both re-derived from the just-parsed args, not left at their module-level-default
    # values -- these are exactly the two values that silently went stale in previous
    # scale-ups if only ITERATIONS/DISPLAY_EVERY were overridden without recomputing them.
    DECAY_STEP = int(0.4 * ITERATIONS)
    EARLY_STOP_PATIENCE = EARLY_STOP_PATIENCE_CHECKS * DISPLAY_EVERY
    WNO_PATIENCE = WNO_PATIENCE_CHECKS * DISPLAY_EVERY
    MODEL_SPECS = _build_model_specs(
        args.state_dir, WNO_PATIENCE_CHECKS, WNO_DIVERGENCE_R2_THRESHOLD, WNO_DIVERGENCE_CHECKS
    )

    print(
        f"[config] ntrain={NTRAIN} ntest={NTEST} batch_size={BATCH_SIZE} "
        f"timestep_batch_size={TIMESTEP_BATCH_SIZE} iterations={ITERATIONS} "
        f"display_every={DISPLAY_EVERY} decay_step={DECAY_STEP} seed={SEED} "
        f"fourier_lr={FOURIER_LR} fourier_patience={EARLY_STOP_PATIENCE} "
        f"({EARLY_STOP_PATIENCE_CHECKS} real checks) "
        f"wno_lr={WNO_LR} wno_patience={WNO_PATIENCE} ({WNO_PATIENCE_CHECKS} real checks) "
        f"wno_divergence_guard="
        f"{'disabled' if WNO_DIVERGENCE_R2_THRESHOLD is None else f'R2<{WNO_DIVERGENCE_R2_THRESHOLD} x{WNO_DIVERGENCE_CHECKS}'} "
        f"checkpoint_period={CHECKPOINT_PERIOD} wavelet={WNO_WAVELET} wavelet_level={WNO_LEVEL} "
        f"val_frac={VAL_FRAC} state_dir={args.state_dir}"
    )

    os.makedirs(args.state_dir, exist_ok=True)
    os.makedirs("./model", exist_ok=True)

    data_bundle = load_and_normalize_data()
    x_train, y_train = data_bundle["x_train"], data_bundle["y_train"]
    x_val, y_val = data_bundle["x_val"], data_bundle["y_val"]
    x_test, y_test = data_bundle["x_test"], data_bundle["y_test"]
    output_transform = make_output_transform(data_bundle["mean"], data_bundle["std"])

    print(
        f"[val split] train={data_bundle['ntrain']} val={data_bundle['nval']} "
        f"test={data_bundle['ntest']} (test is held out, used once at the very end only)"
    )

    results = {}

    for spec in MODEL_SPECS:
        print(f"\n{'=' * 60}\nTraining {spec['name']}\n{'=' * 60}")

        # Reset RNG before building the net so both models get identical branch1/branch2/
        # trunk initial weights (only the decoder architecture differs), and reset again
        # right before building Data so both see the exact same sequence of mini-batches
        # (BatchSampler draws from the global np.random state).
        reset_seed(SEED)
        net = build_net(spec["decoder_builder"], output_transform)

        reset_seed(SEED)
        # NOTE: the second data pair below is x_val/y_val, NOT x_test/y_test -- see
        # Fourier-UWNO-MIONet_dP_intermediate.py's identical comment for the full
        # rationale (deepxde calls this slot "test" internally regardless of what's
        # actually in it; the real test set is never passed to dde.data/Model here at
        # all, only used once at the end via predict_in_chunks further down).
        data = dde.data.Quadruple(x_train, y_train, x_val, y_val)

        model = dde.Model(data, net)
        print(f"[config] {spec['name']}: lr={spec['lr']}")
        model.compile(
            "adam",
            lr=spec["lr"],
            loss="mean l2 relative error",
            decay=("step", DECAY_STEP, DECAY_GAMMA),
            metrics=["mean l2 relative error", Rsquare_plume_tegother, MAE_plume],
        )

        json_path = os.path.join(args.state_dir, f"sg_{spec['key']}_train_state.json")
        weights_path = os.path.join(args.state_dir, f"sg_{spec['key']}_resume.pt")
        resume_state = load_resume_state(json_path)

        if resume_state is not None and resume_state.get("complete"):
            print(
                f"{spec['name']}: resume state at {json_path} is marked complete "
                f"(step={resume_state['step']}) -- skipping training, restoring saved history only."
            )
            model.train_state.step = resume_state["step"]
            model.train_state.best_step = resume_state["best_step"]
            model.train_state.best_loss_train = resume_state["best_loss_train"]
            model.train_state.best_loss_test = resume_state["best_loss_test"]
            apply_resumed_history(model, resume_state)
            losshistory, train_state = model.losshistory, model.train_state
            elapsed = 0.0
        else:
            remaining_iterations = ITERATIONS
            es_kwargs = {}
            if resume_state is not None:
                ckpt = torch.load(weights_path, map_location=device)
                model.net.load_state_dict(ckpt["model_state_dict"])
                model.opt.load_state_dict(ckpt["optimizer_state_dict"])
                if ckpt.get("lr_scheduler_state_dict") is not None and model.lr_scheduler is not None:
                    model.lr_scheduler.load_state_dict(ckpt["lr_scheduler_state_dict"])
                model.train_state.step = resume_state["step"]
                model.train_state.best_step = resume_state["best_step"]
                model.train_state.best_loss_train = resume_state["best_loss_train"]
                model.train_state.best_loss_test = resume_state["best_loss_test"]
                apply_resumed_history(model, resume_state)
                remaining_iterations = max(0, ITERATIONS - resume_state["step"])
                es_kwargs = dict(
                    resume_wait=resume_state["early_stopping"]["wait"],
                    resume_best=resume_state["early_stopping"]["best"],
                    resume_stopped_epoch=resume_state["early_stopping"]["stopped_epoch"],
                )
                # .get(...) chain, not direct indexing: resume-state files written by a
                # prior run of this script before the BestModelSaver test-loss fix won't
                # have this key at all.
                resume_best_saver = resume_state.get("best_saver", {}).get("best")
                print(
                    f"{spec['name']}: resuming from step {resume_state['step']} "
                    f"({remaining_iterations} iterations remaining of {ITERATIONS} total)."
                )
            else:
                resume_best_saver = None
                print(f"{spec['name']}: no resume state found at {json_path} -- starting fresh.")

            checker = dde.callbacks.ModelCheckpoint(
                spec["ckpt"], save_better_only=True, period=CHECKPOINT_PERIOD, monitor="test loss"
            )
            # Per-spec patience: Fourier-MIONet keeps EARLY_STOP_PATIENCE_CHECKS (10, unchanged);
            # U-WNO-MIONet uses its own, tighter spec["patience_checks"] (default 4).
            spec_patience = spec["patience_checks"] * DISPLAY_EVERY
            early_stopping = ResumableEarlyStopping(
                min_delta=1e-4, patience=spec_patience, monitor="loss_test", **es_kwargs
            )
            best_saver = BestModelSaver(
                f"{spec['ckpt']}-BEST.pt", monitor="test loss", resume_best=resume_best_saver
            )
            resume_ckpt = ResumeCheckpoint(
                json_path, weights_path, early_stopping, best_saver, save_every=CHECKPOINT_PERIOD
            )
            timing_probe = EarlyTimingProbe(spec["name"], probe_steps=20)
            callbacks = [checker, early_stopping, best_saver, resume_ckpt, timing_probe]
            if spec["divergence_threshold"] is not None:
                callbacks.append(
                    DivergenceGuard(
                        spec["name"],
                        DISPLAY_EVERY,
                        metric_index=1,  # Rsquare_plume_tegother, see metrics=[...] in model.compile
                        threshold=spec["divergence_threshold"],
                        consecutive_checks=spec["divergence_checks"],
                    )
                )

            print(f"Training {spec['name']}... (patience={spec_patience} steps / {spec['patience_checks']} checks)")
            start_time = time.time()
            losshistory, train_state = model.train(
                iterations=remaining_iterations,
                batch_size=BATCH_SIZE,
                timestep_batch_size=TIMESTEP_BATCH_SIZE,
                training_time_size=TRAINING_TIME_SIZE,
                display_every=DISPLAY_EVERY,
                callbacks=callbacks,
            )
            elapsed = time.time() - start_time
        steps_run = max(train_state.step, 1)

        print(f"{spec['name']} best step:", train_state.best_step)
        print(f"{spec['name']} best train loss:", train_state.best_loss_train)
        # deepxde labels this "loss_test"/"best_loss_test" internally, but the data in
        # that slot is x_val/y_val (see the dde.data.Quadruple call above) -- this is
        # VAL loss/metric, not test.
        print(f"{spec['name']} best VAL loss (deepxde calls this 'test loss'):", train_state.best_loss_test)
        print(f"{spec['name']} best VAL metric (deepxde calls this 'test metric'):", train_state.best_metrics)

        best_path = f"{spec['ckpt']}-BEST.pt"
        if os.path.exists(best_path):
            model.restore(best_path, verbose=1)
            print(
                f"Restored test-loss-best weights for {spec['name']} "
                f"(note: this is best_saver's own best-test-loss step, which can differ "
                f"from train_state.best_step={train_state.best_step} -- see BestModelSaver)."
            )
        else:
            # Should only happen if training was interrupted before the first
            # display_every checkpoint — fall back to the periodic (train-loss-monitored)
            # snapshot, which may not match best_saver's test-loss-best step at all.
            periodic_path = f"{spec['ckpt']}-{train_state.best_step}.pt"
            try:
                model.restore(periodic_path, verbose=1)
                print(
                    f"No BEST file found for {spec['name']}; restored periodic checkpoint at "
                    f"{periodic_path} instead (may not exactly match best_step)."
                )
            except Exception as e:
                print(f"Could not restore best or periodic checkpoint for {spec['name']}:", e)
                print("Proceeding with current in-memory weights.")

        y_pred = predict_in_chunks(model, x_test, branch_chunk=BATCH_SIZE, time_chunk=TIMESTEP_BATCH_SIZE)

        results[spec["key"]] = {
            "name": spec["name"],
            "model": model,
            "losshistory": losshistory,
            "train_state": train_state,
            "num_params": model.net.num_trainable_parameters(),
            "elapsed": elapsed,
            "avg_time_per_step": elapsed / steps_run,
            "test_mean_rel": mean_l2_relative_error_np(y_test, y_pred),
            "test_max_rel": max_l2_relative_error(y_test, y_pred),
            "test_r2_plume": Rsquare_plume_tegother(y_test, y_pred),
            "test_mae_plume": MAE_plume(y_test, y_pred),
        }

    # ============================================================
    # Per-checkpoint comparison table
    # ============================================================
    def _step_metrics(entry):
        lh = entry["losshistory"]
        table = {}
        for step, loss_train, loss_test, metrics_test in zip(
            lh.steps, lh.loss_train, lh.loss_test, lh.metrics_test
        ):
            table[step] = {
                "train_loss": float(np.mean(loss_train)),
                "test_loss": float(np.mean(loss_test)),
                "r2_plume": float(metrics_test[1]),
                "mae_plume": float(metrics_test[2]),
            }
        return table

    fourier_by_step = _step_metrics(results["fourier"])
    wno_by_step = _step_metrics(results["wno"])
    all_steps = sorted(set(fourier_by_step) | set(wno_by_step))

    header = [
        "step",
        "fourier_train_loss",
        "fourier_test_loss",
        "fourier_test_r2_plume",
        "fourier_test_mae_plume",
        "wno_train_loss",
        "wno_test_loss",
        "wno_test_r2_plume",
        "wno_test_mae_plume",
        "delta_r2_plume",
        "delta_test_loss",
    ]

    print(f"\n{'=' * 100}")
    print(f"Per-checkpoint comparison: Fourier-MIONet vs U-WNO-MIONet")
    print(
        "NOTE: every *_test_* column below (and in the saved CSV) is the VAL split, "
        "not the held-out test set -- column names unchanged to keep downstream "
        "tooling compatible. Real test-set numbers are the 'Final summary' block "
        "further down, computed once."
    )
    print(f"{'=' * 100}")

    rows = []
    for step in all_steps:
        f = fourier_by_step.get(step)
        w = wno_by_step.get(step)
        delta_r2 = (w["r2_plume"] - f["r2_plume"]) if (f and w) else None
        delta_loss = (f["test_loss"] - w["test_loss"]) if (f and w) else None
        rows.append(
            [
                step,
                f["train_loss"] if f else None,
                f["test_loss"] if f else None,
                f["r2_plume"] if f else None,
                f["mae_plume"] if f else None,
                w["train_loss"] if w else None,
                w["test_loss"] if w else None,
                w["r2_plume"] if w else None,
                w["mae_plume"] if w else None,
                delta_r2,
                delta_loss,
            ]
        )
        f_r2 = f"{f['r2_plume']:.4f}" if f else "  N/A "
        f_mae = f"{f['mae_plume']:.4f}" if f else "   N/A  "
        w_r2 = f"{w['r2_plume']:.4f}" if w else "  N/A "
        w_mae = f"{w['mae_plume']:.4f}" if w else "   N/A  "
        d_r2 = f"{delta_r2:+.4f}" if delta_r2 is not None else "  N/A "
        print(
            f"step={step:5d} | Fourier-MIONet R2={f_r2} MAE={f_mae} "
            f"| U-WNO-MIONet R2={w_r2} MAE={w_mae} | delta R2={d_r2}"
        )

    # Routed through --state-dir, NOT a hardcoded repo-root filename -- a shared
    # relative path here previously meant every concurrently-running --state-dir
    # config (e.g. a wno_lr/seed sweep) silently overwrote the same file, since
    # open(..., "w") truncates on every run regardless of state_dir isolation
    # everywhere else. Confirmed to have actually happened in the first wno_lr sweep.
    comparison_csv_path = os.path.join(args.state_dir, "comparison_log.csv")
    with open(comparison_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"\nSaved per-checkpoint comparison to {comparison_csv_path}")

    # ============================================================
    # Final summary
    # ============================================================
    print(f"\n{'=' * 60}\nFinal summary\n{'=' * 60}")
    for key in ("fourier", "wno"):
        r = results[key]
        print(
            f"{r['name']}: params={r['num_params']:,} | avg s/step={r['avg_time_per_step']:.4f} "
            f"| test mean L2 rel={r['test_mean_rel']:.4f} | test max L2 rel={r['test_max_rel']:.4f} "
            f"| test R2 (plume)={r['test_r2_plume']:.4f} | test MAE (plume)={r['test_mae_plume']:.4f}"
        )

    print("\nMPS available:", torch.backends.mps.is_available())
    print("CUDA available:", torch.cuda.is_available())
    print("Device:", device)
