"""Intermediate data-scale test for dP (pressure buildup): ntrain=1000 (up from the
400 used for the layers/width capacity A/B tests), carrying forward whichever capacity
config those tests found best (DP_WNO_LAYERS/DP_WNO_WIDTH env vars, same as
Fourier-UWNO-MIONet_dP.py). This is a cheap DIRECTIONAL check (~20-30 epoch-equivalents,
not sg's full 75-epoch scale-up) meant to show whether more data is worth pursuing
further before committing a longer run.

Reuses the resumability infrastructure (ResumeCheckpoint, ResumableEarlyStopping,
EarlyTimingProbe, the EARLY_STOP_PATIENCE_CHECKS display_every-aware patience fix) built
for Fourier-UWNO-MIONet_sg.py's scale-up, ported here unchanged. dP's own architecture
choices (QuadrupleCartesianProd full-24-timestep batching, rmsprop, the masked
relative-L2 + radial-derivative loss, the RESCALE_TEST_TARGETS train/test scale fix) are
preserved from Fourier-UWNO-MIONet_dP.py, NOT replaced with sg's (adam, plain
"mean l2 relative error", Quadruple) -- those are dP-specific fixes/characteristics
unrelated to the data-scale question this script tests.

Usage:
    python Fourier-UWNO-MIONet_dP_intermediate.py [--ntrain 1000] [--ntest 80]
        [--batch-size 10] [--timestep-batch-size 8] [--iterations 2500] ...

DO NOT run the full-scale version of this without first checking the timing probe
output ([timing] ... s/step) against the requested --time budget -- see
run_dp_intermediate.sbatch.
"""
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
# Pressure is defined everywhere (unlike gas saturation), but the padded grid still
# has out-of-domain cells filled with this sentinel value; mask those out rather than
# scoring against padding. Ported unchanged from Fourier-UWNO-MIONet_dP.py.
OUT_OF_DOMAIN_SENTINEL = -0.22228621


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
    """R^2 outside the out-of-domain padding, pooling all timesteps per sample before
    averaging over samples. Ported unchanged from Fourier-UWNO-MIONet_dP.py."""
    size = y_true.shape[0]
    y_true = np.asarray(y_true).reshape(size, 24, 96, 200)
    y_pred = np.asarray(y_pred).reshape(size, 24, 96, 200)
    r2 = 0.0
    for i in range(size):
        z_axis = y_true[i, -1, :, 0]
        mask = ~np.isclose(z_axis, OUT_OF_DOMAIN_SENTINEL)
        y_true_i = y_true[i][:, mask, :]
        y_pred_i = y_pred[i][:, mask, :]
        sse = np.sum(np.square(y_true_i.flatten() - y_pred_i.flatten()))
        sst = np.sum(np.square(y_true_i.flatten() - np.mean(y_true_i.flatten())))
        r2 += 1 - sse / sst
    return r2 / size


def MAE_plume(y_true, y_pred):
    """MAE outside the out-of-domain padding, same masking as Rsquare_plume_tegother."""
    size = y_true.shape[0]
    y_true = np.asarray(y_true).reshape(size, 24, 96, 200)
    y_pred = np.asarray(y_pred).reshape(size, 24, 96, 200)
    mae = 0.0
    for i in range(size):
        z_axis = y_true[i, -1, :, 0]
        mask = ~np.isclose(z_axis, OUT_OF_DOMAIN_SENTINEL)
        y_true_i = y_true[i][:, mask, :]
        y_pred_i = y_pred[i][:, mask, :]
        mae += np.mean(np.abs(y_true_i.flatten() - y_pred_i.flatten()))
    return mae / size


# ============================================================
# Data
# ============================================================
# See Fourier-UWNO-MIONet_dP.py for the full derivation of these two constants: train
# targets are already normalized on disk, test targets are raw physical pressure (bar),
# and this rescale puts them on the same scale (and makes 0.0 map back onto
# OUT_OF_DOMAIN_SENTINEL). Set DP_RESCALE_TEST_TARGETS=0 to disable for A/B comparison.
DP_TARGET_MEAN = 4.172939172019009
DP_TARGET_STD = 18.772821433027488
RESCALE_TEST_TARGETS = os.environ.get("DP_RESCALE_TEST_TARGETS", "1") != "0"


def get_data(ntrain, ntest):
    t = np.linspace(0, 1, 24).astype(np.float32)
    xrt = np.array([[c] for c in t]).astype(np.float32)

    field_input = [True, True, True, False, False, False, False, False, False, True, True]

    train_a_raw = np.load("dP_train_a.npz")["dP_train_a"]
    test_a_raw = np.load("dP_test_a.npz")["dP_test_a"]

    x_train_field = train_a_raw[:ntrain, :, :, field_input].astype(np.float32)
    x_train_MIO = np.load("dP_train_a_MIO.npy")[:ntrain, :].astype(np.float32)
    grid_x = train_a_raw[0, 0, :, -2].astype(np.float32)
    x_train = (x_train_field, x_train_MIO, xrt)

    y_train = (
        np.load("dP_train_u.npz")["dP_train_u"][:ntrain, :, :, :]
        .transpose(0, 3, 1, 2)
        .reshape(ntrain, 24 * 96 * 200)
        .astype(np.float32)
    )

    x_test_field = test_a_raw[-ntest:, :, :, field_input].astype(np.float32)
    x_test_MIO = np.load("dP_test_a_MIO.npy")[-ntest:, :].astype(np.float32)
    x_test = (x_test_field, x_test_MIO, xrt)

    y_test_raw = (
        np.load("dP_test_u.npz")["dP_test_u"][-ntest:, :, :, :]
        .transpose(0, 3, 1, 2)
        .reshape(ntest, 24 * 96 * 200)
        .astype(np.float32)
    )

    print(
        f"[dP target scale] y_train (as loaded, already normalized upstream): "
        f"mean={y_train.mean():.6g} std={y_train.std():.6g} "
        f"min={y_train.min():.6g} max={y_train.max():.6g}"
    )
    print(
        f"[dP target scale] y_test  (as loaded, RAW physical bar):           "
        f"mean={y_test_raw.mean():.6g} std={y_test_raw.std():.6g} "
        f"min={y_test_raw.min():.6g} max={y_test_raw.max():.6g}"
    )
    if RESCALE_TEST_TARGETS:
        y_test = ((y_test_raw - DP_TARGET_MEAN) / DP_TARGET_STD).astype(np.float32)
        print(
            f"[dP target scale] DP_RESCALE_TEST_TARGETS=1: rescaled y_test with "
            f"(raw - {DP_TARGET_MEAN}) / {DP_TARGET_STD} -> "
            f"mean={y_test.mean():.6g} std={y_test.std():.6g} "
            f"min={y_test.min():.6g} max={y_test.max():.6g} "
            f"(now matches y_train's scale, and 0.0 maps to {(0.0 - DP_TARGET_MEAN) / DP_TARGET_STD:.8f} "
            f"~= OUT_OF_DOMAIN_SENTINEL={OUT_OF_DOMAIN_SENTINEL})"
        )
    else:
        y_test = y_test_raw
        print(
            "[dP target scale] DP_RESCALE_TEST_TARGETS=0: using RAW y_test unchanged "
            "(reproduces prior broken behavior -- train/test on mismatched scales, "
            "OUT_OF_DOMAIN_SENTINEL will not match test's mask cells)."
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
    def __init__(self, width, level, size, wavelet, layers=4, width2=128, input_width=None):
        super().__init__()
        self.layers = layers
        self.width = width
        # input_width is the branch/trunk merge output's fixed channel count (36, see
        # DP_WNO_WIDTH below). When it differs from this decoder's internal `width`, a
        # 1x1-conv projection is needed at the entry point, since every WaveConv2d/U_net
        # block below operates at `width` channels internally.
        self.input_proj = (
            nn.Conv2d(input_width, width, 1) if input_width is not None and input_width != width else None
        )
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

        if self.input_proj is not None:
            x = self.input_proj(x)

        batchsize = x.shape[0]
        size_x, size_y = x.shape[2], x.shape[3]
        r = x

        for j, (convl, wl) in enumerate(zip(self.conv, self.w)):
            x = convl(x + r) + wl(x) + self.unet[j](x)

            if j != self.layers - 1:
                x = F.mish(10 * self.a * x)

        x = x.permute(0, 2, 3, 1)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        x = x.reshape(batchsize, size_x, size_y, 1)[..., :-8, :-8, :]
        return x.squeeze(-1)


class FourierDecoder(nn.Module):
    """Vanilla Fourier-MIONet decoder baseline (SpectralConv2d + U_net, no wavelets)."""

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


# Carried forward from the now-completed layers/width capacity A/B tests on the
# ntrain=400 reduced scale (see Fourier-UWNO-MIONet_dP.py for the full results table):
#   layers=4, width=36 (original): 8.39M params, R2=-42.16, MAE=0.727
#   layers=2, width=36:            4.20M params, R2=-2.99,  MAE=0.638
#   layers=1, width=36:            2.10M params, R2=-3.00,  MAE=0.627  <- best MAE
#   layers=1, width=18:            0.53M params, R2=-3.18,  MAE=0.631
# layers=1/width=36 is the CONFIRMED best config (best MAE, R2 tied with the narrower
# width=18 leg) -- width reduction below 36 showed no benefit, so both are now locked-in
# defaults here, not placeholders pending further tuning.
DP_WNO_LAYERS = int(os.environ.get("DP_WNO_LAYERS", "1"))
DP_WNO_WIDTH = int(os.environ.get("DP_WNO_WIDTH", "36"))
_MERGE_WIDTH = 36  # fixed by branch/trunk output channels; not a free parameter


def _build_wavelet_decoder():
    dec = WaveletDecoder(
        width=DP_WNO_WIDTH,
        level=4,
        size=[104, 208],
        wavelet="db6",
        layers=DP_WNO_LAYERS,
        width2=128,
        input_width=_MERGE_WIDTH,
    )
    dec.set_unet(nn.ModuleList([U_net(DP_WNO_WIDTH, DP_WNO_WIDTH, 3, 0.0) for _ in range(DP_WNO_LAYERS)]))
    return dec


def _build_fourier_decoder():
    return FourierDecoder(modes1=10, modes2=10, width=36, width2=128)


class BestModelSaver(dde.callbacks.Callback):
    """Saves net/optimizer state exactly when train_state.best_step advances to the
    current step -- see Fourier-UWNO-MIONet_sg.py's identical class for the full
    rationale (ModelCheckpoint's save_better_only+period can miss or mismatch
    best_step, which is always train-loss-based)."""

    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath
        self._last_saved_step = None

    def on_epoch_end(self):
        ts = self.model.train_state
        if ts.best_step == ts.step and ts.best_step != self._last_saved_step:
            torch.save(
                {
                    "model_state_dict": self.model.net.state_dict(),
                    "optimizer_state_dict": self.model.opt.state_dict(),
                    "best_step": ts.best_step,
                },
                self.filepath,
            )
            self._last_saved_step = ts.best_step


class ResumableEarlyStopping(dde.callbacks.EarlyStopping):
    """Ported unchanged from Fourier-UWNO-MIONet_sg.py: identical to
    dde.callbacks.EarlyStopping, except on_train_begin only resets wait/best/
    stopped_epoch to fresh-start defaults when no resumed values were supplied at
    construction time (Model.train() calls on_train_begin() unconditionally, including
    on a resumed run)."""

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
    """Ported unchanged from Fourier-UWNO-MIONet_sg.py: periodically persists net/
    optimizer/LR-scheduler state plus TrainState/EarlyStopping/LossHistory bookkeeping,
    so a Slurm preemption/timeout can resume without losing the decay schedule, the
    "best so far" tracking, or the per-checkpoint comparison CSV's earlier rows."""

    def __init__(self, json_path, weights_path, early_stopping, save_every=50):
        super().__init__()
        self.json_path = json_path
        self.weights_path = weights_path
        self.early_stopping = early_stopping
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
    """Ported unchanged from Fourier-UWNO-MIONet_sg.py: prints a measured average
    seconds/step after the first `probe_steps` real steps of THIS invocation, so real
    timing at the new ntrain/batch_size is available before committing to a longer
    run."""

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


# ============================================================
# Load + normalize data (fall back to a smaller size if RAM/VRAM can't hold it)
# ============================================================
# Module-level (not local to `if __name__ == "__main__"`) for the same reason as
# Fourier-UWNO-MIONet_sg.py's NTRAIN/NTEST: any future inference/plotting script that
# imports this file via importlib needs these to hold the actually-trained size. CLI
# args in __main__ override these names directly, before training starts.
NTRAIN, NTEST = 1000, 80
# Historical smaller sizes, tried in order if a larger size hits a MemoryError.
DATASET_FALLBACKS = [(400, 80), (200, 40)]


def load_and_normalize_data():
    """Load train/test data and fit all normalizers on train only. Side-effect-free
    beyond disk reads."""
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

    x_train_field, x_train_MIO, xrt_train = x_train
    x_test_field, x_test_MIO, xrt_test = x_test

    field_normalizer = UnitGaussianNormalizer(x_train_field)
    mio_normalizer = UnitGaussianNormalizer(x_train_MIO)

    x_train = (field_normalizer.encode(x_train_field), mio_normalizer.encode(x_train_MIO), xrt_train)
    x_test = (field_normalizer.encode(x_test_field), mio_normalizer.encode(x_test_MIO), xrt_test)

    # Train-only scaler for outputs
    scaler = StandardScaler().fit(y_train)
    mean = scaler.mean_.astype(np.float32)
    std = np.sqrt(scaler.var_.astype(np.float32))
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)

    # grid_dx feeds the radial-derivative loss term.
    grid_dx = grid_x[1:-1] + grid_x[:-2] / 2 + grid_x[2:] / 2
    grid_dx = grid_dx.reshape(1, 1, 1, 198).astype(np.float32)

    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
        "x_test_field_raw": x_test_field,
        "grid_x": grid_x,
        "grid_dx": grid_dx,
        "ntrain": ntrain,
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


def make_loss_fnc(grid_dx, device):
    """Relative-L2 + radial-derivative loss, masked outside the simulation domain.
    Ported unchanged from Fourier-UWNO-MIONet_dP.py."""
    grid_dx_t = torch.as_tensor(grid_dx, dtype=torch.float32, device=device)

    def loss_fnc(y_true, y_pred):
        size = y_true.shape[0]
        timesize = int(y_true.shape[1] / 200 / 96)
        y_true = y_true.reshape(size, timesize, 96, 200)
        y_pred = y_pred.reshape(size, timesize, 96, 200)

        z_axis = y_true[:, -1, :, 0]
        sentinel = torch.as_tensor(OUT_OF_DOMAIN_SENTINEL, dtype=z_axis.dtype, device=z_axis.device)
        mask = (~torch.isclose(z_axis, sentinel)).to(torch.float32)
        mask = mask.reshape(size, 1, 96, 1)

        y_true = y_true * mask
        y_pred = y_pred * mask
        dydx_true_x = (y_true[:, :, :, 2:] - y_true[:, :, :, :-2]) / grid_dx_t
        dydx_pred_x = (y_pred[:, :, :, 2:] - y_pred[:, :, :, :-2]) / grid_dx_t
        y_true = y_true.reshape(size, timesize * 96 * 200)
        y_pred = y_pred.reshape(size, timesize * 96 * 200)
        dydx_true_x = dydx_true_x.reshape(size, timesize * 96 * 198)
        dydx_pred_x = dydx_pred_x.reshape(size, timesize * 96 * 198)
        ori_loss = torch.mean(torch.norm(y_true - y_pred, 2, dim=1) / torch.norm(y_true, 2, dim=1))
        der_loss_x = torch.mean(torch.norm(dydx_true_x - dydx_pred_x, 2, dim=1) / torch.norm(dydx_true_x, 2, dim=1))

        return [ori_loss, der_loss_x]

    return loss_fnc


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
    """Chunk prediction the same way Model._test() chunks during training, to keep
    peak memory bounded for a large NTEST x all-24-timesteps call."""
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


# Intermediate data-scale test (per Task 2 spec): ntrain=1000 (2.5x the 400 used for the
# layers/width capacity A/B tests, but well short of sg's ntrain=2000 scale-up),
# batch_size=10 keeps the same 100-batches/epoch convention sg's scale-up uses
# (1000/10 = 100, matching sg's 2000/20 = 100). ITERATIONS=2500 is ~25 epoch-equivalents
# (25 * 100 = 2500) -- a modest, directional check, not the full 75-epoch target used for
# sg. All CLI-overridable so a longer run can follow if this looks promising.
NTRAIN, NTEST = 1000, 80
ITERATIONS = 2500
BATCH_SIZE = 10  # ntrain/batch_size = 1000/10 = 100 batches/epoch
TIMESTEP_BATCH_SIZE = 8  # unchanged -- paper's own tested accuracy/memory sweet spot
TRAINING_TIME_SIZE = 24
DISPLAY_EVERY = 25  # 2500/25 = 100 printed rows
LR = 1e-3
DECAY_STEP = int(0.4 * ITERATIONS)
DECAY_GAMMA = 0.9

# See Fourier-UWNO-MIONet_sg.py's identical comment: patience counts on_epoch_end()
# calls (every step), but the monitored quantity only changes at display_every cadence,
# so EARLY_STOP_PATIENCE_CHECKS (real evaluation chances) is what should stay fixed
# across display_every changes, not a raw patience step count.
EARLY_STOP_PATIENCE_CHECKS = 10
EARLY_STOP_PATIENCE = EARLY_STOP_PATIENCE_CHECKS * DISPLAY_EVERY
CHECKPOINT_PERIOD = 50

MODEL_SPECS = [
    {
        "key": "fourier",
        "name": "Fourier-MIONet",
        "decoder_builder": _build_fourier_decoder,
        "ckpt": "pre_train/dP_intermediate_fourier_model.ckpt",
    },
    {
        "key": "wno",
        "name": "U-WNO-MIONet",
        "decoder_builder": _build_wavelet_decoder,
        "ckpt": "pre_train/dP_intermediate_wno_model.ckpt",
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
        help="Directory for checkpoints and resume-state files.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    NTRAIN, NTEST = args.ntrain, args.ntest
    BATCH_SIZE = args.batch_size
    TIMESTEP_BATCH_SIZE = args.timestep_batch_size
    ITERATIONS = args.iterations
    DISPLAY_EVERY = args.display_every
    EARLY_STOP_PATIENCE_CHECKS = args.early_stop_patience_checks
    DECAY_STEP = int(0.4 * ITERATIONS)
    EARLY_STOP_PATIENCE = EARLY_STOP_PATIENCE_CHECKS * DISPLAY_EVERY

    print(
        f"[config] ntrain={NTRAIN} ntest={NTEST} batch_size={BATCH_SIZE} "
        f"timestep_batch_size={TIMESTEP_BATCH_SIZE} iterations={ITERATIONS} "
        f"display_every={DISPLAY_EVERY} decay_step={DECAY_STEP} "
        f"early_stop_patience={EARLY_STOP_PATIENCE} ({EARLY_STOP_PATIENCE_CHECKS} real checks) "
        f"DP_WNO_LAYERS={DP_WNO_LAYERS} DP_WNO_WIDTH={DP_WNO_WIDTH}"
    )

    os.makedirs(args.state_dir, exist_ok=True)
    os.makedirs("./model", exist_ok=True)

    data_bundle = load_and_normalize_data()
    x_train, y_train = data_bundle["x_train"], data_bundle["y_train"]
    x_test, y_test = data_bundle["x_test"], data_bundle["y_test"]
    output_transform = make_output_transform(data_bundle["mean"], data_bundle["std"])

    results = {}

    for spec in MODEL_SPECS:
        print(f"\n{'=' * 60}\nTraining {spec['name']}\n{'=' * 60}")

        reset_seed(SEED)
        net = build_net(spec["decoder_builder"], output_transform)
        n_params = sum(p.numel() for p in net.parameters())
        extra = f" (DP_WNO_LAYERS={DP_WNO_LAYERS}, DP_WNO_WIDTH={DP_WNO_WIDTH})" if spec["key"] == "wno" else ""
        print(f"[param count] {spec['name']}: {n_params:,} total params{extra}")

        reset_seed(SEED)
        # Preserved from Fourier-UWNO-MIONet_dP.py: QuadrupleCartesianProd (not
        # Quadruple) -- every training step sees the full 24 timesteps for its sampled
        # branch batch, unlike sg's 8-of-24 timestep chunking.
        data = dde.data.QuadrupleCartesianProd(x_train, y_train, x_test, y_test)

        model = dde.Model(data, net)
        model.compile(
            "rmsprop",
            loss=make_loss_fnc(data_bundle["grid_dx"], device),
            loss_weights=[1, 0.5],
            lr=LR,
            decay=("step", DECAY_STEP, DECAY_GAMMA),
            metrics=["mean l2 relative error", Rsquare_plume_tegother, MAE_plume],
        )

        json_path = os.path.join(args.state_dir, f"dP_intermediate_{spec['key']}_train_state.json")
        weights_path = os.path.join(args.state_dir, f"dP_intermediate_{spec['key']}_resume.pt")
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
                print(
                    f"{spec['name']}: resuming from step {resume_state['step']} "
                    f"({remaining_iterations} iterations remaining of {ITERATIONS} total)."
                )
            else:
                print(f"{spec['name']}: no resume state found at {json_path} -- starting fresh.")

            checker = dde.callbacks.ModelCheckpoint(
                spec["ckpt"], save_better_only=True, period=CHECKPOINT_PERIOD, monitor="test loss"
            )
            early_stopping = ResumableEarlyStopping(min_delta=1e-4, patience=EARLY_STOP_PATIENCE, **es_kwargs)
            best_saver = BestModelSaver(f"{spec['ckpt']}-BEST.pt")
            resume_ckpt = ResumeCheckpoint(json_path, weights_path, early_stopping, save_every=CHECKPOINT_PERIOD)
            timing_probe = EarlyTimingProbe(spec["name"], probe_steps=20)

            print(f"Training {spec['name']}...")
            start_time = time.time()
            losshistory, train_state = model.train(
                iterations=remaining_iterations,
                batch_size=BATCH_SIZE,
                timestep_batch_size=TIMESTEP_BATCH_SIZE,
                training_time_size=TRAINING_TIME_SIZE,
                display_every=DISPLAY_EVERY,
                init_test=True,
                callbacks=[checker, early_stopping, best_saver, resume_ckpt, timing_probe],
            )
            elapsed = time.time() - start_time
        steps_run = max(train_state.step, 1)

        print(f"{spec['name']} best step:", train_state.best_step)
        print(f"{spec['name']} best train loss:", train_state.best_loss_train)
        print(f"{spec['name']} best test loss:", train_state.best_loss_test)
        print(f"{spec['name']} best test metric:", train_state.best_metrics)

        best_path = f"{spec['ckpt']}-BEST.pt"
        if os.path.exists(best_path):
            model.restore(best_path, verbose=1)
            print(f"Restored guaranteed-best weights for {spec['name']} (best_step={train_state.best_step}).")
        else:
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
    print(f"Per-checkpoint comparison: Fourier-MIONet vs U-WNO-MIONet (pressure buildup, intermediate scale)")
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

    with open("comparison_log_dp_intermediate.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print("\nSaved per-checkpoint comparison to comparison_log_dp_intermediate.csv")

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
