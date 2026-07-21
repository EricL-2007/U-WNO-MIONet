import os
os.environ["DDE_BACKEND"] = "pytorch"

import csv
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
# Pressure is defined everywhere (unlike gas saturation), but the padded grid still
# has out-of-domain cells filled with this sentinel value; mask those out rather than
# scoring against padding.
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


def Rsquare_plume(y_true, y_pred):
    """Per-timestep R^2 outside the out-of-domain padding, averaged over timesteps and
    samples. Ported unchanged from the original Fourier-MIONet_dP.py."""
    size = y_true.shape[0]
    y_true = np.asarray(y_true).reshape(size, 24, 96, 200)
    y_pred = np.asarray(y_pred).reshape(size, 24, 96, 200)
    r2 = 0.0
    for i in range(size):
        z_axis = y_true[i, -1, :, 0]
        mask = ~np.isclose(z_axis, OUT_OF_DOMAIN_SENTINEL)
        for j in range(24):
            y_true_i = y_true[i, j, mask, :]
            y_pred_i = y_pred[i, j, mask, :]
            sse = np.sum(np.square(y_true_i.flatten() - y_pred_i.flatten()))
            sst = np.sum(np.square(y_true_i.flatten() - np.mean(y_true_i.flatten())))
            r2 += 1 - sse / sst
    return r2 / 24 / size


def Rsquare_plume_tegother(y_true, y_pred):
    """R^2 outside the out-of-domain padding, pooling all timesteps per sample before
    averaging over samples. Ported unchanged from the original Fourier-MIONet_dP.py."""
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
#  dP_train_u.npz on disk is NOT raw pressure -- confirmed by direct inspection: its
#  values are already ~N(0, 1) (mean 6.9e-5, std 0.999), and its dominant constant value
#  (-0.22228621, ~67% of cells) exactly matches OUT_OF_DOMAIN_SENTINEL above. dP_test_u.npz
#  on disk IS raw physical pressure (bar) -- mean 4.36, std 19.9, with ~65% of cells
#  exactly 0.0 (the raw out-of-domain fill value). These two constants reproduce that
#  exact mapping: (0.0 - DP_TARGET_MEAN) / DP_TARGET_STD == -0.22228620..., matching
#  OUT_OF_DOMAIN_SENTINEL to 8 decimal places. So train and test were, at some point
#  upstream, meant to be on the same normalized scale via this exact transform -- it's
#  the same transform an earlier version of this script applied to y_test only (see the
#  removed-code note this replaces), which a later change dropped on the theory that it
#  was an inconsistency bug. It wasn't: dropping it left y_test raw (up to 445) being
#  compared against a model trained entirely on y_train's already-normalized ~[-1.4, 25]
#  scale, and left OUT_OF_DOMAIN_SENTINEL unable to match test's mask cells at all. Set
#  DP_RESCALE_TEST_TARGETS=0 to disable this and reproduce the old (broken) behavior for
#  A/B comparison.
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
    This is the same active-path cleanup used in Fourier-MIONet_sg.py: the original
    `decoder` class here allocated conv1/conv2/conv4/conv5/unet4/unet5, but its
    forward() never used them, so they're dropped to keep the parameter-count
    comparison honest."""

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


# dP is a smoother, more diffuse field than sg's sharp CO2 plume fronts, and this decoder
# dominates the model's parameter count almost entirely (WaveConv2d's wavelet-domain
# weights scale with the padded [104, 208] grid and level, not with the FC head) -- see
# DP_WNO_LAYERS below. width=36 is NOT a free parameter here: it's the branch/trunk merge
# output width shared with Fourier-MIONet's decoder too (see build_net's layer_sizes_*),
# so changing it would require adding a projection layer into the decoder, confounding a
# capacity change with an architecture change. layers is the one cleanly isolated capacity
# knob available without touching branch/trunk/merge.
DP_WNO_LAYERS = int(os.environ.get("DP_WNO_LAYERS", "4"))


def _build_wavelet_decoder():
    dec = WaveletDecoder(
        width=36, level=4, size=[104, 208], wavelet="db6", layers=DP_WNO_LAYERS, width2=128
    )
    dec.set_unet(nn.ModuleList([U_net(36, 36, 3, 0.0) for _ in range(DP_WNO_LAYERS)]))
    return dec


def _build_fourier_decoder():
    return FourierDecoder(modes1=10, modes2=10, width=36, width2=128)


class BestModelSaver(dde.callbacks.Callback):
    """Saves net/optimizer state exactly when train_state.best_step advances to the
    current step. ModelCheckpoint's own save_better_only+period only checks for an
    improvement every `period` steps — and its `monitor` can be a different quantity
    than best_step's (best_step is always picked by train loss, hardcoded in
    TrainState.update_best(); ModelCheckpoint may be watching test loss instead) — so
    the periodic checkpoint file can silently miss, or never match, the exact step
    DeepXDE itself reports as best. This file is written in the same format
    Model.save()/restore() use, so model.restore(this_path) works unchanged, and it
    always corresponds exactly to train_state.best_step at the time of writing.
    """

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


# ============================================================
# Load + normalize data (fall back to a smaller size if RAM/VRAM can't hold it)
# ============================================================
DATASET_CANDIDATES = [(400, 80), (200, 40)]


def load_and_normalize_data():
    """Load train/test data and fit all normalizers on train only. Side-effect-free
    beyond disk reads, so this is safe to call from other scripts (e.g. an inference/
    plotting script) without triggering training — only `if __name__ == "__main__"`
    below actually trains anything."""
    x_train = y_train = x_test = y_test = grid_x = None
    ntrain = ntest = None
    for cand_ntrain, cand_ntest in DATASET_CANDIDATES:
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

    # grid_dx feeds the radial-derivative loss term; grid_dy was computed in the
    # original script but never actually used anywhere, so it's dropped here.
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

    The original loss_fnc(y_true, y_pred, train_indices, istrain) looked up a
    precomputed per-sample mask via train_indices — but this deepxde fork's
    Quadruple/QuadrupleCartesianProd.losses() only ever calls loss_fn(targets,
    outputs), so train_indices/istrain are never actually passed through and the
    original signature would raise a TypeError on the first training step. Since
    QuadrupleCartesianProd doesn't chunk timesteps, y_true always carries the full
    24 timesteps, so the mask can just be recomputed from y_true itself here —
    simpler than indices, and it works identically for train and test batches.
    """
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
    """model.predict() runs the whole batch through the network in one shot with no
    internal chunking, which would materialize a huge intermediate tensor for a large
    NTEST x all-24-timesteps call. Chunk it the same way Model._test() chunks during
    training to keep peak memory bounded."""
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


ITERATIONS = 1200
BATCH_SIZE = 4
TIMESTEP_BATCH_SIZE = 8
TRAINING_TIME_SIZE = 24
DISPLAY_EVERY = 10
# Preserving the original dP script's optimizer/decay-rate character (rmsprop,
# gamma=0.9) but NOT its decay cadence. Rescaling the original "decay every 3375 of
# 168750 steps" ratio down to this 1200-iteration run gave DECAY_STEP=24 — 50 decay
# events crammed into 1200 steps, collapsing LR to ~7% of initial by the halfway
# point and ~0.5% by the end. That starved the model of any real optimization budget
# for the back half of training (confirmed: test R2 flat/negative from ~step 150
# onward while train loss kept slowly falling — classic frozen-LR-plus-overfitting).
# Match sg's formula instead: few, large-effect decays sized to the actual iteration
# budget, not a ratio inherited from a 140x-longer original run.
LR = 1e-3
DECAY_STEP = int(0.4 * ITERATIONS)
DECAY_GAMMA = 0.9

MODEL_SPECS = [
    {
        "key": "fourier",
        "name": "Fourier-MIONet",
        "decoder_builder": _build_fourier_decoder,
        "ckpt": "pre_train/dP_fourier_model.ckpt",
    },
    {
        "key": "wno",
        "name": "U-WNO-MIONet",
        "decoder_builder": _build_wavelet_decoder,
        "ckpt": "pre_train/dP_wno_model.ckpt",
    },
]


if __name__ == "__main__":
    # ============================================================
    # Train U-WNO-MIONet and vanilla Fourier-MIONet under identical conditions
    # ============================================================
    os.makedirs("./pre_train", exist_ok=True)
    os.makedirs("./model", exist_ok=True)

    data_bundle = load_and_normalize_data()
    x_train, y_train = data_bundle["x_train"], data_bundle["y_train"]
    x_test, y_test = data_bundle["x_test"], data_bundle["y_test"]
    output_transform = make_output_transform(data_bundle["mean"], data_bundle["std"])

    results = {}

    for spec in MODEL_SPECS:
        print(f"\n{'=' * 60}\nTraining {spec['name']}\n{'=' * 60}")

        # Reset RNG before building the net so both models get identical branch1/branch2/
        # trunk initial weights (only the decoder architecture differs), and reset again
        # right before building Data so both see the exact same sequence of mini-batches
        # (BatchSampler draws from the global np.random state).
        reset_seed(SEED)
        net = build_net(spec["decoder_builder"], output_transform)
        n_params = sum(p.numel() for p in net.parameters())
        extra = f" (DP_WNO_LAYERS={DP_WNO_LAYERS})" if spec["key"] == "wno" else ""
        print(f"[param count] {spec['name']}: {n_params:,} total params{extra}")

        reset_seed(SEED)
        # Preserved from the original dP script: QuadrupleCartesianProd (not Quadruple)
        # — every training step sees the full 24 timesteps for its sampled branch batch,
        # unlike sg's 8-of-24 timestep chunking.
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

        # Preserved from the original: monitor="test loss" (sg's checkpointer defaults
        # to train loss instead). Note this means the periodic checkpoint's own
        # save_better_only bookkeeping watches a DIFFERENT quantity than best_step
        # (which is always train-loss-based, hardcoded in TrainState.update_best()) —
        # exactly why BestModelSaver below exists: it's the only thing guaranteed to
        # match best_step regardless of what `checker` is monitoring.
        checker = dde.callbacks.ModelCheckpoint(
            spec["ckpt"], save_better_only=True, period=50, monitor="test loss"
        )
        best_saver = BestModelSaver(f"{spec['ckpt']}-BEST.pt")
        # Added alongside the LR-decay fix above: without this, nothing stopped
        # training from grinding through all 1200 steps on the same 400 samples
        # once test performance plateaued. Matches sg's config and criterion
        # (train loss, same as best_step's own tracking).
        early_stopping = dde.callbacks.EarlyStopping(min_delta=1e-4, patience=100)

        print(f"Training {spec['name']}...")
        start_time = time.time()
        losshistory, train_state = model.train(
            iterations=ITERATIONS,
            batch_size=BATCH_SIZE,
            timestep_batch_size=TIMESTEP_BATCH_SIZE,
            training_time_size=TRAINING_TIME_SIZE,
            display_every=DISPLAY_EVERY,
            init_test=True,
            callbacks=[checker, best_saver, early_stopping],
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
            # Should only happen if training was interrupted before the first
            # display_every checkpoint — fall back to the periodic snapshot, which
            # may not exactly match best_step (see BestModelSaver docstring).
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
    print(f"Per-checkpoint comparison: Fourier-MIONet vs U-WNO-MIONet (pressure buildup)")
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

    with open("comparison_log_dp.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print("\nSaved per-checkpoint comparison to comparison_log_dp.csv")

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
