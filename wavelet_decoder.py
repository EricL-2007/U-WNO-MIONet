# wavelet_decoder.py
# Replaces the decoder() class in Fourier-MIONet_sg.py
# Sources:
#   - Logic/structure: UWNO2d.forward() in U-WNO/uwno2d_Darcy.py
#   - WaveConv2d:      U-WNO/wavelet_convolution.py
#   - U_net class:     reused from Fourier-MIONet_sg.py (imported below)

import torch
import torch.nn as nn
import torch.nn.functional as F
from wavelet_convolution import WaveConv2d


class WaveletDecoder(nn.Module):
    def __init__(self, width, level, size, wavelet, layers=4, width2=128):
        """
        Args:
            width   : int   - channel dimension (36, matches Fourier-MIONet)
            level   : int   - wavelet decomposition levels (4)
            size    : list  - spatial size of input feature map [H, W] = [104, 208]
            wavelet : str   - wavelet family ('db6')
            layers  : int   - number of U-WNO decoder layers (4)
            width2  : int   - hidden dim in projection head (128)
        """
        super(WaveletDecoder, self).__init__()

        self.layers = layers
        self.width  = width

        # Trainable adaptive activation parameter (from uwno2d_Darcy.py line 109)
        # Initialized to 0.1, updated by backprop every step
        self.a = nn.Parameter(torch.FloatTensor([0.1]))

        # WaveConv2d: DWT -> learned weights on 4 coefficient types -> IDWT
        # mode='zero' matches GCS reservoir's zero-flux boundary condition
        # size=[104, 208] is the padded feature map shape from branch1
        self.conv = nn.ModuleList([
            WaveConv2d(width, width, level, size, wavelet, mode='zero')
            for _ in range(layers)
        ])

        # Pointwise Conv2d: mixes channels at each spatial location independently
        # kernel_size=1 means no spatial mixing — purely channel-wise
        self.w = nn.ModuleList([
            nn.Conv2d(width, width, 1)
            for _ in range(layers)
        ])

        # U_net blocks: imported from Fourier-MIONet_sg.py, no changes
        # Loaded dynamically to avoid circular import
        self.unet = None  # set after import in get_model() below

        # Projection head: same as original decoder fc1/fc2
        self.fc1 = nn.Linear(width, width2)
        self.fc2 = nn.Linear(width2, 1)

    def set_unet(self, unet_list):
        """Call this after importing U_net from Fourier-MIONet_sg.py"""
        self.unet = unet_list

    def forward(self, x):
    # Skip empty batches
        if x.shape[0] == 0:
            return torch.zeros(0, 96, 200, device=x.device)

        batchsize      = x.shape[0]
        size_x, size_y = x.shape[2], x.shape[3]
        r = x

        for j, (convl, wl) in enumerate(zip(self.conv, self.w)):
            if j == 0:
                x = convl(x) + wl(x)
            else:
                unetl = self.unet[j]
                x = convl(x + r) + wl(x) + unetl(x)
            if j != self.layers - 1:
                x = F.mish(10 * self.a * x)

        x = x.permute(0, 2, 3, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.view(batchsize, size_x, size_y, 1)[..., :-8, :-8, :]
        return x.squeeze(-1)