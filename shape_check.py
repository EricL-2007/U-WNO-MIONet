# shape_check.py
# Run this on CPU in seconds before committing to a full GPU training run
# Verifies that WaveletDecoder produces the correct output shape

import torch
import torch.nn as nn
from wavelet_decoder import WaveletDecoder

# Simulate the merged branch-trunk tensor z
# Shape: (batchcase * batchtime, width, H_padded, W_padded)
# = (4 * 8, 36, 104, 208)
dummy_z = torch.randn(32, 36, 104, 208)

# Build decoder
decoder = WaveletDecoder(
    width   = 36,
    level   = 4,
    size    = [104, 208],
    wavelet = 'db6',
    layers  = 4,
    width2  = 128
)

# Inject dummy U_net for shape check
# (U_net not imported here to keep this file standalone)
class DummyUNet(nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)

unet_list = nn.ModuleList([
    DummyUNet(), DummyUNet(), DummyUNet(), DummyUNet()
])
decoder.set_unet(unet_list)
decoder.eval()

with torch.no_grad():
    out = decoder(dummy_z)

print(f"Input shape:  {dummy_z.shape}")   # torch.Size([32, 36, 104, 208])
print(f"Output shape: {out.shape}")        # torch.Size([32, 96, 200])

assert out.shape == (32, 96, 200), \
    f"SHAPE MISMATCH: expected (32, 96, 200), got {out.shape}"

print("\nShape check PASSED. Safe to run training.")

# Parameter count
total_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
print(f"WaveletDecoder trainable parameters: {total_params:,}")