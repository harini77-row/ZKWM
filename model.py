"""
model.py — the encoder and decoder architectures, each defined exactly once.

probe.py, train_watermark.py, day2_export.py, and test_real_image.py all
import Decoder (and, where needed, Encoder) from here. Do not redefine
either in any of those files — Day 2's ONNX export must see byte-for-byte
the same Decoder module that produced decoder.pt, and any script pairing
an encoder checkpoint with a decoder checkpoint needs the exact classes
that trained them.

Decoder build: conv (kernel 3, stride 2, padding 1, ReLU) x N -> global
average pool -> linear to 32 logits. Layers are named conv1..convN and fc
so that state_dict keys stay stable and human-readable regardless of
depth; the proven shape is channels=(16, 32, 64), which yields
conv1/conv2/conv3/fc (the exact keys checkpointed in decoder.pt).

Encoder build: message-conditioned residual encoder, verbatim from the
Kaggle notebook that trained encoder.pt. msg (B, 32) --linear--> (B, 1024)
--reshape--> (B, 1, 32, 32), concatenated with the image as a 4th channel,
run through conv1/conv2 (ReLU) -> conv3, squashed through tanh and scaled
by 0.1 to bound the residual to [-0.1, 0.1], added to the image, then
clamped to the valid pixel range [0, 1].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, in_ch=3, channels=(16, 32, 64)):
        super().__init__()
        self.channels = list(channels)
        prev = in_ch
        for i, ch in enumerate(self.channels, start=1):
            setattr(self, f"conv{i}", nn.Conv2d(prev, ch, kernel_size=3, stride=2, padding=1))
            prev = ch
        self.fc = nn.Linear(prev, 32)

    def forward(self, x):
        for i in range(1, len(self.channels) + 1):
            x = F.relu(getattr(self, f"conv{i}")(x))
        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        return self.fc(x)


class Encoder(nn.Module):
    def __init__(self, msg_bits=32):
        super().__init__()
        self.msg_fc = nn.Linear(msg_bits, 32 * 32)
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=3, padding=1)

    def forward(self, img, msg):
        b = img.size(0)
        msg_map = self.msg_fc(msg).view(b, 1, 32, 32)
        x = torch.cat([img, msg_map], dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        residual = torch.tanh(self.conv3(x)) * 0.1
        watermarked = torch.clamp(img + residual, 0.0, 1.0)
        return watermarked
