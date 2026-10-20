"""
test_real_image.py — sanity-check the trained watermark pipeline against a
real photograph, not just CIFAR-10 crops.

Loads tea_plantation.jpeg, cuts three 32x32 crops from named regions (sky,
the sky/mountain boundary, foliage), embeds a random 32-bit message in each
with model.Encoder (encoder.pt), decodes with model.Decoder (decoder_v2.pt),
and reports BER per crop.

decoder_v2.pt, not decoder.pt: decoder.pt is the checkpoint already proven
through EZKL (see results.md / day2_export.py) and must not be touched.
decoder_v2.pt and encoder.pt come from the same train_watermark.py run, so
they are the only pair of checkpoints that were ever trained together.

Crop centers are given as fractions of (width, height), chosen by eye from
the source photo, so they hold regardless of image size. If a region
doesn't match its label, check the saved original_<name>.png files and
adjust CROPS below.

Outputs (under --out-dir, default real_image_test/):
  original_<name>.png      the untouched 32x32 crop
  watermarked_<name>.png   the crop after encoding
  diff_<name>.png          |watermarked - original| * 10, clipped to [0, 1]
  report.md                per-crop BER and message bits

Run:  python test_real_image.py
"""

import argparse
import os
import sys

# torchvision's ONNX-op registration prints non-ASCII glyphs on import;
# cp1252 stdout on Windows cannot encode them.
sys.stdout.reconfigure(encoding="utf-8")

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image

from model import Decoder, Encoder

DECODER_CKPT = "decoder_v2.pt"
ENCODER_CKPT = "encoder.pt"
IMG_SIZE = 32
MSG_BITS = 32

# (center_x_fraction, center_y_fraction) of the source image.
CROPS = {
    "sky": (0.85, 0.05),
    "sky_mountain_boundary": (0.50, 0.35),
    "foliage": (0.80, 0.72),
}


def crop_box(cx_frac, cy_frac, w, h, size=IMG_SIZE):
    cx, cy = cx_frac * w, cy_frac * h
    left = int(min(max(cx - size / 2, 0), w - size))
    top = int(min(max(cy - size / 2, 0), h - size))
    return left, top, left + size, top + size


def load_models(device):
    encoder = Encoder(msg_bits=MSG_BITS).to(device)
    encoder.load_state_dict(torch.load(ENCODER_CKPT, map_location=device))
    encoder.eval()

    decoder = Decoder(in_ch=3, channels=[16, 32, 64]).to(device)
    decoder.load_state_dict(torch.load(DECODER_CKPT, map_location=device))
    decoder.eval()
    return encoder, decoder


def random_message(seed, device):
    g = torch.Generator(device="cpu").manual_seed(seed)
    msg = torch.randint(0, 2, (1, MSG_BITS), generator=g, dtype=torch.float32)
    return msg.to(device)


@torch.no_grad()
def run_crop(encoder, decoder, crop_tensor, msg, device):
    img = crop_tensor.unsqueeze(0).to(device)
    watermarked = encoder(img, msg)
    logits = decoder(watermarked)
    bits = (logits > 0).float()
    ber = (bits != msg).float().mean().item()
    return watermarked.squeeze(0).cpu(), ber


def append_report(path, rows):
    new_file = not os.path.exists(path)
    with open(path, "a", encoding="utf-8") as f:
        if new_file:
            f.write("# Real-image watermark test\n\n")
            f.write("| crop | BER | message (32 bits) |\n")
            f.write("|---|---|---|\n")
        for name, ber, msg_bits in rows:
            bits_str = "".join(str(b) for b in msg_bits)
            f.write(f"| {name} | {ber:.4f} | {bits_str} |\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", default="tea_plantation.jpeg")
    ap.add_argument("--out-dir", default="real_image_test")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    encoder, decoder = load_models(device)

    photo = Image.open(args.image).convert("RGB")
    w, h = photo.size
    print(f"loaded {args.image} ({w}x{h})  device={device}")

    report_rows = []
    for i, (name, (cx, cy)) in enumerate(CROPS.items()):
        box = crop_box(cx, cy, w, h)
        crop = photo.crop(box)
        crop_tensor = to_tensor(crop)  # [0, 1], (3, 32, 32)

        msg = random_message(args.seed + i, device)
        watermarked, ber = run_crop(encoder, decoder, crop_tensor, msg, device)

        diff = (watermarked - crop_tensor).abs() * 10.0
        diff = diff.clamp(0.0, 1.0)

        to_pil_image(crop_tensor).save(os.path.join(args.out_dir, f"original_{name}.png"))
        to_pil_image(watermarked).save(os.path.join(args.out_dir, f"watermarked_{name}.png"))
        to_pil_image(diff).save(os.path.join(args.out_dir, f"diff_{name}.png"))

        msg_bits = msg.squeeze(0).int().tolist()
        report_rows.append((name, ber, msg_bits))
        print(f"  {name:24s} box={box}  BER={ber:.4f}")

    append_report(os.path.join(args.out_dir, "report.md"), report_rows)
    print(f"saved crops and report to {args.out_dir}/")


if __name__ == "__main__":
    main()
