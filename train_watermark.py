"""
train_watermark.py — train an encoder / decoder watermark pair.

Decoder: model.Decoder(in_ch=3, channels=[16, 32, 64]), imported
verbatim from model.py — the exact shape already proven in EZKL at
logrows 20. Day 2 exports THIS decoder to ONNX, so it must not drift;
that is why the class is imported, never re-declared here.

Encoder: model.Encoder, also imported verbatim. 32x32x3 image + random
32-bit message -> watermarked image. The message is linear-expanded to a
32x32 map, concatenated as a 4th channel, run through a small conv stack;
the final conv is squashed through tanh and scaled by 0.1 to bound the
residual to [-0.1, 0.1] before it's added to the image and clamped to the
valid pixel range [0, 1]. test_real_image.py loads this same class against
encoder.pt, so it must not drift either.

Dataset: torchvision CIFAR-10 (labels ignored), pixels in [0, 1] to
match the encoder's clamp range.

Loss: BCEWithLogits(decoder logits, true bits) * w_msg
      + MSE(watermarked, original) * w_img          (w_msg >> w_img)

Outputs:
  decoder_v2.pt  decoder state dict from this run
  encoder.pt     encoder state dict from this run
  train_log.md   one row per epoch: epoch, msg loss, img loss, BER, PSNR

Named decoder_v2.pt, not decoder.pt: results.md already has a proof recorded
against the original decoder.pt (see day2_export.py), so this script must
never overwrite it. Re-run day2_export.py against decoder_v2.pt explicitly
once you want to prove this new checkpoint.

Run:  python train_watermark.py --epochs 5
"""

import argparse
import math
import os
import sys
import time

# torchvision's ONNX-op registration prints non-ASCII glyphs on import;
# cp1252 stdout on Windows cannot encode them.
sys.stdout.reconfigure(encoding="utf-8")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Import the proven decoder shape and the encoder directly. Do NOT copy
# either — Day 2's ONNX export and test_real_image.py must see
# byte-for-byte the same modules.
from model import Decoder, Encoder

MSG_BITS = 32
DECODER_CKPT = "decoder_v2.pt"
ENCODER_CKPT = "encoder.pt"
LOG = "train_log.md"
LOG_HEADER = "| epoch | msg loss | img loss | BER | PSNR (dB) |"


def random_messages(b, device):
    return torch.randint(0, 2, (b, MSG_BITS), device=device, dtype=torch.float32)


def psnr(watermarked, original):
    mse = F.mse_loss(watermarked, original).item()
    if mse <= 1e-12:
        return 99.0
    return -10.0 * math.log10(mse)  # pixels in [0, 1] so MAX_I = 1


@torch.no_grad()
def evaluate(encoder, decoder, eval_imgs, eval_msgs):
    encoder.eval()
    decoder.eval()
    wm = encoder(eval_imgs, eval_msgs)
    logits = decoder(wm)
    bits = (logits > 0).float()
    ber = (bits != eval_msgs).float().mean().item()
    return ber, psnr(wm, eval_imgs)


def append_log(epoch, msg_loss, img_loss, ber, ps):
    new_file = not os.path.exists(LOG)
    with open(LOG, "a", encoding="utf-8") as f:
        if new_file:
            f.write("# Training log\n\n")
            f.write(LOG_HEADER + "\n")
            f.write("|" + "---|" * 5 + "\n")
        f.write(
            f"| {epoch} | {msg_loss:.4f} | {img_loss:.6f} | "
            f"{ber:.4f} | {ps:.2f} |\n"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--w-msg", type=float, default=1.0)
    ap.add_argument("--w-img", type=float, default=0.1)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--limit-batches", type=int, default=0,
                    help="cap training batches per epoch (0 = full epoch)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}  batch_size: {args.batch_size}  "
          f"w_msg/w_img: {args.w_msg}/{args.w_img}")

    tf = transforms.ToTensor()  # PIL uint8 -> float32 (3, 32, 32) in [0, 1]
    train_set = datasets.CIFAR10(args.data_root, train=True, download=True,
                                 transform=tf)
    test_set = datasets.CIFAR10(args.data_root, train=False, download=True,
                                transform=tf)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, drop_last=True)

    # fixed held-out batch: same images and messages every epoch
    eval_loader = DataLoader(test_set, batch_size=256, shuffle=False,
                             num_workers=0)
    eval_imgs = next(iter(eval_loader))[0].to(device)
    torch.manual_seed(0)
    eval_msgs = random_messages(eval_imgs.shape[0], device)

    encoder = Encoder().to(device)
    decoder = Decoder(in_ch=3, channels=[16, 32, 64]).to(device)
    print(f"encoder params: {sum(p.numel() for p in encoder.parameters()):,}  "
          f"decoder params: {sum(p.numel() for p in decoder.parameters()):,}")

    opt = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr
    )
    bce = nn.BCEWithLogitsLoss()

    for epoch in range(1, args.epochs + 1):
        encoder.train()
        decoder.train()
        t0 = time.time()
        run_msg = run_img = 0.0
        n = 0
        for i, (imgs, _) in enumerate(train_loader):
            if args.limit_batches and i >= args.limit_batches:
                break
            imgs = imgs.to(device)
            msgs = random_messages(imgs.shape[0], device)

            wm = encoder(imgs, msgs)
            logits = decoder(wm)

            msg_loss = bce(logits, msgs)
            img_loss = F.mse_loss(wm, imgs)
            loss = args.w_msg * msg_loss + args.w_img * img_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            run_msg += msg_loss.item()
            run_img += img_loss.item()
            n += 1

        avg_msg = run_msg / max(n, 1)
        avg_img = run_img / max(n, 1)
        ber, ps = evaluate(encoder, decoder, eval_imgs, eval_msgs)
        print(f"epoch {epoch}/{args.epochs}  msg_loss={avg_msg:.4f}  "
              f"img_loss={avg_img:.6f}  BER={ber:.4f}  PSNR={ps:.2f}dB  "
              f"({time.time() - t0:.1f}s, {n} batches)")
        append_log(epoch, avg_msg, avg_img, ber, ps)
        torch.save(decoder.state_dict(), DECODER_CKPT)
        torch.save(encoder.state_dict(), ENCODER_CKPT)

    print(f"saved {DECODER_CKPT}  (decoder state dict, {os.path.getsize(DECODER_CKPT)} bytes)")
    print(f"saved {ENCODER_CKPT}  (encoder state dict, {os.path.getsize(ENCODER_CKPT)} bytes)")


if __name__ == "__main__":
    main()
