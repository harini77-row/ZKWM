"""
ablation_study.py — measure the marginal cost/benefit of each pipeline stage
by running the same real photo crop and message through four baselines:

  A  watermark only              embed -> decode (plain PyTorch)
  B  watermark + ZK, no anchor   embed -> decode -> export -> prove -> verify
  C  watermark + anchor, no ZK   embed -> decode -> anchor placeholder hash on-chain
  D  full pipeline               embed -> decode -> export -> prove -> verify -> anchor

C anchors a placeholder proofHash (there is no ZK proof in that baseline, so
there is nothing real to hash) under the assetId of the watermarked image.
D runs its own fresh prove/verify (tag "D", separate from B's artifacts) and
anchors the real keccak256 of that proof's bytes. Both call chain.anchor_proof
against the live Anvil node from CLAUDE.md and immediately confirm the write
with chain.get_anchor before moving on. Because C and D embed the same crop
and message, they hash to the same assetId; D's anchor call overwrites C's
placeholder in that slot, which is correct mapping semantics (last write
wins) — each row's anchor is confirmed at the moment it's written, before
the next baseline can overwrite it.

All four rows use the SAME encoder.pt / decoder_v2.pt checkpoint pair, the
same real image crop, and the same seeded message, so the only thing that
differs row-to-row is which pipeline stages are switched on.

B and D's prove steps can run for a long time on CPU (see results.md's
existing EZKL rows). This script does not try to speed that up; correctness
and clean, incremental logging matter more than wall-clock time. Each
baseline's row is appended to results.md as soon as that baseline finishes,
so a partial run still leaves useful evidence.

Requires Anvil running at 127.0.0.1:8545 with ProvenanceAnchor deployed at
the address in chain.py.

Run:  python ablation_study.py
      python ablation_study.py --baselines A B    # skip the on-chain rows
"""

import argparse
import asyncio
import io
import json
import os
import sys
import time

# torch's dynamo ONNX exporter prints non-ASCII status glyphs; the default
# Windows console codec (cp1252) cannot encode them and the export crashes.
sys.stdout.reconfigure(encoding="utf-8")

import torch
import torch.nn.functional as F
import ezkl
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image
from web3 import Web3

from chain import anchor_proof, get_anchor, w3
from model import Decoder, Encoder
from probe import PeakRSS
from smoke_test import call

ENCODER_CKPT = "encoder.pt"
DECODER_CKPT = "decoder_v2.pt"
IMG_SIZE = 32
MSG_BITS = 32

RESULTS = "results.md"
ABLATION_HEADING = "## Ablation study"
ABLATION_HEADER = (
    "| baseline | description | BER | PSNR (dB) | logrows | peak RAM | "
    "prove time | proof size | verify | anchor |"
)
WORK = "run_ablation"
os.makedirs(WORK, exist_ok=True)


def P(name):
    return os.path.join(WORK, name)


def hx(b):
    return "0x" + b.hex()


def compute_asset_id(watermarked):
    """keccak256 of the watermarked image, PNG-encoded (see CLAUDE.md's
    Phase 0 note: assetId is a plain cryptographic hash of the watermarked
    image, not a perceptual hash)."""
    buf = io.BytesIO()
    to_pil_image(watermarked.clamp(0.0, 1.0)).save(buf, format="PNG")
    return bytes(Web3.keccak(buf.getvalue()))


def compute_proof_hash(proof_path):
    """keccak256 of the raw proof bytes, per CLAUDE.md's anchoring scheme."""
    with open(proof_path, "rb") as f:
        return bytes(Web3.keccak(f.read()))


def load_models(device="cpu"):
    encoder = Encoder(msg_bits=MSG_BITS).to(device)
    encoder.load_state_dict(torch.load(ENCODER_CKPT, map_location=device))
    encoder.eval()

    decoder = Decoder(in_ch=3, channels=[16, 32, 64]).to(device)
    decoder.load_state_dict(torch.load(DECODER_CKPT, map_location=device))
    decoder.eval()
    return encoder, decoder


def load_crop(image_path, box, seed):
    photo = Image.open(image_path).convert("RGB")
    crop = photo.crop(box)
    if crop.size != (IMG_SIZE, IMG_SIZE):
        raise ValueError(f"crop {box} is {crop.size}, expected {IMG_SIZE}x{IMG_SIZE}")
    crop_tensor = to_tensor(crop)  # [0, 1], (3, 32, 32)

    g = torch.Generator(device="cpu").manual_seed(seed)
    msg = torch.randint(0, 2, (1, MSG_BITS), generator=g, dtype=torch.float32)
    return crop_tensor, msg


def psnr(watermarked, original):
    mse = F.mse_loss(watermarked, original).item()
    if mse <= 1e-12:
        return 99.0
    return -10.0 * torch.log10(torch.tensor(mse)).item()  # pixels in [0, 1]


@torch.no_grad()
def embed_and_decode(encoder, decoder, crop_tensor, msg):
    img = crop_tensor.unsqueeze(0)
    watermarked = encoder(img, msg)
    logits = decoder(watermarked)
    bits = (logits > 0).float()
    ber = (bits != msg).float().mean().item()
    ps = psnr(watermarked.squeeze(0), crop_tensor)
    return watermarked.squeeze(0), ber, ps


def export_decoder(decoder, watermarked, model_path, data_path, dynamo):
    x = watermarked.unsqueeze(0)
    with torch.no_grad():
        y = decoder(x)
    torch.onnx.export(
        decoder,
        x,
        model_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamo=dynamo,
    )
    json.dump(
        {"input_data": [x.flatten().tolist()], "output_data": [y.flatten().tolist()]},
        open(data_path, "w"),
    )


async def run_zk(decoder, watermarked, tag):
    """embed's output -> ONNX -> gen_settings -> calibrate -> compile ->
    get_srs -> setup -> gen_witness -> prove -> verify, on the actual
    watermarked crop (not random noise)."""
    model_path = P(f"{tag}_network.onnx")
    data_path = P(f"{tag}_input.json")
    settings_path = P(f"{tag}_settings.json")
    compiled_path = P(f"{tag}_network.ezkl")
    vk_path = P(f"{tag}_vk.key")
    pk_path = P(f"{tag}_pk.key")
    witness_path = P(f"{tag}_witness.json")
    proof_path = P(f"{tag}_proof.json")

    def step(label, seconds):
        print(f"    {label}: {seconds:.2f}s")

    # dynamo exporter first; retry with the legacy path if EZKL cannot parse
    # the graph (see CLAUDE.md environment quirks).
    for dynamo in (True, False):
        try:
            export_decoder(decoder, watermarked, model_path, data_path, dynamo)

            t = time.time()
            await call(ezkl.gen_settings, model=model_path, output=settings_path)
            step("gen_settings", time.time() - t)

            t = time.time()
            await call(
                ezkl.calibrate_settings,
                data=data_path,
                model=model_path,
                settings=settings_path,
                settings_path=settings_path,
                target="resources",
            )
            step("calibrate_settings", time.time() - t)

            t = time.time()
            await call(
                ezkl.compile_circuit,
                model=model_path,
                compiled_circuit=compiled_path,
                settings_path=settings_path,
            )
            step("compile_circuit", time.time() - t)
            break
        except Exception as e:
            if dynamo:
                print(f"    dynamo export failed ({e}); retrying with dynamo=False")
                continue
            raise

    logrows = json.load(open(settings_path))["run_args"]["logrows"]
    print(f"    logrows: {logrows}")

    t = time.time()
    await call(ezkl.get_srs, settings_path=settings_path)
    step("get_srs", time.time() - t)

    t = time.time()
    await call(ezkl.setup, model=compiled_path, vk_path=vk_path, pk_path=pk_path)
    step("setup", time.time() - t)

    t = time.time()
    await call(ezkl.gen_witness, data=data_path, model=compiled_path, output=witness_path)
    step("gen_witness", time.time() - t)

    meter = PeakRSS()
    meter.start()
    t = time.time()
    await call(
        ezkl.prove,
        witness=witness_path,
        model=compiled_path,
        pk_path=pk_path,
        proof_path=proof_path,
    )
    prove_time = time.time() - t
    peak_rss = meter.stop()
    step("prove", prove_time)
    print(f"    peak RSS during prove: {peak_rss / 1e6:.0f} MB")

    proof_bytes = os.path.getsize(proof_path)

    t = time.time()
    ok = await call(
        ezkl.verify, proof_path=proof_path, settings_path=settings_path, vk_path=vk_path
    )
    step("verify", time.time() - t)

    return {
        "logrows": logrows,
        "peak_rss": peak_rss,
        "prove_time": prove_time,
        "proof_bytes": proof_bytes,
        "verify": ok,
    }


def ensure_ablation_header():
    text = open(RESULTS, encoding="utf-8").read() if os.path.exists(RESULTS) else ""
    if ABLATION_HEADING in text:
        return
    with open(RESULTS, "a", encoding="utf-8") as f:
        if text and not text.endswith("\n"):
            f.write("\n")
        f.write("\n" + ABLATION_HEADING + "\n\n")
        f.write(ABLATION_HEADER + "\n")
        f.write("|" + "---|" * 10 + "\n")


def append_ablation_row(
    baseline,
    description,
    ber=None,
    ps=None,
    logrows=None,
    peak_rss=None,
    prove_time=None,
    proof_bytes=None,
    verify=None,
    anchor_tx=None,
):
    ensure_ablation_header()

    cells = [
        baseline,
        description,
        f"{ber:.4f}",
        f"{ps:.2f}",
        str(logrows) if logrows is not None else "n/a",
        f"{peak_rss / 1e6:.0f} MB" if peak_rss is not None else "n/a",
        f"{prove_time:.2f}s" if prove_time is not None else "n/a",
        f"{proof_bytes} B" if proof_bytes is not None else "n/a",
        str(verify) if verify is not None else "n/a",
        anchor_tx if anchor_tx is not None else "n/a",
    ]
    row = "| " + " | ".join(cells) + " |"
    with open(RESULTS, "a", encoding="utf-8") as f:
        f.write(row + "\n")
    print(f"  logged: {row}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", default="tea_plantation.jpeg")
    ap.add_argument(
        "--box",
        type=int,
        nargs=4,
        default=[752, 905, 784, 937],
        metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"),
        help="pixel box for the 32x32 crop (default: the 'foliage' region "
        "used by test_real_image.py)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--baselines",
        nargs="+",
        default=["A", "B", "C", "D"],
        choices=["A", "B", "C", "D"],
    )
    args = ap.parse_args()

    encoder, decoder = load_models()
    crop_tensor, msg = load_crop(args.image, tuple(args.box), args.seed)
    print(f"image={args.image} box={args.box} seed={args.seed}")

    watermarked, ber, ps = embed_and_decode(encoder, decoder, crop_tensor, msg)
    print(f"embed+decode: BER={ber:.4f} PSNR={ps:.2f}dB")

    if "A" in args.baselines:
        print("=== Baseline A: watermark only (no ZK, no anchor) ===")
        append_ablation_row("A", "watermark only (no ZK, no anchor)", ber=ber, ps=ps)

    if "B" in args.baselines:
        print("=== Baseline B: watermark + ZK, no anchor ===")
        zk = asyncio.run(run_zk(decoder, watermarked, tag="B"))
        print(
            f"  VERIFY: {zk['verify']}  logrows={zk['logrows']}  "
            f"prove={zk['prove_time']:.2f}s  proof={zk['proof_bytes']}B"
        )
        append_ablation_row("B", "watermark + ZK, no anchor", ber=ber, ps=ps, **zk)

    if "C" in args.baselines:
        print("=== Baseline C: watermark + anchor, no ZK ===")
        asset_id = compute_asset_id(watermarked)
        proof_hash = bytes(Web3.keccak(b"baseline-C-placeholder-no-zk-proof"))
        print(f"  assetId={hx(asset_id)}")
        print(f"  proofHash={hx(proof_hash)} (placeholder -- no ZK proof exists in this baseline)")
        tx_hash = anchor_proof(asset_id, proof_hash)
        receipt = w3.eth.get_transaction_receipt(tx_hash)
        confirmed = get_anchor(asset_id) == proof_hash
        assert confirmed, "get_anchor did not return the anchored placeholder value"
        print(f"  tx={hx(tx_hash)}  gas={receipt.gasUsed}  confirmed={confirmed}")
        append_ablation_row(
            "C",
            "watermark + anchor, no ZK",
            ber=ber,
            ps=ps,
            anchor_tx=(
                f"tx={hx(tx_hash)} gas={receipt.gasUsed} "
                f"assetId={hx(asset_id)} proofHash={hx(proof_hash)} (placeholder, no ZK)"
            ),
        )

    if "D" in args.baselines:
        print("=== Baseline D: full pipeline (watermark + ZK + anchor) ===")
        zk = asyncio.run(run_zk(decoder, watermarked, tag="D"))
        print(
            f"  VERIFY: {zk['verify']}  logrows={zk['logrows']}  "
            f"prove={zk['prove_time']:.2f}s  proof={zk['proof_bytes']}B"
        )
        proof_hash = compute_proof_hash(P("D_proof.json"))
        asset_id = compute_asset_id(watermarked)
        print(f"  assetId={hx(asset_id)}")
        print(f"  proofHash={hx(proof_hash)}")
        tx_hash = anchor_proof(asset_id, proof_hash)
        receipt = w3.eth.get_transaction_receipt(tx_hash)
        confirmed = get_anchor(asset_id) == proof_hash
        assert confirmed, "get_anchor did not return the anchored proof hash"
        print(f"  tx={hx(tx_hash)}  gas={receipt.gasUsed}  confirmed={confirmed}")
        append_ablation_row(
            "D",
            "full pipeline (watermark + ZK + anchor)",
            ber=ber,
            ps=ps,
            **zk,
            anchor_tx=(
                f"tx={hx(tx_hash)} gas={receipt.gasUsed} "
                f"assetId={hx(asset_id)} proofHash={hx(proof_hash)}"
            ),
        )

    print("done.")


if __name__ == "__main__":
    main()
