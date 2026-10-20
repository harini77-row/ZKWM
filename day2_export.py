"""
Day 2 - export the trained decoder and push it through the full EZKL pipeline.

Loads decoder.pt (state dict trained in train_watermark.py: model.Decoder(in_ch=3,
channels=[16, 32, 64]) - the shape already proven in EZKL at logrows 20 in
probe.py). Exports to ONNX at opset 18, then runs gen_settings ->
calibrate_settings -> compile_circuit -> get_srs -> setup -> gen_witness ->
prove -> verify through the same call() helper used in smoke_test.py.

Run:  python day2_export.py
"""

import asyncio
import json
import os
import sys
import time

# torch's dynamo ONNX exporter prints non-ASCII status glyphs; the default
# Windows console codec (cp1252) cannot encode them and the export crashes.
sys.stdout.reconfigure(encoding="utf-8")

import torch
import ezkl

from model import Decoder
from probe import PeakRSS, append_row
from smoke_test import call

WORK = "run_day2"
os.makedirs(WORK, exist_ok=True)


def P(name):
    return os.path.join(WORK, name)


CKPT = "decoder.pt"
MODEL = P("network.onnx")
DATA = P("input.json")
SETTINGS = P("settings.json")
COMPILED = P("network.ezkl")
VK = P("vk.key")
PK = P("pk.key")
WITNESS = P("witness.json")
PROOF = P("proof.json")

IN_CH = 3
IMG_SIZE = 32
CHANNELS = [16, 32, 64]


def load_decoder():
    model = Decoder(in_ch=IN_CH, channels=CHANNELS)
    state = torch.load(CKPT, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def export_model(model, dynamo):
    # pixels in [0, 1], matching the ToTensor() range the decoder was trained on.
    x = torch.rand(1, IN_CH, IMG_SIZE, IMG_SIZE)
    with torch.no_grad():
        y = model(x)

    torch.onnx.export(
        model,
        x,
        MODEL,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamo=dynamo,
    )

    json.dump(
        {
            "input_data": [x.flatten().tolist()],
            "output_data": [y.flatten().tolist()],
        },
        open(DATA, "w"),
    )
    print(f"exported {MODEL} (dynamo={dynamo})")


async def run():
    shape = (
        f"trained decoder (real weights) {IMG_SIZE}x{IMG_SIZE}x{IN_CH}, "
        f"conv {'/'.join(map(str, CHANNELS))} s2, GAP, linear 32"
    )
    model = load_decoder()

    def step(label, seconds):
        print(f"  {label}: {seconds:.2f}s")

    for dynamo in (True, False):
        try:
            export_model(model, dynamo)

            t = time.time()
            await call(ezkl.gen_settings, model=MODEL, output=SETTINGS)
            step("gen_settings", time.time() - t)

            t = time.time()
            await call(
                ezkl.calibrate_settings,
                data=DATA,
                model=MODEL,
                settings=SETTINGS,
                settings_path=SETTINGS,
                target="resources",
            )
            step("calibrate_settings", time.time() - t)

            t = time.time()
            await call(
                ezkl.compile_circuit,
                model=MODEL,
                compiled_circuit=COMPILED,
                settings_path=SETTINGS,
            )
            step("compile_circuit", time.time() - t)
            break
        except Exception as e:
            if dynamo:
                print(f"  dynamo path failed ({e}); retrying with dynamo=False")
                continue
            raise

    logrows = json.load(open(SETTINGS))["run_args"]["logrows"]
    print(f"  logrows: {logrows}")

    t = time.time()
    await call(ezkl.get_srs, settings_path=SETTINGS)
    step("get_srs", time.time() - t)

    t = time.time()
    await call(ezkl.setup, model=COMPILED, vk_path=VK, pk_path=PK)
    step("setup", time.time() - t)

    t = time.time()
    await call(ezkl.gen_witness, data=DATA, model=COMPILED, output=WITNESS)
    step("gen_witness", time.time() - t)

    meter = PeakRSS()
    meter.start()
    t = time.time()
    await call(
        ezkl.prove,
        witness=WITNESS,
        model=COMPILED,
        pk_path=PK,
        proof_path=PROOF,
    )
    prove_time = time.time() - t
    peak_rss = meter.stop()
    step("prove", prove_time)
    print(f"  peak RSS during prove: {peak_rss / 1e6:.0f} MB")

    proof_bytes = os.path.getsize(PROOF)
    print(f"  proof size: {proof_bytes} bytes")

    t = time.time()
    ok = await call(
        ezkl.verify,
        proof_path=PROOF,
        settings_path=SETTINGS,
        vk_path=VK,
    )
    step("verify", time.time() - t)

    print()
    print("=" * 46)
    print(f"VERIFY: {ok}")
    print(
        f"logrows={logrows}  prove={prove_time:.2f}s  "
        f"proof={proof_bytes}B  peakRSS={peak_rss / 1e6:.0f}MB"
    )
    print("=" * 46)

    append_row(shape, logrows, peak_rss, prove_time, proof_bytes, ok)


if __name__ == "__main__":
    asyncio.run(run())
