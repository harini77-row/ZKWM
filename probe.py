"""
Probe CNN through the full EZKL pipeline.

Decoder-shaped, random weights, no training. The point is a circuit-budget
reading: logrows, peak RSS during prove, prove time, proof size, verify.

Build:  conv (stride 2, ReLU) x N  ->  global average pool  ->  linear to 32 logits
Only the decoder shape matters for the circuit budget (see CLAUDE.md).

Run:
    python probe.py                              # 32x32 RGB, channels 8/16/32
    python probe.py --input-size 32 --channels 8 16 32
    python probe.py --channels 8 16 32 --layers 4   # pads last width if layers > widths
"""

import argparse
import asyncio
import inspect
import json
import os
import sys
import threading
import time

# torch's dynamo ONNX exporter prints non-ASCII status glyphs; the default
# Windows console codec (cp1252) cannot encode them and the export crashes.
sys.stdout.reconfigure(encoding="utf-8")

import psutil
import torch
import ezkl

from model import Decoder

WORK = "run_probe"
RESULTS = "results.md"
RESULTS_HEADER = (
    "| model shape | logrows | peak RAM | prove time | proof size | verify |"
)
os.makedirs(WORK, exist_ok=True)


def P(name):
    return os.path.join(WORK, name)


MODEL = P("network.onnx")
DATA = P("input.json")
SETTINGS = P("settings.json")
COMPILED = P("network.ezkl")
VK = P("vk.key")
PK = P("pk.key")
WITNESS = P("witness.json")
PROOF = P("proof.json")


async def call(fn, **kwargs):
    """
    Call an ezkl function with only the keyword arguments its installed
    version actually accepts, and await the result if it is a coroutine.
    Argument names drift between ezkl releases, so this avoids hardcoding them.
    """
    try:
        accepted = set(inspect.signature(fn).parameters)
        kwargs = {k: v for k, v in kwargs.items() if k in accepted}
    except (ValueError, TypeError):
        pass

    result = fn(**kwargs)
    if inspect.isawaitable(result):
        result = await result
    return result


class PeakRSS(threading.Thread):
    """Poll this process's RSS on a background thread and keep the maximum."""

    def __init__(self, interval=0.02):
        super().__init__(daemon=True)
        self.interval = interval
        self.peak = 0
        self._proc = psutil.Process()
        # not _stop: threading.Thread uses that name internally.
        self._done = threading.Event()

    def run(self):
        while not self._done.is_set():
            try:
                rss = self._proc.memory_info().rss
            except psutil.Error:
                break
            if rss > self.peak:
                self.peak = rss
            self._done.wait(self.interval)

    def stop(self):
        self._done.set()
        self.join()
        return self.peak


def export_model(model, size, in_ch, dynamo):
    model.eval()
    x = torch.randn(1, in_ch, size, size)
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


def append_row(shape, logrows, peak_rss, prove_time, proof_bytes, verify):
    new_file = not os.path.exists(RESULTS)
    with open(RESULTS, "a", encoding="utf-8") as f:
        if new_file:
            f.write("# Results\n\n")
            f.write(RESULTS_HEADER + "\n")
            f.write("|" + "---|" * 6 + "\n")
        f.write(
            f"| {shape} | {logrows} | {peak_rss / 1e6:.0f} MB | "
            f"{prove_time:.2f}s | {proof_bytes} B | {verify} |\n"
        )
    print(f"appended row to {RESULTS}")


async def run(size, in_ch, channels):
    shape = (
        f"probe CNN {size}x{size}x{in_ch}, conv {'/'.join(map(str, channels))} "
        f"s2, GAP, linear 32"
    )
    model = Decoder(in_ch, channels)

    def step(label, seconds):
        print(f"  {label}: {seconds:.2f}s")

    # dynamo exporter first; fall back to the legacy path if EZKL cannot parse
    # the graph (CLAUDE.md: retry conv exports with dynamo=False).
    for dynamo in (True, False):
        try:
            export_model(model, size, in_ch, dynamo)

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


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-size", type=int, default=32,
                   help="square input edge in pixels (default 32)")
    p.add_argument("--in-channels", type=int, default=3,
                   help="input channels (default 3, RGB)")
    p.add_argument("--channels", type=int, nargs="+", default=[8, 16, 32],
                   help="conv block output widths (default 8 16 32)")
    p.add_argument("--layers", type=int, default=None,
                   help="conv block count; truncates or pads --channels "
                        "with its last width to this length")
    return p.parse_args()


def main():
    args = parse_args()
    channels = list(args.channels)
    if args.layers is not None:
        if args.layers <= len(channels):
            channels = channels[: args.layers]
        else:
            channels += [channels[-1]] * (args.layers - len(channels))
    asyncio.run(run(args.input_size, args.in_channels, channels))


if __name__ == "__main__":
    main()
