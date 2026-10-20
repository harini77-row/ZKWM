"""
EZKL smoke test - trivial MLP, full pipeline.
Goal: prove the toolchain works end to end before building the probe CNN.
Run:  python smoke_test.py
"""

import asyncio
import inspect
import json
import os
import sys
import time

# torch's dynamo ONNX exporter prints non-ASCII status glyphs; the default
# Windows console codec (cp1252) cannot encode them and the export crashes.
sys.stdout.reconfigure(encoding="utf-8")

import torch
import torch.nn as nn
import ezkl

WORK = "run_smoke"
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


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 4)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def export_model():
    model = TinyMLP()
    model.eval()
    x = torch.randn(1, 8)

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
    )

    json.dump(
        {
            "input_data": [x.flatten().tolist()],
            "output_data": [y.flatten().tolist()],
        },
        open(DATA, "w"),
    )
    print(f"exported {MODEL}")


async def main():
    export_model()

    def step(label, seconds):
        print(f"  {label}: {seconds:.2f}s")

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

    logrows = json.load(open(SETTINGS))["run_args"]["logrows"]
    print(f"  logrows: {logrows}")

    t = time.time()
    await call(
        ezkl.compile_circuit,
        model=MODEL,
        compiled_circuit=COMPILED,
        settings_path=SETTINGS,
    )
    step("compile_circuit", time.time() - t)

    t = time.time()
    await call(ezkl.get_srs, settings_path=SETTINGS)
    step("get_srs", time.time() - t)

    t = time.time()
    await call(ezkl.setup, model=COMPILED, vk_path=VK, pk_path=PK)
    step("setup", time.time() - t)

    t = time.time()
    await call(
        ezkl.gen_witness,
        data=DATA,
        model=COMPILED,
        output=WITNESS,
    )
    step("gen_witness", time.time() - t)

    t = time.time()
    await call(
        ezkl.prove,
        witness=WITNESS,
        model=COMPILED,
        pk_path=PK,
        proof_path=PROOF,
    )
    prove_time = time.time() - t
    step("prove", prove_time)

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
    print(f"logrows={logrows}  prove={prove_time:.2f}s  proof={proof_bytes}B")
    print("=" * 46)


if __name__ == "__main__":
    asyncio.run(main())