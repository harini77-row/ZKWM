"""demo_server.py -- local FastAPI demo server for the panel review.

Streams the full pipeline (crop -> embed -> diffmap -> decode -> zk prove ->
verify -> anchor on Sepolia) as Server-Sent Events over GET /run-demo, one
JSON event per completed stage, so demo.html can render a card per stage as
it lands.

The EZKL circuit (settings / compiled circuit / proving+verifying keys) is
compiled ONCE at server startup, not per request -- circuit compilation
depends only on the decoder architecture, not on which photo crop is being
proven, exactly like a real deployment would compile once and prove many
times. Only gen_witness + prove + verify run per demo request, so each
click still costs a real EZKL prove (~1min on CPU, see results.md) -- that
wait is the point of the demo, not something to hide.

Anchoring goes to Sepolia (this is what CLAUDE.md calls a "recorded
result"), so ZKWM_NETWORK defaults to "sepolia" here; SEPOLIA_RPC_URL and
SEPOLIA_PRIVATE_KEY must be set in the environment or a .env file before
starting the server (see chain.py -- never hardcode a real private key).

Run:  uvicorn demo_server:app
      then open http://127.0.0.1:8000/ in a browser.
"""

import asyncio
import base64
import io
import json
import os
import sys
import time
from contextlib import asynccontextmanager

sys.stdout.reconfigure(encoding="utf-8")

import ezkl
import torch
import torch.nn.functional as F
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from PIL import Image
from torchvision.transforms.functional import to_pil_image, to_tensor
from web3 import Web3

os.environ.setdefault("ZKWM_NETWORK", "sepolia")

from chain import CONTRACT_ADDRESS, NETWORK, anchor_proof_receipt, get_anchor
from model import Decoder, Encoder
from smoke_test import call

IMG_SIZE = 32
MSG_BITS = 32
ENCODER_CKPT = "encoder.pt"
DECODER_CKPT = "decoder_v2.pt"

WORK = "run_demo"
os.makedirs(WORK, exist_ok=True)


def P(name):
    return os.path.join(WORK, name)


# each crop: (source image, center x frac, center y frac).
# foliage / sky / sky_mountain_boundary are the three regions
# test_real_image.py already validated against tea_plantation.jpeg.
# fog_bank and rock_texture are deliberate contrasting cases on the two
# new photos: a smooth gray gradient (hard) vs. dense rock detail.
CROPS = {
    "foliage": ("tea_plantation.jpeg", 0.80, 0.72),
    "sky": ("tea_plantation.jpeg", 0.85, 0.05),
    "sky_mountain_boundary": ("tea_plantation.jpeg", 0.50, 0.35),
    "fog_bank": ("fog_image.jpeg", 0.50, 0.17),
    "rock_texture": ("sun_image.jpeg", 0.50, 0.83),
}


def crop_box(cx_frac, cy_frac, w, h, size=IMG_SIZE):
    cx, cy = cx_frac * w, cy_frac * h
    left = int(min(max(cx - size / 2, 0), w - size))
    top = int(min(max(cy - size / 2, 0), h - size))
    return left, top, left + size, top + size


def png_bytes(tensor):
    buf = io.BytesIO()
    to_pil_image(tensor.clamp(0.0, 1.0)).save(buf, format="PNG")
    return buf.getvalue()


def image_data_uri(tensor):
    b64 = base64.b64encode(png_bytes(tensor)).decode("ascii")
    return f"data:image/png;base64,{b64}"


def source_data_uri(pil_img, max_side=760, quality=82):
    """Downscaled JPEG data URI of a full source photo, for the crop-stage
    animation. The crop box is reported to the client in ORIGINAL pixel
    coordinates (see CROPS / crop_box), so the front end positions the
    highlight as a percentage and this thumbnail scaling is irrelevant."""
    im = pil_img.copy()
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def psnr(watermarked, original):
    mse = F.mse_loss(watermarked, original).item()
    if mse <= 1e-12:
        return 99.0
    return -10.0 * torch.log10(torch.tensor(mse)).item()


class ZKCircuit:
    """Compiled once at startup; every demo request reuses these paths."""

    def __init__(self):
        self.model_path = P("network.onnx")
        self.settings_path = P("settings.json")
        self.compiled_path = P("network.ezkl")
        self.vk_path = P("vk.key")
        self.pk_path = P("pk.key")
        self.logrows = None

    async def compile(self, decoder):
        x = torch.rand(1, 3, IMG_SIZE, IMG_SIZE)
        with torch.no_grad():
            y = decoder(x)
        data_path = P("calib_input.json")
        json.dump(
            {"input_data": [x.flatten().tolist()], "output_data": [y.flatten().tolist()]},
            open(data_path, "w"),
        )

        for dynamo in (True, False):
            try:
                torch.onnx.export(
                    decoder, x, self.model_path,
                    export_params=True, opset_version=18, do_constant_folding=True,
                    input_names=["input"], output_names=["output"], dynamo=dynamo,
                )
                await call(ezkl.gen_settings, model=self.model_path, output=self.settings_path)
                await call(
                    ezkl.calibrate_settings, data=data_path, model=self.model_path,
                    settings=self.settings_path, settings_path=self.settings_path,
                    target="resources",
                )
                await call(
                    ezkl.compile_circuit, model=self.model_path,
                    compiled_circuit=self.compiled_path, settings_path=self.settings_path,
                )
                break
            except Exception:
                if dynamo:
                    continue
                raise

        await call(ezkl.get_srs, settings_path=self.settings_path)
        await call(ezkl.setup, model=self.compiled_path, vk_path=self.vk_path, pk_path=self.pk_path)
        self.logrows = json.load(open(self.settings_path))["run_args"]["logrows"]


circuit = ZKCircuit()
encoder = None
decoder = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global encoder, decoder
    encoder = Encoder(msg_bits=MSG_BITS)
    encoder.load_state_dict(torch.load(ENCODER_CKPT, map_location="cpu"))
    encoder.eval()

    decoder = Decoder(in_ch=3, channels=[16, 32, 64])
    decoder.load_state_dict(torch.load(DECODER_CKPT, map_location="cpu"))
    decoder.eval()

    print("compiling EZKL circuit (one-time)...")
    await circuit.compile(decoder)
    print(f"circuit ready: logrows={circuit.logrows}  network={NETWORK}  contract={CONTRACT_ADDRESS}")
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
def index():
    return open("demo.html", encoding="utf-8").read()


@app.get("/crops")
def list_crops():
    return {"crops": list(CROPS.keys()), "network": NETWORK, "contract": CONTRACT_ADDRESS}


def sse(stage, status, data=None, message=None):
    payload = {"stage": stage, "status": status}
    if data is not None:
        payload["data"] = data
    if message is not None:
        payload["message"] = message
    return f"data: {json.dumps(payload)}\n\n"


async def run_pipeline(crop_name: str):
    if crop_name not in CROPS:
        yield sse("error", "error", message=f"unknown crop {crop_name!r}")
        return

    try:
        img_path, cx_frac, cy_frac = CROPS[crop_name]
        photo = Image.open(img_path).convert("RGB")
        w, h = photo.size
        box = crop_box(cx_frac, cy_frac, w, h)
        crop = photo.crop(box)
        crop_tensor = to_tensor(crop)

        yield sse("crop", "done", {
            "image": image_data_uri(crop_tensor),
            "source_image": source_data_uri(photo),
            "source_w": w,
            "source_h": h,
            "cx_frac": round(cx_frac, 4),
            "cy_frac": round(cy_frac, 4),
            "crop_name": crop_name,
            "box": list(box),
        })
        await asyncio.sleep(0)

        seed = abs(hash(crop_name)) % (2**32)
        g = torch.Generator(device="cpu").manual_seed(seed)
        msg = torch.randint(0, 2, (1, MSG_BITS), generator=g, dtype=torch.float32)

        with torch.no_grad():
            watermarked = encoder(crop_tensor.unsqueeze(0), msg).squeeze(0)
        yield sse("embed", "done", {
            "image": image_data_uri(watermarked),
            "original_image": image_data_uri(crop_tensor),
            "bits": [int(b) for b in msg.flatten().tolist()],
        })
        await asyncio.sleep(0)

        diff = (watermarked - crop_tensor).abs() * 10.0
        yield sse("diffmap", "done", {"image": image_data_uri(diff)})
        await asyncio.sleep(0)

        with torch.no_grad():
            logits = decoder(watermarked.unsqueeze(0))
        bits = (logits > 0).float()
        ber = (bits != msg).float().mean().item()
        ps = psnr(watermarked, crop_tensor)
        yield sse("decode", "done", {"ber": ber, "psnr": ps})
        await asyncio.sleep(0)

        # ---- ZK stage: witness + prove for THIS crop's watermarked image ----
        data_path = P(f"{crop_name}_input.json")
        witness_path = P(f"{crop_name}_witness.json")
        proof_path = P(f"{crop_name}_proof.json")
        x = watermarked.unsqueeze(0)
        with torch.no_grad():
            y = decoder(x)
        json.dump(
            {"input_data": [x.flatten().tolist()], "output_data": [y.flatten().tolist()]},
            open(data_path, "w"),
        )

        t0 = time.time()
        await call(ezkl.gen_witness, data=data_path, model=circuit.compiled_path, output=witness_path)
        await call(
            ezkl.prove, witness=witness_path, model=circuit.compiled_path,
            pk_path=circuit.pk_path, proof_path=proof_path,
        )
        prove_time = time.time() - t0
        proof_bytes = os.path.getsize(proof_path)

        yield sse("zk", "done", {
            "logrows": circuit.logrows,
            "prove_time": round(prove_time, 2),
            "proof_bytes": proof_bytes,
        })
        await asyncio.sleep(0)

        t0 = time.time()
        ok = await call(
            ezkl.verify, proof_path=proof_path,
            settings_path=circuit.settings_path, vk_path=circuit.vk_path,
        )
        verify_time = time.time() - t0
        yield sse("verify", "done", {"verify": bool(ok), "verify_time": round(verify_time, 2)})
        await asyncio.sleep(0)

        if not ok:
            yield sse("anchor", "error", message="proof failed verification; refusing to anchor")
            return

        asset_id = bytes(Web3.keccak(png_bytes(watermarked)))
        with open(proof_path, "rb") as f:
            proof_hash = bytes(Web3.keccak(f.read()))

        rcpt = anchor_proof_receipt(asset_id, proof_hash)
        confirmed = get_anchor(asset_id) == proof_hash
        tx_hex = rcpt["tx_hash"]
        etherscan_url = f"https://sepolia.etherscan.io/tx/{tx_hex}" if NETWORK == "sepolia" else None

        yield sse("anchor", "done", {
            "network": NETWORK,
            "contract": CONTRACT_ADDRESS,
            "tx_hash": tx_hex,
            "etherscan_url": etherscan_url,
            "block_number": rcpt["block_number"],
            "block_hash": rcpt["block_hash"],
            "gas_used": rcpt["gas_used"],
            "asset_id": "0x" + asset_id.hex(),
            "proof_hash": "0x" + proof_hash.hex(),
            "confirmed": confirmed,
        })

        yield sse("complete", "done")

    except Exception as e:
        yield sse("error", "error", message=str(e))


@app.get("/run-demo")
def run_demo(crop: str = "foliage"):
    return StreamingResponse(
        run_pipeline(crop),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
