# Project context

**Title:** Blockchain-Anchored Zero-Knowledge Verification of Neural Watermarks
for Publicly Auditable Media Provenance

Capstone project, panel review format. Working directory is `C:\Users\hrni0\zkwm`,
Windows, PowerShell, venv at `.venv`.

## Core contribution

A hash *derived from the ZK proof* is anchored on-chain — not the watermark, not
the proof itself. Prior work either exposes watermark data publicly on-chain or
uses ZK proofs without persistent auditability. Neither achieves both.

## Frozen decisions — do not revisit

1. **Hash-only anchoring.** The contract stores a proof-derived hash. Proof
   verification happens off-chain. No on-chain verifier contract.
2. **Anvil for development, Sepolia for recorded results.**
3. **32-bit watermark payload** (HiDDeN convention).

## Architecture in one line

Encoder CNN embeds 32 bits → decoder CNN extracts them → decoder runs inside an
EZKL circuit → proof generated → keccak256 of proof bytes anchored on-chain →
verifier verifies proof off-chain, recomputes hash, compares against anchor.

**Only the decoder enters the ZK circuit.** Its size is the binding constraint on
the whole architecture. The encoder is irrelevant to circuit budget — never size
work around it.

## Correct terminology

EZKL uses Halo2 with KZG commitments, which needs a universal SRS from the
Perpetual Powers of Tau ceremony.

- Correct: "universal, updatable SRS — no per-circuit ceremony"
- Wrong: "transparent setup", "no trusted ceremony"

This distinction matters against both the literature and the ZKROWNN patent.
Never write the wrong version into code comments, docs, or the paper.

## Environment quirks already discovered

- ezkl 23.0.5, Python 3.12, torch 2.14, onnx 1.22, onnxscript installed.
- **The ezkl CLI is not on PATH and is not needed.** Use the Python bindings.
- **ezkl function signatures drift between versions.** In 23.0.5,
  `prove(witness, model, pk_path, proof_path, srs_path)` — there is no
  `proof_type` parameter. Always call ezkl functions through the `call()` helper
  in `smoke_test.py`, which inspects the signature and filters kwargs. Do not
  hardcode argument lists.
- torch 2.14 uses the dynamo ONNX exporter. Export at `opset_version=18`.
  If a conv model fails to parse in EZKL, retry the export with `dynamo=False`.
- `forward pass failed ... decomposition error` lines during
  `calibrate_settings` are normal — EZKL sweeps 12 scale combinations and keeps
  the passing ones. Only worry if *all* combinations fail; that means the model
  is too wide for the field.
- SRS is cached after first download.

## Three-day plan

Target is a complete thin slice: every layer present and connected, small
numbers, honest limitations. One deep component with three missing layers reads
as unfinished; a working end-to-end path does not.

**Day 1** — Circuit budget and watermark baseline.
- Smoke test to `VERIFY: True` (trivial MLP).
- Probe CNN, decoder-shaped, random weights, no training. 32x32 RGB input,
  three conv layers 8/16/32 channels stride 2, global average pool, linear to
  32 logits. Log logrows, peak RAM, proving time, proof size.
- Two or three size variants maximum, then stop.
- Train encoder/decoder pair on whichever decoder shape proved. Small dataset
  of 32x32 crops. Loss = BCE on the 32-bit message + MSE on the image.
- Target: clean-channel BER near zero, one robustness point (JPEG or resize).

**Day 2** — Decoder into the circuit.
- Export trained decoder to ONNX, quantise, tune scale until in-circuit output
  matches PyTorch on the same input.
- Prove and verify. keccak256 the proof bytes to get the anchor value.
- This is where schedules slip. It owns the whole day.

**Day 3** — Chain, glue, evidence.
- Contract: `mapping(bytes32 => bytes32) anchors`, `setAnchor` with event,
  `getAnchor` view. Anvil first, then Sepolia for a real transaction hash.
- One script running the whole path start to finish, printing pass/fail.
- Ablation: Baseline A (watermark only) and Baseline D (full pipeline) minimum.
  B (watermark + ZK, no anchor) and C (watermark + anchor, no ZK) if time.

## Explicitly out of scope for these three days

Full robustness sweep, PSNR/SSIM tuning, on-chain proof verification, proof
batching, larger images, perceptual hashing for image binding. Each of these
goes in Limitations. Naming them honestly is a stronger position than a
half-finished attempt.

Open Phase 0 decision deferred: the public input binding proof to image is a
plain cryptographic hash of the watermarked image for now. Perceptual hashing is
the research direction, not the three-day build.

## Logging discipline

`results.md` is the evidence file. Append one row after **every** EZKL run:

| model shape | logrows | peak RAM | prove time | proof size | verify |

The Implementation and Experimental Setup section is written directly from this
file. Never report numbers only to the terminal.
