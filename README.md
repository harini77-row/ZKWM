# ZKWM — Blockchain-Anchored Zero-Knowledge Verification of Neural Watermarks

Capstone project (VIT SCOPE, BCSE497J Project-I) for publicly auditable, privacy-preserving media provenance.

## Overview

ZKWM lets a publisher prove that an image contains a specific embedded watermark **without revealing the watermark itself**, and anchor that proof publicly and immutably on-chain.

The core novelty: instead of anchoring the watermark or the ZK proof on-chain, we anchor a `keccak256` hash *derived from* the proof. This gives persistent public auditability while keeping the watermark confidential — no prior work combines neural watermarking, ZK proving, and blockchain anchoring in this configuration.

## Architecture

The system has three layers:

1. **Neural watermarking** — a HiDDeN-convention CNN encoder/decoder pair, trained on CIFAR-10, embedding a 32-bit publisher payload into an image.
2. **ZK proving** — the decoder is compiled into a Halo2/PLONKish circuit via [EZKL](https://github.com/zkonduit/ezkl), using KZG commitments and a universal, updatable SRS (no per-circuit trusted setup ceremony).
3. **Blockchain anchoring** — `ProvenanceAnchor.sol`, a Solidity contract deployed on Sepolia testnet, stores and retrieves the proof-derived hash via `setAnchor` / `getAnchor`.

```
image → CNN decoder → ZK proof (EZKL/Halo2) → keccak256(proof) → on-chain anchor (Sepolia)
```

## Results

| Metric | Value |
|---|---|
| Bit error rate (BER) | 0.0000 |
| PSNR | 30.93 dB (CIFAR-10, 15 epochs) |
| Proving time | ~56s at logrows 20 |
| Deployed contract | `0x4Dda995CD649B203b828a97d0b571259923895eF` (Sepolia) |

## Repository structure

```
zkwm/
├── contracts/        # Foundry project — ProvenanceAnchor.sol, tests, scripts
├── model.py           # Canonical encoder/decoder model classes
├── demo_server.py      # FastAPI + SSE demo backend
├── demo.html            # Demo frontend
├── results.md             # EZKL run log (one row per run)
├── train_log.md            # Training log (Kaggle)
└── ...
```

## Stack

- **ML**: PyTorch, CIFAR-10, ONNX export (opset 18)
- **ZK**: EZKL 23.0.5, Halo2/PLONKish, KZG commitments, Perpetual Powers of Tau SRS
- **Blockchain**: Foundry (forge/anvil/cast), web3.py, Sepolia testnet
- **Demo**: FastAPI, Server-Sent Events, Uvicorn

## Setup

```bash
# Python environment
pip install -r requirements.txt   # torch, ezkl, web3, fastapi, uvicorn

# Contracts (WSL)
cd contracts
forge install
forge build
forge test

# Run local chain for testing
anvil
```

## Running the demo

```bash
uvicorn demo_server:app --reload
```

Then open `demo.html` in a browser. The demo walks through watermark embedding, proof generation, and on-chain anchoring against a local Anvil node or Sepolia.

## Status

Currently in Review-2 preparation for the VIT SCOPE capstone evaluation. See `results.md` for the latest verified EZKL runs.

## Team

3-person capstone team, VIT Vellore.

## License

TBD.
