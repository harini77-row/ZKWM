# Progress

Status: **all three days complete.** Full evidence in `results.md`. This file
is the short summary; `results.md` is authoritative for exact numbers.

Ablation study (`ablation_study.py`, full rows in `results.md` under
"## Ablation study"). All four baselines use the same encoder.pt /
decoder_v2.pt checkpoint pair, the same real photo crop (tea_plantation.jpeg,
foliage region), and the same seeded message — the only variable is which
pipeline stages are switched on.

## Ablation — all four baselines complete

| baseline | description | BER | PSNR | logrows | peak RAM | prove time | proof size | verify | anchor |
|---|---|---|---|---|---|---|---|---|---|
| A | watermark only (no ZK, no anchor) | 0.0000 | 30.23 dB | n/a | n/a | n/a | n/a | n/a | n/a |
| B | watermark + ZK, no anchor | 0.0000 | 30.23 dB | 20 | 308 MB | 48.98s | 491871 B | True | n/a |
| C | watermark + anchor, no ZK | 0.0000 | 30.23 dB | n/a | n/a | n/a | n/a | n/a | tx=0x7edfa63c6ee044465ec5d931723f8310b416fe8eabf7bedf928311f5feacddb3 gas=46392 (placeholder proofHash, no ZK) |
| D | full pipeline (watermark + ZK + anchor) | 0.0000 | 30.23 dB | 20 | 379 MB | 54.27s | 491890 B | True | tx=0x31b673e09839b0e57d2e2a9c71b728c675c0b40bb8943e03653145e3ce1d4bb0 gas=29292 |

- A: `model.Encoder` embeds the message into the crop, `model.Decoder`
  decodes it in plain PyTorch — no circuit involved.
- B: same embed, decoder exported to ONNX and run through the full EZKL
  pipeline (gen_settings → calibrate_settings → compile_circuit → get_srs →
  setup → gen_witness → prove → verify) against the actual watermarked crop.
  `VERIFY: True`.
- C: same embed and decode, plus a real on-chain `setAnchor` — but the
  anchored value is a placeholder hash, not derived from a ZK proof.
- D: the complete contribution — watermark + off-chain ZK proof + keccak256
  of the proof bytes anchored on-chain. `VERIFY: True`, anchor confirmed.
  assetId and proofHash for C and D share the same crop
  (0xf7db211e86d08e54bc1895dc829104037b9e03292b71410bef099dd0aea306ba);
  D proofHash = 0xf6bc69076978bb3c52ed3cd0701b6b0144dcd8384dc03327b2eb81946dec2d2c.

## On-chain anchoring — done on Anvil and Sepolia

Anvil (local dev, `chain.py` against a live node): manual anchor test,
assetId 0x18f7…be21, proofHash 0x3a41…e4a0, contract
0x5FbDB2315678afecb367f032d93F642f64180aa3, tx 0x7477…a7a, block 3,
gas 46392, `getAnchor` round-trip confirmed.

Sepolia (recorded result):

| field | value |
|---|---|
| Contract | 0x4Dda995CD649B203b828a97d0b571259923895eF |
| Deploy tx | 0x3a40613175c85ee90e48e7a228add020efbf1a5cb0f478d90c40a4796bd7bb0a |
| setAnchor tx | 0x17093ad77a5ef49d2284860b04ba3ad7e0053e2d7c600c1976befcc0907ebb33 |
| Block | 11638655 |
| Gas used | 46392 |
| Status | success |
| assetId | 0x18f79853237ed3e05be211b4a77ec3f56423339710fe49a089b7bd8c14f9be21 |
| proofHash | 0x3a41b30b42bdfc92dda6e1fe9332c9c4b342c5fb12714ab6bf13a8525724e4a0 |
| getAnchor round-trip | confirmed — returned the exact proofHash |

setAnchor gas is identical on Anvil and Sepolia (46392), confirming
portability from local dev to public testnet.

## Demo — built and dry-run tested

- `demo_server.py` — local FastAPI server that streams the full pipeline
  (crop → embed → diffmap → decode → ZK prove → verify → anchor on Sepolia)
  as Server-Sent Events, one JSON event per completed stage. The EZKL
  circuit is compiled once at startup; each request runs a real
  gen_witness + prove + verify (~1 min on CPU).
- `demo.html` — front end that renders a card per stage as it lands.
  Crop stage animates a highlight box onto the real backend crop region of
  the full source photo, then flies the 32×32 crop out as the pipeline
  input; embed stage reveals the real 32-bit payload and cascades it into
  the crop before showing a before/after slider (original vs watermarked);
  anchor stage shows the block number and block hash from the transaction
  receipt (visible for Anvil and Sepolia). All animation is driven by real
  backend data — no fabricated network-internals visuals.
- `run_demo/` — artifacts from a dry run: compiled circuit
  (`network.ezkl`), proving/verifying keys, and a fresh proof
  `foliage_proof.json` over the tea_plantation foliage crop.

## Nothing outstanding for the three-day slice

Out-of-scope items (full robustness sweep, PSNR/SSIM tuning, on-chain proof
verification, proof batching, larger images, perceptual hashing) remain in
Limitations by design — see `CLAUDE.md`.
