# Results

Evidence file. One row appended after **every** EZKL run.

| model shape | logrows | peak RAM | prove time | proof size | verify |
|---|---|---|---|---|---|
| smoke test — TinyMLP (8→4→ReLU→2) | 15 | not measured | 1.15s | 18782 B | True |
| probe CNN 32x32x3, conv 8/16/32 s2, GAP, linear 32 | 18 | 418 MB | 14.55s | 493816 B | True |
| probe CNN 32x32x3, conv 16/32/64 s2, GAP, linear 32 | 20 | 254 MB | 56.08s | 493781 B | True |
| probe CNN 32x32x3, conv 32/64/128 s2, GAP, linear 32 | 21 | 50 MB | 241.49s | 493755 B | True |
| probe CNN 64x64x3, conv 8/16/32 s2, GAP, linear 32 | 20 | 219 MB | 61.70s | 1907871 B | True |
| trained decoder (real weights) 32x32x3, conv 16/32/64 s2, GAP, linear 32 | 20 | 50 MB | 57.55s | 492315 B | True |

## On-chain anchoring events

Manual anchor test — assetId is a real photo hash, proofHash is from
day2_export.py's decoder proof (decoder.pt, run on synthetic input to
validate the EZKL pipeline, not tied to this specific photo). Baseline D
below is the row where proof, image, and anchor are all from one coherent
real-photo run.

| date | assetId (image hash) | proofHash (anchored value) | contract address | tx hash | block | gas used | status | getAnchor round-trip |
|---|---|---|---|---|---|---|---|---|
| 2026-09-05 | 0x18f79853237ed3e05be211b4a77ec3f56423339710fe49a089b7bd8c14f9be21 | 0x3a41b30b42bdfc92dda6e1fe9332c9c4b342c5fb12714ab6bf13a8525724e4a0 | 0x5FbDB2315678afecb367f032d93F642f64180aa3 | 0x7477f3399ce4c2c12698901f51805a9a781093a0f99552cbf3b5356e94fefa7a | 3 | 46392 | success | confirmed — returned value matches proofHash exactly (re-verified via chain.py against live Anvil) |

Anchored manually via WSL/Anvil outside the main dev session.

## Sepolia deployment

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

Etherscan: sepolia.etherscan.io/tx/0x3a40613175c85ee90e48e7a228add020efbf1a5cb0f478d90c40a4796bd7bb0a (deploy),
sepolia.etherscan.io/tx/0x17093ad77a5ef49d2284860b04ba3ad7e0053e2d7c600c1976befcc0907ebb33 (setAnchor)

Same assetId/proofHash pair as the manual Anvil anchor above. Gas used for
setAnchor is identical on both chains (46392), confirming portability from
local dev to public testnet as designed.

## Ablation study

| baseline | description | BER | PSNR (dB) | logrows | peak RAM | prove time | proof size | verify | anchor |
|---|---|---|---|---|---|---|---|---|---|
| A | watermark only (no ZK, no anchor) | 0.0000 | 30.23 | n/a | n/a | n/a | n/a | n/a | n/a |
| B | watermark + ZK, no anchor | 0.0000 | 30.23 | 20 | 308 MB | 48.98s | 491871 B | True | n/a |
| C | watermark + anchor, no ZK | 0.0000 | 30.23 | n/a | n/a | n/a | n/a | n/a | tx=0x7edfa63c6ee044465ec5d931723f8310b416fe8eabf7bedf928311f5feacddb3 gas=46392 assetId=0xf7db211e86d08e54bc1895dc829104037b9e03292b71410bef099dd0aea306ba proofHash=0x0a2befe6bcf3c3624b79c81cb5d4e9acb41d0e87b08a35bbea4a5ff2a3abf621 (placeholder, no ZK) |
| D | full pipeline (watermark + ZK + anchor) | 0.0000 | 30.23 | 20 | 379 MB | 54.27s | 491890 B | True | tx=0x31b673e09839b0e57d2e2a9c71b728c675c0b40bb8943e03653145e3ce1d4bb0 gas=29292 assetId=0xf7db211e86d08e54bc1895dc829104037b9e03292b71410bef099dd0aea306ba proofHash=0xf6bc69076978bb3c52ed3cd0701b6b0144dcd8384dc03327b2eb81946dec2d2c |
