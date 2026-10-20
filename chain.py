"""Web3 glue for anchoring proof hashes on ProvenanceAnchor.

Network is selected via the ZKWM_NETWORK env var: "anvil" (default) or
"sepolia". Sepolia requires SEPOLIA_RPC_URL and SEPOLIA_PRIVATE_KEY in the
environment (a .env file in the project root is loaded automatically if
present) -- a real private key must never be hardcoded in source, unlike
Anvil's public, well-known dev key below.
"""
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from web3 import Web3

load_dotenv()

ABI_PATH = Path(__file__).parent / "contracts" / "out" / "ProvenanceAnchor.sol" / "ProvenanceAnchor.json"

ANVIL_RPC_URL = "http://127.0.0.1:8545"
ANVIL_CONTRACT_ADDRESS = "0x5FbDB2315678afecb367f032d93F642f64180aa3"
ANVIL_PRIVATE_KEY = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"

SEPOLIA_CONTRACT_ADDRESS = "0x4Dda995CD649B203b828a97d0b571259923895eF"

NETWORK = os.environ.get("ZKWM_NETWORK", "anvil").lower()

if NETWORK == "sepolia":
    RPC_URL = os.environ["SEPOLIA_RPC_URL"]
    _private_key = os.environ["SEPOLIA_PRIVATE_KEY"]
    CONTRACT_ADDRESS = Web3.to_checksum_address(SEPOLIA_CONTRACT_ADDRESS)
elif NETWORK == "anvil":
    RPC_URL = ANVIL_RPC_URL
    _private_key = ANVIL_PRIVATE_KEY
    CONTRACT_ADDRESS = Web3.to_checksum_address(ANVIL_CONTRACT_ADDRESS)
else:
    raise ValueError(f"unknown ZKWM_NETWORK {NETWORK!r}, expected 'anvil' or 'sepolia'")

w3 = Web3(Web3.HTTPProvider(RPC_URL))
_account = w3.eth.account.from_key(_private_key)
_abi = json.loads(ABI_PATH.read_text())["abi"]
_contract = w3.eth.contract(address=CONTRACT_ADDRESS, abi=_abi)


def _send_anchor(asset_id: bytes, proof_hash: bytes):
    tx = _contract.functions.setAnchor(asset_id, proof_hash).build_transaction({
        "from": _account.address,
        "nonce": w3.eth.get_transaction_count(_account.address),
    })
    signed = _account.sign_transaction(tx)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    return tx_hash, receipt


def anchor_proof(asset_id: bytes, proof_hash: bytes) -> bytes:
    tx_hash, _ = _send_anchor(asset_id, proof_hash)
    return tx_hash


def anchor_proof_receipt(asset_id: bytes, proof_hash: bytes) -> dict:
    """Like anchor_proof, but returns the block the transaction actually
    landed in -- block number and block hash straight off the mined
    receipt -- so a caller (e.g. the demo page) can show that a new block
    was produced, not just that a transaction was sent."""
    tx_hash, receipt = _send_anchor(asset_id, proof_hash)
    return {
        "tx_hash": "0x" + bytes(tx_hash).hex(),
        "block_number": int(receipt.blockNumber),
        "block_hash": "0x" + bytes(receipt.blockHash).hex(),
        "gas_used": int(receipt.gasUsed),
        "status": int(receipt.status),
    }


def get_anchor(asset_id: bytes) -> bytes:
    return _contract.functions.getAnchor(asset_id).call()


if __name__ == "__main__":
    asset_id = bytes.fromhex("18f79853237ed3e05be211b4a77ec3f56423339710fe49a089b7bd8c14f9be21")
    proof_hash = bytes.fromhex("3a41b30b42bdfc92dda6e1fe9332c9c4b342c5fb12714ab6bf13a8525724e4a0")

    print(f"network: {NETWORK}  contract: {CONTRACT_ADDRESS}")
    result = get_anchor(asset_id)
    print("get_anchor ->", result.hex())
    assert result == proof_hash, "mismatch against the recorded anchor value"
    print("MATCH: get_anchor returns the recorded proofHash")
