// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract ProvenanceAnchor {
    mapping(bytes32 => bytes32) public anchors;

    event Anchored(bytes32 indexed assetId, bytes32 proofHash);

    function setAnchor(bytes32 assetId, bytes32 proofHash) external {
        anchors[assetId] = proofHash;
        emit Anchored(assetId, proofHash);
    }

    function getAnchor(bytes32 assetId) external view returns (bytes32) {
        return anchors[assetId];
    }
}
