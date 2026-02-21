// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title  ModelRegistry
 * @author AURA — Team Trinetra (NexJam 2026)
 * @notice Immutable audit registry for federated AI model weight hashes.
 *
 * @dev    ARCHITECTURAL ROLE:
 *         This contract provides NON-REPUDIATION for the AURA federated
 *         learning pipeline.  After each federation round, the central
 *         server aggregates client weight updates and stores the SHA-256
 *         hash of the resulting global model file here.
 *
 *         Before any client organisation deploys a new model update,
 *         their local AURA agent:
 *           1. Computes SHA-256 of the received .pth file locally.
 *           2. Calls verifyHash() on this contract.
 *           3. Only deploys if the hashes match.
 *
 *         A tampered model file changes the hash → verifyHash returns false
 *         → the client REJECTS the update and alerts the analyst.
 *
 * @dev    STORAGE EFFICIENCY:
 *         We store hashes as bytes32 (32 bytes = 256 bits = exact SHA-256 size).
 *         This is gas-optimal on Ethereum.  string keys are used for human-
 *         readable model versioning (e.g., "v2.3").
 *
 * @dev    ACCESS CONTROL:
 *         Only the contract owner (the AURA central server's wallet address)
 *         can register new model hashes.  Read (verification) is public.
 *         In production, multi-sig (e.g., Gnosis Safe) would replace single-owner.
 */
contract ModelRegistry {

    // ─────────────────────────────────────────────────────────────────────────
    // State Variables
    // ─────────────────────────────────────────────────────────────────────────

    /// @notice The address authorised to register model hashes (AURA server)
    address public immutable owner;

    /// @notice modelVersion → modelHash mapping
    /// @dev    bytes32(0) = unregistered hash (zero-value default)
    mapping(string => bytes32) private _modelHashes;

    /// @notice modelVersion → registration timestamp
    mapping(string => uint256) private _registrationTimestamps;

    /// @notice Ordered list of all registered model versions (for auditability)
    string[] public versionHistory;

    // ─────────────────────────────────────────────────────────────────────────
    // Events
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * @notice Emitted whenever a new model version hash is registered.
     * @dev    Indexed modelVersion allows clients to efficiently filter for
     *         specific versions using Web3.py's event log queries.
     */
    event ModelRegistered(
        string indexed modelVersion,
        bytes32        modelHash,
        uint256        timestamp,
        address        registeredBy
    );

    /**
     * @notice Emitted when a client verifies a model hash.
     * @param  success true = hash match (safe to deploy), false = tampered
     */
    event ModelVerified(
        string indexed modelVersion,
        bool           success,
        address        verifiedBy
    );

    // ─────────────────────────────────────────────────────────────────────────
    // Errors (Solidity 0.8+ custom errors — cheaper than revert strings)
    // ─────────────────────────────────────────────────────────────────────────

    error NotOwner(address caller);
    error AlreadyRegistered(string modelVersion);
    error NotRegistered(string modelVersion);
    error EmptyVersion();
    error ZeroHash();

    // ─────────────────────────────────────────────────────────────────────────
    // Modifiers
    // ─────────────────────────────────────────────────────────────────────────

    modifier onlyOwner() {
        if (msg.sender != owner) revert NotOwner(msg.sender);
        _;
    }

    modifier versionNotEmpty(string memory version) {
        if (bytes(version).length == 0) revert EmptyVersion();
        _;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Constructor
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * @param  _owner The Ethereum address of the AURA central server.
     *         On Ganache, this should be accounts[0] (the deployer).
     */
    constructor(address _owner) {
        require(_owner != address(0), "Owner cannot be zero address");
        owner = _owner;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Write Functions (Only Owner)
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * @notice Register the SHA-256 hash of a new global model.
     * @dev    Called by the AURA server (via Web3.py) immediately after
     *         Krum aggregation completes.  Gas cost: ~45,000 (one SSTORE).
     *
     * @param  modelVersion  Human-readable version string, e.g. "v2.3"
     * @param  modelHash     bytes32 SHA-256 of the .pth weight file
     *
     * IMMUTABILITY: Once registered, a version CANNOT be overwritten.
     * This prevents the server from retroactively claiming it deployed
     * a different model than what was actually distributed.
     */
    function registerModel(
        string  memory modelVersion,
        bytes32        modelHash
    )
        external
        onlyOwner
        versionNotEmpty(modelVersion)
    {
        if (modelHash == bytes32(0))              revert ZeroHash();
        if (_modelHashes[modelVersion] != bytes32(0)) revert AlreadyRegistered(modelVersion);

        _modelHashes[modelVersion]           = modelHash;
        _registrationTimestamps[modelVersion] = block.timestamp;
        versionHistory.push(modelVersion);

        emit ModelRegistered(modelVersion, modelHash, block.timestamp, msg.sender);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Read Functions (Public)
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * @notice Retrieve the registered hash for a model version.
     * @param  modelVersion  The version string to query.
     * @return The registered bytes32 hash, or bytes32(0) if not registered.
     */
    function getHash(string memory modelVersion)
        external
        view
        returns (bytes32)
    {
        return _modelHashes[modelVersion];
    }

    /**
     * @notice Verify that a provided hash matches the registered canonical hash.
     * @dev    This is the function called by AURA CLIENTS before deploying
     *         any new model update.  If it returns false, the update is poisoned.
     *
     * @param  modelVersion  The version string to verify against.
     * @param  modelHash     The hash the client computed from its local file.
     * @return True if the hash matches the registered value.
     */
    function verifyHash(
        string  memory modelVersion,
        bytes32        modelHash
    )
        external
        returns (bool)
    {
        if (_modelHashes[modelVersion] == bytes32(0)) revert NotRegistered(modelVersion);

        bool success = (_modelHashes[modelVersion] == modelHash);
        emit ModelVerified(modelVersion, success, msg.sender);
        return success;
    }

    /**
     * @notice Get the number of registered model versions.
     * @return Count of all versions in the audit trail.
     */
    function getVersionCount() external view returns (uint256) {
        return versionHistory.length;
    }

    /**
     * @notice Get full audit record for a version (hash + timestamp).
     * @param  modelVersion The version to query.
     * @return hash_      The registered SHA-256 hash.
     * @return timestamp_ UNIX timestamp of registration.
     */
    function getAuditRecord(string memory modelVersion)
        external
        view
        versionNotEmpty(modelVersion)
        returns (bytes32 hash_, uint256 timestamp_)
    {
        if (_modelHashes[modelVersion] == bytes32(0)) revert NotRegistered(modelVersion);
        return (_modelHashes[modelVersion], _registrationTimestamps[modelVersion]);
    }
}
