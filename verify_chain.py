"""
Quick blockchain integrity verifier — run this during demo to show tamper-detection.
Usage: python verify_chain.py

How it works:
  - hash_registry.json  : written by FL server at aggregation time (trusted ground truth)
  - blockchain_fallback.jsonl : the public ledger (can be tampered to simulate an attack)
  verify_chain.py cross-checks each ledger entry against the registry.
  If an attacker corrupts the ledger, the hash won't match the registry → TAMPER DETECTED.
"""
import sys, json, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from aura.blockchain import AURABlockchainLogger

bc       = AURABlockchainLogger()
ledger   = pathlib.Path("logs/blockchain_fallback.jsonl")
registry = pathlib.Path("logs/hash_registry.json")

print("=" * 60)
print("  AURA Blockchain Integrity Verifier")
print(f"  Mode: {bc.mode.upper()}")
print("=" * 60)

# ── Load trusted registry (ground truth written at FL aggregation) ────────────
if not registry.exists():
    print("[!] Trusted registry not found.")
    print("    --> Run FL Simulation first to generate hash_registry.json")
    sys.exit(0)

trusted: dict = json.loads(registry.read_text())
print(f"  Trusted registry entries : {len(trusted)}")

# ── Load ledger (the file that can be tampered) ────────────────────────────────
if not ledger.exists() or ledger.stat().st_size == 0:
    print("[!] No ledger entries found.")
    sys.exit(0)

entries = [json.loads(l) for l in ledger.read_text().splitlines() if l.strip()]

# Keep only the LATEST entry per version (ledger is append-only; old runs reuse same tags)
latest: dict = {}
for e in entries:
    latest[e["model_version"]] = e   # later entries overwrite earlier ones

# Only verify entries that exist in the trusted registry (FL-generated hashes)
fl_entries = [v for k, v in latest.items() if k in trusted]
print(f"  Ledger entries (FL rounds): {len(fl_entries)}")
print()

if not fl_entries:
    print("[!] No FL-round entries found in ledger to verify.")
    print("    --> Run FL Simulation first, then re-run verify_chain.py")
    sys.exit(0)

all_ok = True
for e in fl_entries:
    version      = e["model_version"]
    ledger_hash  = e["model_hash"]
    trusted_hash = trusted[version]
    ok           = ledger_hash == trusted_hash
    status       = "VERIFIED OK     [+]" if ok else "TAMPER DETECTED [!]"
    mark         = "OK" if ok else "FAIL"
    if not ok:
        all_ok = False
    print(f"  [{mark}]  {version:<20}  {status}")
    print(f"         ledger  = {ledger_hash[:32]}...")
    if not ok:
        print(f"         trusted = {trusted_hash[:32]}...")
    print()

print("=" * 60)
if all_ok:
    print("  RESULT: All model hashes verified. Chain is INTACT.")
else:
    print("  RESULT: TAMPER DETECTED in one or more entries!")
    print("  ACTION: Block deployment. Alert security team.")
print("=" * 60)

