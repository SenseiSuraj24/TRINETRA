"""
Quick blockchain integrity verifier — run this during demo to show tamper-detection.
Usage: python verify_chain.py
"""
import sys, json, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from aura.blockchain import AURABlockchainLogger

bc  = AURABlockchainLogger()
log = pathlib.Path("logs/blockchain_fallback.jsonl")

print("=" * 60)
print("  AURA Blockchain Integrity Verifier")
print(f"  Mode: {bc.mode.upper()}")
print("=" * 60)

if not log.exists() or log.stat().st_size == 0:
    print("[!] No blockchain entries found.")
    print("    --> Click 'Run Federation Round' or 'Register Test Hash' first.")
    sys.exit(0)

entries = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
print(f"  Total ledger entries: {len(entries)}\n")

all_ok = True
for e in entries:
    ok     = bc.verify_model(e["model_version"], e["model_hash"])
    status = "VERIFIED OK     [+]" if ok else "TAMPER DETECTED [!]"
    mark   = "OK" if ok else "FAIL"
    if not ok:
        all_ok = False
    print(f"  [{mark}]  {e['model_version']:<20}  {status}")
    print(f"         hash = {e['model_hash'][:32]}...")
    print()

print("=" * 60)
if all_ok:
    print("  RESULT: All model hashes verified. Chain is INTACT.")
else:
    print("  RESULT: TAMPER DETECTED in one or more entries!")
print("=" * 60)
