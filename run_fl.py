"""Quick FL simulation runner — avoids PowerShell quoting issues."""
import sys
sys.path.insert(0, ".")

from aura.blockchain import AURABlockchainLogger
from aura.fl_server import run_federation_simulation

results = run_federation_simulation(blockchain_module=AURABlockchainLogger())

print("\n=== ROUND SUMMARY ===")
for r in results:
    ver = r.get("model_version", "?")
    h   = r.get("model_hash", "")
    print(f"  Round {r['round']:>1}: version={ver:<14}  hash={h[:20]}...")
